"""M2 真实训练：三阶段共享 4096 step / 3600s 预算。

阶段1 SEP A/B（R1 GRU：legacy vs SEP 协议对照）；
阶段2 六架构初筛（等步数，固定面板）；
阶段3 双种子复验（初筛前 2 名 × 2 seed）。
所有 step 计入同一预算台账，Final 先于 Best 保存，逐张量核验。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from tokenizer_m2 import M2Tokenizer, vocab_sha, SEP
from candidate_registry import build, candidate_names
from curriculum_m2 import BalancedSampler, GateTracker, stage_for, STAGES, PANEL_TASKS
from budget_manager import BudgetManager
from evaluate_m2 import (load_rows, build_fixed_panel, make_batch, grouped_loss,
                         evaluate_panel)

CHECK = HERE / "checkpoints"
SEED = 20261201
BS = 16
LR = 0.001
CLIP = 1.0
EVAL_EVERY = 96
PANEL_PER_TASK = 25


def state_sha(model) -> str:
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def clone_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def set_train_mode(model):
    model.train(True)


def save_checkpoint(path, model, optimizer, *, step, run_id, architecture, seed,
                    proto="sep", extra=None):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "format": "nmind-math-m2/1", "architecture": architecture,
        "protocol": proto, "config": model.config_dict(),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer_steps_executed": int(step), "run_id": run_id,
        "seed": int(seed), "tokenizer_sha256": vocab_sha(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "torch_rng_state": torch.get_rng_state(),
        "extra": extra or {}, "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    tmp = p.with_suffix(p.suffix + ".tmp")
    torch.save(blob, tmp)
    with open(tmp, "r+b") as f:
        os.fsync(f.fileno())
    tmp.replace(p)
    check = torch.load(p, map_location="cpu", weights_only=False)
    for k, v in blob["state_dict"].items():
        if not torch.equal(check["state_dict"][k], v):
            raise RuntimeError(f"checkpoint 张量核验失败: {k}")
    return {"path": str(p.relative_to(HERE)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "bytes": p.stat().st_size, "step": step}


def legacy_make_batch(tok, rows):
    """阶段1 legacy 对照：无 SEP，序列=prompt+answer+EOS。"""
    seqs = []
    for r in rows:
        seqs.append(tok.encode(r["prompt"]) + tok.encode(r["answer"]) + [2])  # EOS=2
    n = max(len(s) for s in seqs); B = len(rows)
    x = torch.full((B, n), 0, dtype=torch.long)
    valid = torch.zeros(B, n, dtype=torch.bool)
    weights = torch.zeros(B, n - 1, dtype=torch.float32)
    for j, r in enumerate(rows):
        s = seqs[j]; x[j, :len(s)] = torch.tensor(s); valid[j, :len(s)] = True
        p_len = len(tok.encode(r["prompt"])); a_len = len(tok.encode(r["answer"]))
        for k in range(a_len):
            weights[j, p_len - 1 + k] = 2.0 if k == 0 else 1.0
        weights[j, p_len + a_len - 1] = 0.25
    return x, valid, weights


def run_one(name, seed, max_steps, run_id, stage, budget, tok, panel, train_rows,
            proto="sep"):
    torch.manual_seed(seed); random.seed(seed)
    model = build(name, tok.vocab_size)
    init_sha = state_sha(model)
    set_train_mode(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    rows_by_task = defaultdict(list)
    for r in train_rows:
        rows_by_task[r["task"]].append(r)
    sampler = BalancedSampler(rows_by_task, seed=seed)
    gates = GateTracker()

    ev0 = evaluate_panel(model, tok, panel)
    set_train_mode(model)
    gates.update({t: d["accuracy"] for t, d in ev0["per_task"].items()})
    best_score = ev0["macro_accuracy"]; best_step = 0
    best_state = clone_state(model)
    history = [{"step": 0, "val_macro": best_score}]
    checkpoints = {}
    local_step = 0
    stop_reason = "max_steps"
    t0 = time.perf_counter()

    print(json.dumps({"phase": "run_start", "run_id": run_id, "arch": name,
                      "seed": seed, "max_steps": max_steps, "stage": stage,
                      "protocol": proto, "params": model.n_params(),
                      "state_elements": model.state_elements(),
                      "init_sha256": init_sha}, ensure_ascii=False), flush=True)

    while local_step < max_steps:
        ok, reason = budget.can_step()
        if not ok:
            stop_reason = reason; break
        stage_cfg = stage_for(local_step, max_steps)
        sidx = STAGES.index(stage_cfg)
        focus = stage_cfg["tasks"]
        failing = STAGES[max(0, sidx - 1)]["tasks"] if sidx > 0 else []
        replay = STAGES[0]["tasks"] if sidx > 1 else []
        rows = sampler.sample(focus, failing, replay, BS)
        if not rows:
            raise RuntimeError(f"采样为空 {focus}")

        if proto == "legacy":
            x, valid, weights = legacy_make_batch(tok, rows)
        else:
            x, valid, weights = make_batch(tok, rows)
        logits, _ = model(x[:, :-1], valid_mask=valid[:, :-1])
        loss = grouped_loss(logits, x[:, 1:], weights)
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite loss {name} step {local_step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        budget.record_step(run_id, name, seed, local_step + 1, float(loss.detach()))
        local_step += 1

        if local_step % EVAL_EVERY == 0 or local_step == max_steps:
            ev = evaluate_panel(model, tok, panel)
            set_train_mode(model)
            score = ev["macro_accuracy"]
            history.append({"step": local_step, "val_macro": score,
                            "val_per_task": {t: d["accuracy"] for t, d in ev["per_task"].items()},
                            "immediate_eos_rate": ev["immediate_eos_rate"],
                            "overlong_rate": ev["overlong_rate"],
                            "stage": stage_cfg["id"], "wall": round(budget.current_wall(), 1)})
            print(json.dumps({"phase": "eval", "run_id": run_id, "step": local_step,
                              "val_macro": score, "wall": round(budget.current_wall(), 1)},
                             ensure_ascii=False), flush=True)
            if score > best_score or (score == best_score and local_step > best_step):
                best_score, best_step = score, local_step
                best_state = clone_state(model)
            last = save_checkpoint(CHECK / f"{run_id}_last.pt", model, optimizer,
                                   step=budget.steps_used, run_id=run_id,
                                   architecture=name, seed=seed, proto=proto,
                                   extra={"role": "last", "stage": stage_cfg["id"]})
            checkpoints["last"] = last

    wall = time.perf_counter() - t0
    final = save_checkpoint(CHECK / f"{run_id}_final.pt", model, optimizer,
                            step=budget.steps_used, run_id=run_id,
                            architecture=name, seed=seed, proto=proto,
                            extra={"role": "final", "stop_reason": stop_reason,
                                   "wall": round(wall, 1)})
    checkpoints["final"] = final
    model.load_state_dict(best_state)
    best = save_checkpoint(CHECK / f"{run_id}_best.pt", model, None,
                           step=best_step, run_id=run_id, architecture=name,
                           seed=seed, proto=proto,
                           extra={"role": "best", "selection": "fixed_panel_macro",
                                  "score": best_score})
    checkpoints["best"] = best
    budget.record_run_end(run_id, name, seed, local_step, wall, stop_reason, best_score)
    return {"run_id": run_id, "architecture": name, "seed": seed, "stage": stage,
            "protocol": proto, "steps": local_step, "stop_reason": stop_reason,
            "wall_seconds": round(wall, 1), "params": model.n_params(),
            "state_elements": model.state_elements(), "config": model.config_dict(),
            "init_sha256": init_sha, "best_score": best_score, "best_step": best_step,
            "final_score": history[-1]["val_macro"], "history": history,
            "checkpoints": checkpoints, "model_weights_uploaded": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute-train", action="store_true")
    ap.add_argument("--authorized", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(20)
    tok = M2Tokenizer()
    train = load_rows("train"); val = load_rows("val"); sealed = load_rows("sealed")
    panel = build_fixed_panel(val, per_task=PANEL_PER_TASK, seed=SEED)
    budget = BudgetManager()

    # 冻结门槛
    for fname, key, val_expected in (
        ("M2_DATA_AUDIT.json", "status", "M2_DATA_FROZEN"),
        ("M2_MATH_TESTS.json", "status", "M2_MATH_TESTS_PASS"),
        ("M2_LEAKAGE_AUDIT.json", "verdict", "PASS"),
    ):
        blob = json.loads((HERE / fname).read_text(encoding="utf-8"))
        if blob.get(key) != val_expected:
            raise RuntimeError(f"{fname} 未通过门槛，禁止训练")

    if not (args.execute_train and args.authorized):
        print(json.dumps({"status": "DRY_RUN_NO_TRAIN", "train": len(train),
                          "val": len(val), "sealed": len(sealed), "panel": len(panel),
                          "tokenizer_sha": vocab_sha(), "sep": SEP,
                          "budget": budget.summary(), "candidates": candidate_names()},
                         ensure_ascii=False, indent=2))
        return

    budget.start_session()
    sep_ab = []; screen = []; recheck = []
    selected = []
    try:
        # 阶段1：SEP A/B（R1，legacy vs sep），各 256 步
        for proto in ("legacy", "sep"):
            ok, reason = budget.can_step()
            if not ok:
                print(json.dumps({"phase": "budget_stop", "reason": reason}), flush=True)
                break
            sep_ab.append(run_one("R1", SEED, 256, f"S1_R1_{proto}", "sep_ab",
                                  budget, tok, panel, train, proto=proto))

        # 阶段2：六架构初筛，各 192 步（SEP 协议）
        for name in candidate_names():
            ok, reason = budget.can_step()
            if not ok:
                print(json.dumps({"phase": "budget_stop", "reason": reason}), flush=True)
                break
            screen.append(run_one(name, SEED, 192, f"S2_{name}", "screen",
                                  budget, tok, panel, train, proto="sep"))
        # 选前 2（新架构优先，但允许参考入围）
        ranked = sorted(screen, key=lambda r: (r["best_score"], r["steps"]), reverse=True)
        selected = [r["architecture"] for r in ranked[:2]]

        # 阶段3：双种子复验，2 架构 × 2 seed × 剩余均分
        remaining = budget.max_steps - budget.steps_used
        per_run = max(64, remaining // max(1, len(selected) * 2))
        for name in selected:
            for seed in (20261202, 20261203):
                ok, reason = budget.can_step()
                if not ok:
                    break
                steps = min(per_run, budget.max_steps - budget.steps_used)
                if steps < 32:
                    break
                recheck.append(run_one(name, seed, steps, f"S3_{name}_s{seed}",
                                       "recheck", budget, tok, panel, train, proto="sep"))
        budget.end_session("done")
    except Exception as exc:
        budget.record_uncertain(1, f"{type(exc).__name__}: {exc}"[:200])
        budget.end_session(f"error:{type(exc).__name__}")
        raise

    results = {
        "status": "M2_TRAINING_COMPLETED", "tokenizer_sha256": vocab_sha(),
        "fixed_panel": {"n": len(panel), "per_task": PANEL_PER_TASK, "tasks": PANEL_TASKS},
        "budget": budget.summary(),
        "stage1_sep_ab": sep_ab,
        "stage2_screen": screen,
        "selection": {"selected": selected,
                      "rule": "screen fixed-panel macro; tie -> later step"},
        "stage3_recheck": recheck,
        "not_run": ["uncertainty_memory_in_main_model", "full_dnc_allocation"],
        "model_weights_uploaded": False,
    }
    (HERE / "M2_TRAINING_RESULTS.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"phase": "done", "steps": budget.steps_used,
                      "wall": round(budget.current_wall(), 1),
                      "sep_ab": {r["protocol"]: r["best_score"] for r in sep_ab},
                      "screen_scores": {r["architecture"]: r["best_score"] for r in screen},
                      "selected": selected}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
