"""M1 真实训练：四架构128步初筛 + 双种子384步公平复验，全局2048步预算。"""
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

from dataset_m1 import M1Tokenizer, vocab_sha
from candidate_registry import CANDIDATES, build_model, candidate_names
from curriculum_m1 import BalancedSampler, GateTracker, STAGES, stage_for, PANEL_TASKS
from budget_manager import BudgetManager
from evaluate_m1 import (load_rows, build_fixed_panel, make_batch, grouped_loss,
                         evaluate_panel)

CHECK = HERE / "checkpoints"
SEED = 20261124
BS = 16
LR = 0.001
CLIP = 1.0
EVAL_EVERY = 64
PANEL_PER_TASK = 30


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
    if hasattr(model, "use_router_load_update"):
        model.use_router_load_update = True
    if hasattr(model, "router_gradient_path_enabled"):
        model.router_gradient_path_enabled = True


def save_checkpoint(path, model, optimizer, *, step, run_id, architecture, seed,
                    extra=None):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "format": "nmind-math-m1/1",
        "architecture": architecture,
        "config": model.config_dict(),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer_steps_executed": int(step),
        "run_id": run_id,
        "seed": int(seed),
        "tokenizer_sha256": vocab_sha(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "torch_rng_state": torch.get_rng_state(),
        "extra": extra or {},
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
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
    return {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "bytes": p.stat().st_size, "step": step}


def macro_of(ev):
    return ev["macro_accuracy"]


def run_one(name, seed, max_steps, run_id, stage, budget, tok, panel, train_rows):
    torch.manual_seed(seed)
    random.seed(seed)
    model = build_model(name, tok.vocab_size)
    init_sha = state_sha(model)
    set_train_mode(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    py_rng = random.Random(seed)

    rows_by_task = defaultdict(list)
    for r in train_rows:
        rows_by_task[r["task"]].append(r)
    sampler = BalancedSampler(rows_by_task, seed=seed)
    gates = GateTracker()

    ev0 = evaluate_panel(model, tok, panel)
    set_train_mode(model)
    per_task_acc0 = {t: d["accuracy"] for t, d in ev0["per_task"].items()}
    gates.update(per_task_acc0)
    best_score = macro_of(ev0)
    best_step = 0
    best_state = clone_state(model)
    history = [{"step": 0, "val_macro": best_score,
                "val_per_task": {t: d["accuracy"] for t, d in ev0["per_task"].items()},
                "stable": sorted(gates.stable)}]
    checkpoints = {}
    local_step = 0
    stop_reason = "max_steps"
    t0 = time.perf_counter()

    print(json.dumps({"phase": "run_start", "run_id": run_id, "arch": name,
                      "seed": seed, "max_steps": max_steps, "stage": stage,
                      "params": model.n_params(), "state_elements": model.state_elements(),
                      "init_sha256": init_sha}, ensure_ascii=False), flush=True)

    while local_step < max_steps:
        ok, reason = budget.can_step()
        if not ok:
            stop_reason = reason
            break
        stage_cfg = stage_for(local_step, max_steps)
        stage_idx = STAGES.index(stage_cfg)
        focus = stage_cfg["tasks"]
        failing = STAGES[max(0, stage_idx - 1)]["tasks"] if stage_idx > 0 else []
        replay = STAGES[0]["tasks"] if stage_idx > 1 else []
        rows = sampler.sample(focus, failing, replay, BS)
        if not rows:
            raise RuntimeError(f"采样为空: {focus}")

        x, valid, weights = make_batch(tok, rows)
        logits, _ = model(x[:, :-1], valid_mask=valid[:, :-1])
        loss = grouped_loss(logits, x[:, 1:], weights)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss {name} step {local_step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        budget.record_step(run_id, name, seed, local_step + 1, float(loss.detach()))
        local_step += 1

        if local_step % EVAL_EVERY == 0 or local_step == max_steps:
            ev = evaluate_panel(model, tok, panel)
            set_train_mode(model)
            stable = gates.update({t: d["accuracy"] for t, d in ev["per_task"].items()})
            score = macro_of(ev)
            entry = {"step": local_step, "val_macro": score,
                     "val_per_task": {t: d["accuracy"] for t, d in ev["per_task"].items()},
                     "immediate_eos_rate": ev["immediate_eos_rate"],
                     "stable": stable, "stage": stage_cfg["id"],
                     "wall": round(budget.current_wall(), 1)}
            history.append(entry)
            print(json.dumps({"phase": "eval", "run_id": run_id, **entry},
                             ensure_ascii=False), flush=True)
            if score > best_score or (score == best_score and local_step > best_step):
                best_score, best_step = score, local_step
                best_state = clone_state(model)
            last = save_checkpoint(
                CHECK / f"{run_id}_last.pt", model, optimizer,
                step=budget.steps_used, run_id=run_id, architecture=name, seed=seed,
                extra={"role": "last", "stage": stage_cfg["id"]})
            checkpoints["last"] = last

    wall = time.perf_counter() - t0
    final = save_checkpoint(
        CHECK / f"{run_id}_final.pt", model, optimizer,
        step=budget.steps_used, run_id=run_id, architecture=name, seed=seed,
        extra={"role": "final", "stop_reason": stop_reason, "wall": round(wall, 1)})
    checkpoints["final"] = final
    model.load_state_dict(best_state)
    best = save_checkpoint(
        CHECK / f"{run_id}_best.pt", model, None,
        step=best_step, run_id=run_id, architecture=name, seed=seed,
        extra={"role": "best", "selection": "fixed_panel_macro", "score": best_score})
    checkpoints["best"] = best
    budget.record_run_end(run_id, name, seed, local_step, wall, stop_reason, best_score)

    return {
        "run_id": run_id, "architecture": name, "seed": seed, "stage": stage,
        "steps": local_step, "stop_reason": stop_reason, "wall_seconds": round(wall, 1),
        "params": model.n_params(), "state_elements": model.state_elements(),
        "config": model.config_dict(), "init_sha256": init_sha,
        "best_score": best_score, "best_step": best_step,
        "final_score": history[-1]["val_macro"],
        "history": history, "checkpoints": checkpoints,
        "model_weights_uploaded": False,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute-train", action="store_true")
    ap.add_argument("--authorized", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(20)
    tok = M1Tokenizer()
    train = load_rows("train")
    val = load_rows("val")
    sealed = load_rows("sealed")
    panel = build_fixed_panel(val, per_task=PANEL_PER_TASK, seed=SEED)
    budget = BudgetManager()

    data_audit = json.loads((HERE / "M1_DATA_AUDIT.json").read_text(encoding="utf-8"))
    if data_audit.get("status") != "M1_DATA_FROZEN":
        raise RuntimeError("M1 数据未冻结，禁止训练")
    math_tests = json.loads((HERE / "M1_MATH_TESTS.json").read_text(encoding="utf-8"))
    if math_tests.get("status") != "M1_MATH_TESTS_PASS":
        raise RuntimeError("数学验证未通过，禁止训练")
    leakage = json.loads((HERE / "M1_LEAKAGE_AUDIT.json").read_text(encoding="utf-8"))
    if leakage.get("verdict") != "PASS":
        raise RuntimeError("泄漏审计未通过，禁止训练")

    if not (args.execute_train and args.authorized):
        print(json.dumps({"status": "DRY_RUN_NO_TRAIN", "train": len(train),
                          "val": len(val), "sealed": len(sealed), "panel": len(panel),
                          "tokenizer_sha": vocab_sha(), "budget": budget.summary(),
                          "candidates": candidate_names()}, ensure_ascii=False, indent=2))
        return

    budget.start_session()
    stage1 = []
    stage2 = []
    try:
        for name in candidate_names():
            ok, reason = budget.can_step()
            if not ok:
                print(json.dumps({"phase": "budget_stop", "reason": reason}, ensure_ascii=False), flush=True)
                break
            stage1.append(run_one(name, SEED, 128, f"S1_{name}", "stage1", budget,
                                  tok, panel, train))
        new_cands = [r for r in stage1 if r["architecture"] in ("M1-B", "M1-C")]
        ref_cands = [r for r in stage1 if r["architecture"] in ("M0", "M1-A")]
        def pick(rows):
            return sorted(rows, key=lambda r: (r["best_score"], r["steps"]), reverse=True)[0]["architecture"] if rows else None
        best_new, best_ref = pick(new_cands), pick(ref_cands)
        for name in (best_new, best_ref):
            if name is None:
                continue
            for seed in (20261125, 20261126):
                ok, reason = budget.can_step()
                if not ok:
                    print(json.dumps({"phase": "budget_stop", "reason": reason}, ensure_ascii=False), flush=True)
                    break
                stage2.append(run_one(name, seed, 384, f"S2_{name}_s{seed}", "stage2",
                                      budget, tok, panel, train))
        budget.end_session("done")
    except Exception as exc:
        budget.record_uncertain(1, f"{type(exc).__name__}: {exc}"[:200])
        budget.end_session(f"error:{type(exc).__name__}")
        raise

    results = {
        "status": "M1_TRAINING_COMPLETED",
        "tokenizer_sha256": vocab_sha(),
        "fixed_panel": {"n": len(panel), "per_task": PANEL_PER_TASK, "tasks": PANEL_TASKS},
        "budget": budget.summary(),
        "stage1": stage1,
        "selection": {"best_new": best_new, "best_reference": best_ref,
                      "rule": "stage1 fixed-panel macro; tie -> later step"},
        "stage2": stage2,
        "not_run": ["uncertainty_memory_in_main_model", "short_conv_in_main_model",
                    "unbounded_autonomous_learning"],
        "model_weights_uploaded": False,
    }
    (HERE / "M1_TRAINING_RESULTS.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"phase": "done", "steps": budget.steps_used,
                      "wall": round(budget.current_wall(), 1),
                      "best_new": best_new, "best_ref": best_ref,
                      "stage1_scores": {r["architecture"]: r["best_score"] for r in stage1}},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
