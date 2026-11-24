"""C1.1 真实训练主程序：继承 C1.0 best，2048 step / 1800s 共享预算。

本轮修复全部接入训练循环：
- 任务均衡采样 60% 当前重点 / 20% 未达标基础 / 20% 已学回放
- 连续两次评估达标才认定稳定（GateTracker）
- 首 Token 权重 2.0 / 其他内容 1.0 / EOS 0.25，按样本归一化 loss
- padding 不写 Workspace、不更新 Router load（model.forward(valid_mask=...)）
- 固定验证面板选 Best（另存复制支线 Best / 绑定支线 Best）
- 跨重启预算台账；Final 先于 Best 保存；5 个 checkpoint 逐张量核验
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
C1 = HERE.parent / "compute_c1"
sys.path.insert(0, str(C1))
sys.path.insert(0, str(HERE))

from tokenizer_compute import ComputeTokenizer, EOS, PAD, VOCAB_SIZE, vocab_sha
from model_compute import ComputeLM, load_compute, save_compute, model_sha
from evaluate_c11 import build_fixed_panel, evaluate_panel
from budget_manager import BudgetManager
from curriculum_c11 import BalancedSampler, GateTracker, stage_for, STAGES, LINE_A, LINE_B

SEED = 20261124
LR = 0.001
BS = 16
CLIP = 1.0
W_FIRST = 2.0
W_CONTENT = 1.0
W_EOS = 0.25
EVAL_EVERY = 128
PANEL_PER_TASK = 60

PANEL_TASKS = ["A1_copy1", "A2_copyN", "A3_index", "A4_reverse",
               "B2_bind", "B4_bind", "B2_counter", "B5_reassign", "B6_indirect"]
COPY_TASKS = ["A1_copy1", "A2_copyN"]
BIND_TASKS = ["B2_bind", "B4_bind", "B2_counter", "B5_reassign", "B6_indirect"]


def load_rows(split: str) -> list[dict]:
    p = HERE / "data" / f"c11_{split}.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def clone_state(model: torch.nn.Module) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def make_batch(tok: ComputeTokenizer, rows: list[dict]):
    """分段编码 + 加权答案掩码；padding 位置 valid_mask=False、loss 权重=0。"""
    seqs = []
    for r in rows:
        p = tok.encode(r["prompt"])
        a = tok.encode(r["answer"])
        seqs.append(p + a + [EOS])
    n = max(len(s) for s in seqs)
    B = len(rows)

    x = torch.full((B, n), PAD, dtype=torch.long)
    valid = torch.zeros((B, n), dtype=torch.bool)
    weights = torch.zeros((B, n - 1), dtype=torch.float32)

    for j, (r, s) in enumerate(zip(rows, seqs)):
        x[j, :len(s)] = torch.tensor(s, dtype=torch.long)
        valid[j, :len(s)] = True
        p_len = len(tok.encode(r["prompt"]))
        a_len = len(tok.encode(r["answer"]))
        # 目标位置（logits 下标）：p_len-1=首答案；...；p_len+a_len-1=EOS
        for k in range(a_len):
            pos = p_len - 1 + k
            weights[j, pos] = W_FIRST if k == 0 else W_CONTENT
        eos_pos = p_len + a_len - 1
        weights[j, eos_pos] = W_EOS
    return x, valid, weights


def grouped_loss(logits: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """逐 token loss × 权重，按样本权重和归一化，再对 batch 平均。"""
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_target = target.reshape(-1)
    flat_w = weights.reshape(-1)
    token_loss = F.cross_entropy(flat_logits, flat_target, reduction="none", ignore_index=PAD)
    B = weights.size(0)
    per_sample = (token_loss * flat_w).reshape(B, -1).sum(dim=1)
    denom = flat_w.reshape(B, -1).sum(dim=1).clamp_min(1e-6)
    return (per_sample / denom).mean()


def macro_of(ev: dict, tasks: list[str]) -> float:
    per = ev.get("per_task", {})
    vals = [per[t]["accuracy"] for t in tasks if t in per]
    return round(sum(vals) / max(1, len(vals)), 4)


def choose_focus(stage: dict, gates: GateTracker) -> list[str]:
    focus = [t for t in stage["focus"] if t in PANEL_TASKS]
    # 已稳定任务退出当前重点，但仍进入 20% 回放池
    unstable = [t for t in focus if t not in gates.stable]
    focus = unstable or focus
    failing = gates.failing(PANEL_TASKS)
    if stage["id"] == "S4" or not focus:
        focus = failing or PANEL_TASKS
    return focus


def set_train_mode(model: ComputeLM) -> None:
    model.train(True)
    model.use_router_load_update = True
    model.router_gradient_path_enabled = True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute-train", action="store_true")
    ap.add_argument("--authorized", action="store_true")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--max-wall", type=int, default=None)
    args = ap.parse_args()

    torch.set_num_threads(20)
    torch.manual_seed(SEED)
    random.seed(SEED)

    tok = ComputeTokenizer()
    train = load_rows("train")
    val = load_rows("val")
    sealed = load_rows("sealed")
    panel = build_fixed_panel(val, per_task=PANEL_PER_TASK, seed=SEED)

    parent_audit = json.loads((HERE / "C11_PARENT_AUDIT.json").read_text(encoding="utf-8"))
    if parent_audit.get("status") != "PARENT_AUDIT_DONE":
        raise RuntimeError("父模型审计未完成，禁止开始 C1.1 训练")
    parent_name = parent_audit["selected_parent"]
    parent_path = C1 / f"candidate_compute_{parent_name}.pt"

    kwargs = {}
    if args.max_steps is not None:
        kwargs["max_steps"] = args.max_steps
    if args.max_wall is not None:
        kwargs["max_wall"] = args.max_wall
    budget = BudgetManager(**kwargs)

    if not (args.execute_train and args.authorized):
        print(json.dumps({
            "status": "DRY_RUN_NO_TRAIN",
            "parent": parent_name,
            "parent_path": str(parent_path),
            "train": len(train), "val": len(val), "sealed": len(sealed),
            "panel_size": len(panel),
            "tokenizer_sha": vocab_sha(),
            "budget": budget.summary(),
            "loss_weights": {"first": W_FIRST, "content": W_CONTENT, "eos": W_EOS},
        }, ensure_ascii=False, indent=2))
        return

    if not parent_path.exists():
        raise FileNotFoundError(f"父 checkpoint 缺失: {parent_path}")
    ok, reason = budget.can_step()
    if not ok:
        print(json.dumps({"status": "BUDGET_EXHAUSTED_BEFORE_START", "reason": reason,
                          "budget": budget.summary()}, ensure_ascii=False, indent=2))
        return

    model, parent_blob = load_compute(parent_path, tok)
    parent_sha = model_sha(model)
    parent_steps = int(parent_blob.get("optimizer_steps_executed", 0))
    set_train_mode(model)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    rng = random.Random(SEED)

    rows_by_task = defaultdict(list)
    for r in train:
        rows_by_task[r["task"]].append(r)
    sampler = BalancedSampler(rows_by_task, seed=SEED)
    gates = GateTracker()

    # 父模型先过固定面板，作为 step=0 基线与 best 初值
    initial_ev = evaluate_panel(model, tok, panel)
    set_train_mode(model)
    per_task0 = {t: d["accuracy"] for t, d in initial_ev["per_task"].items()}
    gates.update(per_task0)
    best_score = initial_ev["macro_accuracy"]
    copy_score = macro_of(initial_ev, COPY_TASKS)
    bind_score = macro_of(initial_ev, BIND_TASKS)
    best_step = 0
    copy_step = 0
    bind_step = 0
    best_state = clone_state(model)
    copy_state = clone_state(model)
    bind_state = clone_state(model)

    history = [{
        "step": 0, "stage": "parent", "focus": [],
        "val_macro": best_score, "val_copy": copy_score, "val_binding": bind_score,
        "val_per_task": per_task0, "stable": sorted(gates.stable),
        "train_loss_last100": None, "wall": 0.0,
    }]
    stage_log = []
    loss_window: list[float] = []
    last_eval_step = 0
    local_step = 0
    stop_reason = "max_steps"

    print(json.dumps({
        "phase": "train_start", "parent": parent_name, "parent_steps": parent_steps,
        "parent_sha256": parent_sha, "params": model.n_params(),
        "tokenizer_sha": vocab_sha(), "panel_size": len(panel),
        "budget": budget.summary(), "lr": LR, "batch_size": BS,
        "loss_weights": {"first": W_FIRST, "content": W_CONTENT, "eos": W_EOS},
    }, ensure_ascii=False), flush=True)

    budget.start_session()
    try:
        while True:
            ok, reason = budget.can_step()
            if not ok:
                stop_reason = reason
                break

            global_step = budget.steps_used
            stage = stage_for(global_step)
            if not stage_log or stage_log[-1]["id"] != stage["id"]:
                stage_log.append({"id": stage["id"], "start_step": global_step,
                                  "until": stage["until"], "note": stage["note"]})
                print(json.dumps({"phase": "stage", "id": stage["id"],
                                  "start_step": global_step, "tasks": stage["focus"]},
                                 ensure_ascii=False), flush=True)

            focus = choose_focus(stage, gates)
            failing = gates.failing(PANEL_TASKS)
            stable = sorted(gates.stable)
            rows = sampler.sample(focus, failing, stable, BS)
            if not rows:
                raise RuntimeError(f"采样为空: focus={focus}, failing={failing}")

            x, valid, weights = make_batch(tok, rows)
            logits, _ = model(x[:, :-1], valid_mask=valid[:, :-1])
            loss = grouped_loss(logits, x[:, 1:], weights)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at global_step={global_step}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()
            opt.zero_grad(set_to_none=True)

            budget.record_step(local_step + 1, float(loss.detach()))
            local_step += 1
            loss_window.append(float(loss.detach()))
            if len(loss_window) > 100:
                loss_window.pop(0)

            if local_step % 32 == 0:
                print(json.dumps({
                    "phase": "step", "global_step": budget.steps_used,
                    "local_step": local_step, "stage": stage["id"],
                    "loss": round(float(loss.detach()), 5),
                    "loss_last100": round(sum(loss_window) / len(loss_window), 5),
                    "wall": round(budget.current_wall(), 1),
                }, ensure_ascii=False), flush=True)

            if global_step + 1 - last_eval_step >= EVAL_EVERY or budget.steps_used >= budget.max_steps:
                set_train_mode(model)
                ev = evaluate_panel(model, tok, panel)
                set_train_mode(model)
                per_task = {t: d["accuracy"] for t, d in ev["per_task"].items()}
                stable = sorted(gates.update(per_task))
                score = ev["macro_accuracy"]
                c_score = macro_of(ev, COPY_TASKS)
                b_score = macro_of(ev, BIND_TASKS)
                entry = {
                    "step": budget.steps_used, "stage": stage["id"],
                    "focus": focus, "failing": failing,
                    "val_macro": score, "val_copy": c_score, "val_binding": b_score,
                    "val_per_task": per_task, "stable": stable,
                    "immediate_eos_rate": ev["immediate_eos_rate"],
                    "train_loss_last100": round(sum(loss_window) / max(1, len(loss_window)), 5),
                    "wall": round(budget.current_wall(), 1),
                }
                history.append(entry)
                print(json.dumps({"phase": "eval", **entry}, ensure_ascii=False), flush=True)
                last_eval_step = budget.steps_used

                if score > best_score:
                    best_score, best_step = score, budget.steps_used
                    best_state = clone_state(model)
                if c_score > copy_score:
                    copy_score, copy_step = c_score, budget.steps_used
                    copy_state = clone_state(model)
                if b_score > bind_score:
                    bind_score, bind_step = b_score, budget.steps_used
                    bind_state = clone_state(model)

                last_sha = save_compute(
                    HERE / "candidate_c11_last.pt", model, tok,
                    step=budget.steps_used, optimizer=opt, rng=rng.getstate(),
                    extra={"role": "last", "stage": stage["id"],
                           "parent": parent_name, "parent_sha256": parent_sha,
                           "focus": focus, "stable": stable})
                if entry["step"] == 128 or entry["step"] % 512 == 0:
                    print(json.dumps({"phase": "last_saved", "sha": last_sha,
                                      "step": budget.steps_used}, ensure_ascii=False), flush=True)
    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"
        budget.record_uncertain(1, "keyboard_interrupt")
    except Exception as exc:
        stop_reason = f"error:{type(exc).__name__}"
        budget.record_uncertain(1, str(exc)[:200])
        budget.end_session(stop_reason)
        raise
    else:
        budget.end_session(stop_reason)

    wall = round(budget.current_wall(), 1)
    if local_step == 0:
        print(json.dumps({"status": "NO_STEPS_EXECUTED", "budget": budget.summary(),
                          "stop_reason": stop_reason}, ensure_ascii=False), flush=True)
        return

    # Final 在恢复 Best 之前保存（历史教训）
    final_state = clone_state(model)
    final_sha = save_compute(
        HERE / "candidate_c11_final.pt", model, tok,
        step=budget.steps_used, optimizer=opt, rng=rng.getstate(),
        extra={"role": "final", "parent": parent_name, "parent_sha256": parent_sha,
               "stop_reason": stop_reason, "wall_seconds": wall})
    last_sha = save_compute(
        HERE / "candidate_c11_last.pt", model, tok,
        step=budget.steps_used, optimizer=opt, rng=rng.getstate(),
        extra={"role": "last", "parent": parent_name, "parent_sha256": parent_sha,
               "stop_reason": stop_reason, "wall_seconds": wall})

    model.load_state_dict(best_state)
    best_sha = save_compute(
        HERE / "candidate_c11_best.pt", model, tok, step=best_step,
        optimizer=None, rng=None,
        extra={"role": "best", "selection": "fixed_panel_macro_accuracy",
               "selected_score": best_score, "parent": parent_name,
               "parent_sha256": parent_sha})
    model.load_state_dict(copy_state)
    copy_sha = save_compute(
        HERE / "candidate_c11_best_copy.pt", model, tok, step=copy_step,
        optimizer=None, rng=None,
        extra={"role": "best_copy", "selection": "copy_line_macro",
               "selected_score": copy_score, "parent": parent_name,
               "parent_sha256": parent_sha})
    model.load_state_dict(bind_state)
    bind_sha = save_compute(
        HERE / "candidate_c11_best_binding.pt", model, tok, step=bind_step,
        optimizer=None, rng=None,
        extra={"role": "best_binding", "selection": "binding_line_macro",
               "selected_score": bind_score, "parent": parent_name,
               "parent_sha256": parent_sha})

    result = {
        "status": "TRAINING_COMPLETED",
        "seed": SEED,
        "parent": {"name": parent_name, "path": str(parent_path),
                   "steps": parent_steps, "sha256": parent_sha},
        "optimizer_steps_executed": local_step,
        "global_budget_steps": budget.steps_used,
        "stop_reason": stop_reason,
        "wall_seconds": wall,
        "budget": budget.summary(),
        "config": model.cfg.to_dict(),
        "tokenizer_sha256": vocab_sha(),
        "params": model.n_params(),
        "lr": LR, "batch_size": BS, "clip": CLIP,
        "loss_supervision": {"first_answer_token": W_FIRST,
                             "other_answer_tokens": W_CONTENT,
                             "eos": W_EOS,
                             "normalization": "sum(weight*token_loss)/sum(weight) per sample, then batch mean"},
        "padding_policy": "valid_mask=False positions do not write workspace, do not update router load, and carry zero loss weight",
        "sampling_policy": "focus60/failing20/replay20; stable requires two consecutive eval points",
        "fixed_panel": {"size": len(panel), "per_task": PANEL_PER_TASK,
                        "tasks": sorted({r["task"] for r in panel})},
        "selection": {"best": {"score": best_score, "step": best_step, "rule": "fixed_panel_macro_accuracy"},
                      "best_copy": {"score": copy_score, "step": copy_step, "rule": "copy_line_macro"},
                      "best_binding": {"score": bind_score, "step": bind_step, "rule": "binding_line_macro"}},
        "history": history,
        "stage_log": stage_log,
        "checkpoint_sha256": {"best": best_sha, "final": final_sha, "last": last_sha,
                              "best_copy": copy_sha, "best_binding": bind_sha},
        "stable_modified": ["model_compute.forward(valid_mask=...)"],
        "external_models": 0,
        "model_weights_uploaded": False,
    }
    (HERE / "C11_TRAINING_RESULTS.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (HERE / "C11_BUDGET_LEDGER_SUMMARY.json").write_text(
        json.dumps(budget.summary(), ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "phase": "done", "steps": local_step, "global_steps": budget.steps_used,
        "wall": wall, "stop_reason": stop_reason,
        "best": {"score": best_score, "step": best_step},
        "copy": {"score": copy_score, "step": copy_step},
        "binding": {"score": bind_score, "step": bind_step},
        "final_macro": history[-1]["val_macro"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
