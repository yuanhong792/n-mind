"""C1.0 真实训练主程序（2048 optimizer.step / 1800s 硬预算）。

工程修复：
- eval 后恢复原 train 模式与 router 统计开关（历史 bug）
- 完整答案掩码含 EOS；loss 只作用于答案区
- Final 先于 Best 保存；定期 checkpoint；优化器/RNG 状态；逐张量核验
- 实时步数/墙钟输出；到达预算前预留保存时间
- 课程分阶段：C0(<=256) → C2(<=768) → C3(<=1536) → 对照/高级(<=2048)
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import torch

from tokenizer_compute import ComputeTokenizer, EOS, PAD, VOCAB_SIZE, vocab_sha
from model_compute import ComputeConfig, ComputeLM, GRUReference, save_compute, model_sha
from evaluate_compute import evaluate, evaluate_counterfactual, router_audit, router_gradient_check

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
SEED = 20261124
MAX_STEPS = 2048
MAX_WALL = 1800
SAVE_RESERVE = 45          # 预留保存时间
LR = 0.001
BS = 16
CLIP = 1.0

STAGES = [
    {"name": "C0_foundation", "until": 256,
     "tasks": ["C0_copy1", "C0_copyN", "C0_reverse", "C0_index", "C0_next"]},
    {"name": "C2_binding", "until": 768,
     "tasks": ["C0_copyN", "C0_index", "C2_bind1", "C2_bindN", "C2_counter",
               "C2_reassign", "C2_indirect"]},
    {"name": "C3_math", "until": 1536,
     "tasks": ["C0_copyN", "C0_index", "C2_bind1", "C2_bindN", "C2_counter",
               "C2_reassign", "C3_arith", "C3_var_arith"]},
    {"name": "C3_consolidate", "until": 2048,
     "tasks": ["C0_copyN", "C0_reverse", "C0_index", "C2_bindN", "C2_counter",
               "C2_reassign", "C3_arith", "C3_var_arith"]},
]


def load_rows(split):
    return [json.loads(l) for l in (DATA / f"c1_{split}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


def make_batch(tok, rows, idxs):
    """分段编码 + 答案区掩码（含 EOS），不依赖任何外部答案。"""
    seqs = [tok.encode(rows[i]["prompt"]) + tok.encode(rows[i]["answer"]) + [EOS] for i in idxs]
    n = max(len(s) for s in seqs)
    x = torch.zeros((len(idxs), n), dtype=torch.long)
    mask = torch.zeros((len(idxs), n - 1), dtype=torch.bool)
    for j, s in enumerate(seqs):
        x[j, :len(s)] = torch.tensor(s)
        p_start = len(tok.encode(rows[idxs[j]]["prompt"])) - 1
        for p in range(p_start, len(s) - 1):
            mask[j, p] = True
    return x, mask


def pool_indices(rows, tasks):
    return [i for i, r in enumerate(rows) if r["task"] in tasks]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute-train", action="store_true")
    ap.add_argument("--authorized", action="store_true")
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--max-wall", type=int, default=MAX_WALL)
    args = ap.parse_args()
    torch.set_num_threads(20)
    torch.manual_seed(SEED)
    random.seed(SEED)

    train = load_rows("train")
    val = load_rows("val")
    sealed = load_rows("sealed")
    pairs = [json.loads(l) for l in (DATA / "c1_counterfactual.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    tok = ComputeTokenizer()
    assert tok.vocab_size == VOCAB_SIZE

    if not (args.execute_train and args.authorized):
        print(json.dumps({"status": "DRY_RUN_NO_TRAIN", "train": len(train),
                          "val": len(val), "sealed": len(sealed),
                          "tokenizer_sha": vocab_sha()[:16]}, ensure_ascii=False))
        return

    cfg = ComputeConfig(vocab_size=tok.vocab_size)
    model = ComputeLM(cfg)
    init_sha = model_sha(model)
    model.train()
    model.use_router_load_update = True
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    rng = random.Random(SEED)
    t0 = time.perf_counter()
    step = 0
    history = []
    best_score = -1.0
    best_step = 0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    stop_reason = "max_steps"
    stage_log = []
    last_eval_step = -1000
    loss_window = []

    print(json.dumps({"phase": "train_start", "params": model.n_params(),
                      "max_steps": args.max_steps, "max_wall": args.max_wall,
                      "tokenizer_sha": vocab_sha()}, ensure_ascii=False), flush=True)

    while step < args.max_steps:
        elapsed = time.perf_counter() - t0
        if elapsed > args.max_wall - SAVE_RESERVE:
            stop_reason = "wall_budget"; break
        stage = next((s for s in STAGES if step < s["until"]), STAGES[-1])
        if not stage_log or stage_log[-1]["name"] != stage["name"]:
            stage_log.append({"name": stage["name"], "start_step": step,
                              "tasks": stage["tasks"]})
            print(json.dumps({"phase": "stage", "name": stage["name"],
                              "step": step, "tasks": stage["tasks"]}, ensure_ascii=False), flush=True)
        idx_pool = pool_indices(train, stage["tasks"])
        if not idx_pool:
            idx_pool = list(range(len(train)))
        idxs = [rng.choice(idx_pool) for _ in range(BS)]
        x, mask = make_batch(tok, train, idxs)
        logits, _ = model(x[:, :-1])
        flat_logits = logits[mask].reshape(-1, logits.size(-1))
        target = x[:, 1:][mask].reshape(-1)
        loss = torch.nn.functional.cross_entropy(flat_logits, target)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step(); opt.zero_grad(set_to_none=True)
        step += 1
        loss_window.append(float(loss.detach()))
        if len(loss_window) > 100:
            loss_window.pop(0)

        # 每 128 步验证（VAL，不用 SEALED 选模）
        # 关键修复：只用「当前阶段已纳入训练的任务」构造评测子集，
        # 否则未训练的 C2/C3 会把 overall acc 拉成 0，掩盖 C0 真实进步，
        # 也会让 best checkpoint 选择失真。
        if step % 128 == 0 or step == 1:
            taught = set(stage["tasks"])
            # 累积已教过的任务（进入新阶段后仍保留回放任务）
            taught_accum = set()
            for sg in STAGES:
                if step >= sg["until"] or sg["name"] == stage["name"]:
                    taught_accum.update(sg["tasks"])
            val_subset = [r for r in val if r["task"] in (taught_accum or taught)]
            val_ev = evaluate(model, tok, val_subset, limit=192, free_n=96,
                              device=torch.device("cpu"))
            acc = val_ev["overall_accuracy"]
            per_task_acc = {t: d["free_generation_accuracy"]
                            for t, d in val_ev["per_task"].items()}
            entry = {"step": step, "stage": stage["name"],
                     "val_taught_acc": acc, "val_per_task": per_task_acc,
                     "eval_tasks": sorted(taught_accum),
                     "train_loss_last100": round(sum(loss_window)/max(1,len(loss_window)), 5),
                     "wall": round(time.perf_counter() - t0, 1)}
            history.append(entry)
            print(json.dumps({"phase": "eval", **entry}, ensure_ascii=False), flush=True)
            last_eval_step = step
            if acc > best_score:
                best_score = acc; best_step = step
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            # 定期 checkpoint（最近有效权重）
            save_compute(HERE / "candidate_compute_last.pt", model, tok, step=step,
                         optimizer=opt, rng=rng.getstate(),
                         extra={"stage": stage["name"], "init_sha256": init_sha})
            # 早停：仅在训练 loss 长期停滞 且 已教任务 acc 长期为 0 时触发，
            # 且至少训练到 1024 步（保证课程覆盖 C0+C2+C3 主干）。
            recent_loss = [h["train_loss_last100"] for h in history[-4:]]
            recent_acc = [h["val_taught_acc"] for h in history[-4:]]
            if (len(recent_loss) == 4 and step >= 1024
                    and max(recent_acc) == 0.0
                    and max(recent_loss) - min(recent_loss) < 0.01):
                stop_reason = "no_improvement"; break

    wall = time.perf_counter() - t0
    # Final 在恢复 Best 之前保存（历史教训）
    final_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    final_sha = save_compute(HERE / "candidate_compute_final.pt", model, tok, step=step,
                             optimizer=opt, rng=rng.getstate(),
                             extra={"role": "final", "init_sha256": init_sha})
    model.load_state_dict(best_state)
    best_sha = save_compute(HERE / "candidate_compute_best.pt", model, tok, step=best_step,
                            optimizer=None, rng=None,
                            extra={"role": "best", "init_sha256": init_sha})
    # Best 权重上的封存评测（只评一次）
    sealed_ev = evaluate(model, tok, sealed, device=torch.device("cpu"))
    cf_ev = evaluate_counterfactual(model, tok, pairs)
    r_audit = router_audit(model, tok, val)
    r_grad = router_gradient_check(model, tok, train)

    result = {
        "status": "TRAINING_COMPLETED", "seed": SEED,
        "optimizer_steps_executed": step, "best_step": best_step,
        "wall_seconds": round(wall, 1), "stop_reason": stop_reason,
        "config": cfg.to_dict(), "tokenizer_sha256": vocab_sha(),
        "init_model_sha256": init_sha,
        "params_nmind": model.n_params(),
        "batch_size": BS, "lr": LR, "clip": CLIP,
        "train_pool": len(train), "val_pool": len(val), "sealed_pool": len(sealed),
        "best_val_accuracy": best_score,
        "history": history, "stage_log": stage_log,
        "sealed_evaluation": sealed_ev, "counterfactual": cf_ev,
        "router_audit": r_audit, "router_gradient": r_grad,
        "checkpoint_sha256": {"final": final_sha, "best": best_sha},
        "stable_modified": False, "external_models": 0,
        "model_weights_uploaded": False,
    }
    (HERE / "C1_TRAINING_RESULTS.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (HERE / "C1_GENERATION.jsonl").write_text(
        "\n".join(json.dumps(d, ensure_ascii=False) for d in sealed_ev["details"]) + "\n",
        encoding="utf-8")
    print(json.dumps({"phase": "done", "steps": step, "best_step": best_step,
                      "wall": round(wall, 1), "sealed_acc": sealed_ev["overall_accuracy"],
                      "cf": cf_ev["paired_accuracy"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
