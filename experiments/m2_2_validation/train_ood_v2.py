"""H3b/H4b cued v2 小规模同口径重训。

这是修正数据定义后的首轮可运行基线，不替代完整 10-seed 验收。默认每个任务运行
A/C/F10 × 3 seeds × 240 updates，所有权重随机初始化、CPU、无预训练权重。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from nmind.config import default_config  # noqa: E402
from nmind.learning import ood_layouts as O  # noqa: E402
from nmind.learning.trainer import _assemble, evaluate, train  # noqa: E402

HERE = Path(__file__).resolve().parent
SEEDS = (20261201, 20261202, 20261203)
ARMS = {
    "A_dense_ce": {"execution": "dense", "use_router": False, "loss_mode": "ce"},
    "C_sparse_ce": {"execution": "sparse", "use_router": True, "loss_mode": "ce"},
    "F10_ste": {"execution": "sparse", "use_router": True, "loss_mode": "joint",
                 "router_ste": True,
                 "loss_overrides": {"detach_pred_target": 1.0,
                                    "balance": 0.02, "entropy": 0.002}},
}


def combine(parts: list[dict], split_name: str) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    hashes: list[str] = []
    for part in parts:
        rows.extend(part["rows"])
        hashes.append(part["manifest"]["manifest_hash"])
    return rows, hashes


def data_for(task_name: str) -> tuple[dict, int, dict]:
    if task_name == "h3b":
        train_parts = [O.make_h3b_split("dev", n=32, seed=s) for s in range(3001, 3009)]
        val_parts = [O.make_h3b_split("dev", n=32, seed=s) for s in (4001, 4002)]
        test_parts = [O.make_h3b_split("holdout", n=32, seed=s) for s in (5001, 5002)]
        obs_dim = O.OBS_DIM_WITH_MAP
    else:
        train_parts = [O.make_h4b_cued_split("dev", n=32, seed=s) for s in range(6001, 6009)]
        val_parts = [O.make_h4b_cued_split("dev", n=32, seed=s) for s in (7001, 7002)]
        test_parts = [O.make_h4b_cued_split("holdout", n=32, seed=s) for s in (8001, 8002)]
        obs_dim = O.OBS_DIM_WITH_RULE_CUE
    train_rows, train_hashes = combine(train_parts, "train")
    val_rows, val_hashes = combine(val_parts, "val")
    test_rows, test_hashes = combine(test_parts, "test")
    all_hashes = train_hashes + val_hashes + test_hashes
    fingerprint = hashlib.sha256("|".join(all_hashes).encode()).hexdigest()[:16]
    data = {"train": train_rows, "val": val_rows, "test": test_rows,
            "dataset_hash": fingerprint, "split_version": f"{task_name}-v2-multi-layout"}
    audit = {"train_n": len(train_rows), "val_n": len(val_rows), "test_n": len(test_rows),
             "train_manifests": train_hashes, "val_manifests": val_hashes,
             "test_manifests": test_hashes}
    return data, obs_dim, audit


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=("h3b", "h4b"), required=True)
    ap.add_argument("--updates", type=int, default=240)
    args = ap.parse_args()
    data, obs_dim, audit = data_for(args.task)
    cfg = default_config()
    cfg = cfg.model_copy(update={
        "model": cfg.model.model_copy(update={"event_numeric_dim": obs_dim}),
        "learning": cfg.learning.model_copy(update={"traj_len": 24 if args.task == "h3b" else 16,
                                                      "eval_interval": 40})})
    epochs = max(1, args.updates // cfg.learning.n_updates)
    runs = []
    for seed in SEEDS:
        for arm, flags in ARMS.items():
            started = time.perf_counter()
            result = train(cfg, epochs=epochs, seed=seed, device="cpu", quiet=True,
                           data_override=data, **flags)
            test = result.final["eval"]
            row = {"task": args.task, "arm": arm, "seed": seed,
                   "accuracy_occluded": test["accuracy_occluded"],
                   "accuracy_all": result.final["eval_all"]["accuracy"],
                   "wall_s": round(time.perf_counter() - started, 2)}
            if args.task == "h3b":
                for family in O.FAMILIES:
                    subset = [r for r in data["test"] if r["task"] == f"h3b_{family}"]
                    row[f"{family}_accuracy_occluded"] = evaluate(
                        result.model, None, subset, device=torch.device("cpu"),
                        occluded_only=True)["accuracy_occluded"]
            else:
                normal = [r for r in data["test"] if r["task"] == "h4b_normal"]
                reversed_rows = [r for r in data["test"] if r["task"] == "h4b_reversed"]
                row["normal_accuracy_occluded"] = evaluate(
                    result.model, None, normal, device=torch.device("cpu"),
                    occluded_only=True)["accuracy_occluded"]
                row["reversed_accuracy_occluded"] = evaluate(
                    result.model, None, reversed_rows, device=torch.device("cpu"),
                    occluded_only=True)["accuracy_occluded"]
                batch = _assemble(reversed_rows, torch.device("cpu"))
                with torch.no_grad():
                    pred = result.model.forward_sequence(
                        batch["obs"], batch["actions"], mem=batch["mem"]
                    )["action_logits"].argmax(-1)
                mask = torch.zeros_like(batch["actions"], dtype=torch.bool)
                for i, sample in enumerate(reversed_rows):
                    goal = tuple(sample["goal"])
                    for t in range(4, sample["len"]):
                        pos = O.decode_pos(sample["obs"][t])
                        normal_action = O.oracle_action(
                            pos, goal, tuple(), 6, reversed_control=False)
                        mask[i, t] = normal_action != int(sample["actions"][t])
                row["reversed_conflict_accuracy"] = float(
                    (pred[mask] == batch["actions"][mask]).float().mean())
            runs.append(row)
            print(f"[{args.task} {arm} s{seed}] occ={test['accuracy_occluded']:.3f}", flush=True)
    summary = {}
    for arm in ARMS:
        values = [r["accuracy_occluded"] for r in runs if r["arm"] == arm]
        summary[arm] = {"mean": round(statistics.mean(values), 4),
                        "std_ddof1": round(statistics.stdev(values), 4),
                        "per_seed": values}
    payload = {"experiment": f"{args.task} v2 corrected-input retrain smoke",
               "device": "cpu", "updates": args.updates, "seeds": list(SEEDS),
               "obs_dim": obs_dim, "data_audit": audit, "arms": ARMS,
               "runs": runs, "summary": summary,
               "claim_limit": "3-seed smoke；完成 10-seed 与闭环评测前不得升级 ACCEPTED"}
    dest = HERE / f"{args.task}_v2_retrain_smoke.json"
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"→ {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
