"""M1 验收实验 A：极小批次过拟合 + 加长训练 + 多数动作基线。

回答两个可证伪的问题（同一 seed，全部数字来自本次实际运行）：
  A. 梯度通路是否健全：32 条训练轨迹、800 次更新内，训练损失能否压下去、
     训练集动作准确率能否显著超过随机（0.25）。压不下去就是工程 bug，不是"任务难"。
  B. 默认数据规模（96 条）加长训练后，测试集遮挡段准确率能到多少；
     并给出"永远猜同一动作"的多数类基线作参照下限。

用法（仓库根目录）:
    python experiments/2026-11-21_m1_overfit.py            # 全部跑完写 JSON
    python experiments/2026-11-21_m1_overfit.py --quick    # 只跑 A，步数减半
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nmind.config import default_config  # noqa: E402
from nmind.learning import datasets as D  # noqa: E402
from nmind.learning.trainer import evaluate, make_data, train  # noqa: E402

SEED = 20261121


def majority_action_baseline(rows: list[dict]) -> float:
    """永远猜训练集里最常见的动作：无任何状态、无记忆的下限参照。"""
    counts = np.bincount(np.concatenate([r["actions"] for r in rows]), minlength=4)
    p = counts.max() / counts.sum()
    return float(p)


def run(cfg_overrides: dict, epochs: int, *, label: str, eval_interval: int = 5) -> dict:
    cfg = default_config()
    cfg = cfg.model_copy(update={"learning": cfg.learning.model_copy(update=cfg_overrides)})
    t0 = time.perf_counter()
    res = train(cfg, epochs=epochs, seed=SEED, device="cpu", quiet=True,
                eval_interval=eval_interval)
    wall = time.perf_counter() - t0
    # 同 (cfg, seed) 重建同一份数据：dataset_hash 一致即为同一数据集
    task = D.GridTask(goal_visible_steps=4, seed=SEED)
    data = make_data(cfg, task, seed=SEED)
    ev_train = evaluate(res.model, res.probe, data["train"], device="cpu")
    ev_test = evaluate(res.model, res.probe, data["test"], device="cpu")
    return {
        "label": label,
        "overrides": cfg_overrides,
        "epochs": epochs,
        "grad_steps": res.steps,
        "wall_s": round(wall, 2),
        "params_total": res.final["params"]["total"],
        "dataset_hash": data["dataset_hash"],
        "loss_first": round(res.curve["total"][0], 4),
        "loss_last": round(res.curve["total"][-1], 4),
        "non_finite_skipped": res.final["non_finite_steps_skipped"],
        "train": ev_train,
        "test": ev_test,
        "route": res.final["route"],
        "eval_curve_test": [
            {"epoch": e["epoch"],
             "test_acc_all": round(e["test"]["accuracy_all"], 4),
             "test_acc_occ": round(e["test"].get("accuracy_occluded", float("nan")), 4),
             "test_probe": round(e["test"]["probe_accuracy"], 4)}
            for e in res.evals
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="A 减半步数、跳过 B")
    args = ap.parse_args()

    out: dict = {
        "experiment": "M1 验收：过拟合健全性 + 加长训练",
        "seed": SEED,
        "device": "cpu",
        "torch_threads": "min(8, logical/2)=8（doctor 实测）",
        "date": "2026-11-21",
        "runs": [],
    }

    # --- A. 极小批次过拟合：32 条轨迹，800 次更新 ---
    a_epochs = 50 if args.quick else 100
    print(f"[A] 过拟合 {a_epochs} epoch × 8 步 = {a_epochs * 8} 次更新 …", flush=True)
    ra = run({"n_train": 32, "n_test": 32, "batch_size": 16, "n_updates": 8,
              "eval_interval": 10}, a_epochs, label="A_tiny_overfit", eval_interval=10)
    ra["majority_baseline_train"] = None  # 32 条训练集自身（下面补）
    out["runs"].append(ra)
    t = ra["test"]
    print(f"[A] loss {ra['loss_first']} → {ra['loss_last']}  "
          f"train acc={ra['train']['accuracy']:.3f} occ={ra['train'].get('accuracy_occluded', float('nan')):.3f}  "
          f"probe(train)={ra['train']['probe_accuracy']:.3f}  "
          f"test acc={t['accuracy']:.3f} occ={t.get('accuracy_occluded', float('nan')):.3f}  "
          f"用时 {ra['wall_s']}s", flush=True)

    if not args.quick:
        # --- B. 默认数据规模加长训练：96 条，40 epoch × 6 步 = 240 次更新 ---
        print("[B] 标准 96 条 × 40 epoch …", flush=True)
        rb = run({"n_train": 96, "n_test": 32, "batch_size": 16, "n_updates": 6,
                  "eval_interval": 5}, 40, label="B_long_default_data", eval_interval=5)
        out["runs"].append(rb)
        t = rb["test"]
        print(f"[B] loss {rb['loss_first']} → {rb['loss_last']}  "
              f"test acc={t['accuracy']:.3f} occ={t.get('accuracy_occluded', float('nan')):.3f}  "
              f"probe={t['probe_accuracy']:.3f}  用时 {rb['wall_s']}s", flush=True)
        # 多数动作基线（对 B 的测试集）
        cfg = default_config()
        cfg = cfg.model_copy(update={"learning": cfg.learning.model_copy(update={
            "n_train": 96, "n_test": 32, "batch_size": 16, "n_updates": 6, "eval_interval": 5})})
        data = make_data(cfg, D.GridTask(goal_visible_steps=4, seed=SEED), seed=SEED)
        out["majority_action_baseline_test"] = {
            "always_most_common_action": majority_action_baseline(data["test"]),
            "dataset_hash": data["dataset_hash"],
        }
        print(f"[baseline] 永远猜多数动作 = {out['majority_action_baseline_test']['always_most_common_action']:.3f}",
              flush=True)

    dest = ROOT / "experiments" / "2026-11-21_m1_overfit.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"→ {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
