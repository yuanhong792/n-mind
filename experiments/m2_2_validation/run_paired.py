"""M2.2-A 配对 seed 定案运行器：C / F1 / F10 / A 同 seed 配对 + 配对统计。

复用 2026-11-21_m2_2x2.py 的 run_arm（同一条训练/评测代码路径），
F10 通过 flags 传 router_ste=True（F9/F10 实验引入的机制开关）。

预登记：experiments/m2_2_validation/paired_seeds_preregister.json
统计口径：std 为 ddof=1（样本）；配对差值 C−F10 逐 seed 报告；
bootstrap 95% CI 以 seed 为重抽样单位（10000 次）。

用法：
  python -u experiments/m2_2_validation/run_paired.py
"""

from __future__ import annotations

import csv
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "m2_runner", ROOT / "experiments" / "2026-11-21_m2_2x2.py")
_runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_runner)

HERE = Path(__file__).resolve().parent
OUT = HERE
CKPT_DIR = ROOT / "experiments" / "m2_ablation"

PRE = json.loads((HERE / "paired_seeds_preregister.json").read_text(encoding="utf-8"))
SEEDS = PRE["seeds_primary"]
STEPS = PRE["steps"]

# (arm 名, flags, loss_overrides, 说明)
ARMS = [
    ("A_dense_ce", {"execution": "dense", "use_router": False, "loss_mode": "ce"}, None),
    ("C_sparse_ce", {"execution": "sparse", "use_router": True, "loss_mode": "ce"}, None),
    ("F1_detach", {"execution": "sparse", "use_router": True, "loss_mode": "joint"},
     {"detach_pred_target": 1.0}),
    ("F10_ste", {"execution": "sparse", "use_router": True, "loss_mode": "joint",
                 "router_ste": True},
     {"detach_pred_target": 1.0, "balance": 0.02, "entropy": 0.002}),
]


def bootstrap_paired_ci(diff: list[float], *, n_boot: int = 10000, seed: int = 1234) -> dict:
    """配对差值的 bootstrap 95% CI：重抽样单位 = seed。"""
    rng = np.random.default_rng(seed)
    diffs = np.array(diff, dtype=np.float64)
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(diffs), size=len(diffs))
        means[i] = diffs[idx].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"point_mean": round(float(diffs.mean()), 4),
            "ci95_low": round(float(lo), 4), "ci95_high": round(float(hi), 4),
            "n_seeds": len(diffs), "unit": "seed(重抽样)"}


def main() -> int:
    cfg = _runner.default_config()
    cfg.learning.epochs = max(1, STEPS // cfg.learning.n_updates)
    cfg.learning.traj_len = 24
    cfg.learning.eval_interval = 10
    cfg.runtime.device = "cpu"

    data = _runner.make_data(cfg, _runner.D.GridTask(goal_visible_steps=4, seed=SEEDS[0]),
                             seed=SEEDS[0])
    print(f"dataset_hash={data['dataset_hash']} split={data.get('split_version')} "
          f"train={len(data['train'])} val={len(data['val'])} test={len(data['test'])}", flush=True)

    rows: list[dict] = []
    for seed in SEEDS:
        for name, flags, overrides in ARMS:
            t0 = time.perf_counter()
            row = _runner.run_arm(name, dict(flags), cfg=cfg, seed=seed, device="cpu",
                                  out_dir=CKPT_DIR, T=24, loss_overrides=overrides)
            rows.append(row)
            print(f"[{name} s{seed}] occ={row['acc_occ']} all={row['acc_all']} "
                  f"probe_te={row['probe_posthoc_test']} ({time.perf_counter() - t0:.0f}s)",
                  flush=True)
            # 每臂完成即写盘（防中断丢全部）
            (HERE / "f10_paired_seeds_partial.json").write_text(
                json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 配对统计 ----
    by: dict[str, dict[int, dict]] = {}
    for r in rows:
        by.setdefault(r["arm"], {})[r["seed"]] = r

    def series(arm: str, key: str) -> list[float]:
        return [float(by[arm][s][key]) for s in SEEDS]

    def stats(arm: str, key: str = "acc_occ") -> dict:
        v = series(arm, key)
        return {f"{key}_mean": round(statistics.mean(v), 4),
                f"{key}_std_ddof1": round(statistics.stdev(v), 4) if len(v) > 1 else 0.0,
                f"{key}_worst_seed": min(v), "worst_seed_id": SEEDS[v.index(min(v))],
                "per_seed": {str(s): round(x, 4) for s, x in zip(SEEDS, v)}}

    paired = {}
    if all(a in by for a in ("C_sparse_ce", "F10_ste")):
        diff_occ = [by["C_sparse_ce"][s]["acc_occ"] - by["F10_ste"][s]["acc_occ"] for s in SEEDS]
        paired["C_minus_F10_acc_occ"] = {
            "per_seed": {str(s): round(d, 4) for s, d in zip(SEEDS, diff_occ)},
            **bootstrap_paired_ci(diff_occ)}
        diff_f1 = [by["C_sparse_ce"][s]["acc_occ"] - by["F1_detach"][s]["acc_occ"] for s in SEEDS]
        paired["C_minus_F1_acc_occ"] = {
            "per_seed": {str(s): round(d, 4) for s, d in zip(SEEDS, diff_f1)},
            **bootstrap_paired_ci(diff_f1)}

    gate = {
        "rule": "沿用冻结的 ≤3pp 平均差距门槛（C−F10 ≤ 0.03）；均值达标不自动 ACCEPTED",
        "C_minus_F10_mean_pp": round(paired["C_minus_F10_acc_occ"]["point_mean"] * 100, 2),
        "ci95_pp": [paired["C_minus_F10_acc_occ"]["ci95_low"] * 100,
                    paired["C_minus_F10_acc_occ"]["ci95_high"] * 100],
    }

    payload = {
        "experiment": "F10 配对 seed 定案",
        "preregister": str(HERE / "paired_seeds_preregister.json"),
        "steps": STEPS, "seeds": SEEDS, "device": "cpu",
        "dataset_hash": data["dataset_hash"], "split_version": data.get("split_version"),
        "arms": {name: {"flags": flags, "loss_overrides": overrides}
                 for name, flags, overrides in ARMS},
        "runs": rows,
        "summary": {name: stats(name) for name, _, _ in ARMS},
        "paired": paired, "gate_3pp": gate,
        "honesty": "std 为 ddof=1；bootstrap 重抽样单位=seed；禁止用单 seed 最优值下结论",
    }
    dest = HERE / "f10_paired_seeds.json"
    tmp = dest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(dest)
    (HERE / "f10_paired_seeds.csv").write_text(
        "\n".join([",".join(rows[0].keys())] +
                  [",".join(str(v) for v in r.values()) for r in rows]),
        encoding="utf-8")
    print(json.dumps({"summary": {k: {kk: vv for kk, vv in v.items() if kk != "per_seed"}
                                  for k, v in payload["summary"].items()},
                      "paired": paired, "gate": gate}, ensure_ascii=False, indent=2), flush=True)
    print(f"→ {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
