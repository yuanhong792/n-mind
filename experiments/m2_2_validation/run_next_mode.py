"""pred_mode="next"（真世界模型目标）验证训练：F10 配置 + next 预测。

对比基线：F10_ste（current 模式，pred_eval 已判 world_model_validated=false）。
问题：把 pred 损失改成真正的 (s_t, a_t)→s_{t+1} 目标后——
  1. pred_eval 是否翻转为 validated=true（动作敏感性 + 未来状态读出超拷贝基线）；
  2. 主任务（遮挡段准确率）是否不退化（world-model 目标应提供有用的表征压力）。

用法：
  python -u experiments/m2_2_validation/run_next_mode.py --seeds 20261121,20261122,20261123
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from nmind.config import default_config  # noqa: E402
from nmind.learning.trainer import train  # noqa: E402

HERE = Path(__file__).resolve().parent
CKPT_DIR = ROOT / "experiments" / "m2_ablation"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="20261121,20261122,20261123")
    ap.add_argument("--steps", type=int, default=480)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    rows = []
    for seed in seeds:
        cfg = default_config()
        cfg.learning.epochs = max(1, args.steps // cfg.learning.n_updates)
        cfg.learning.eval_interval = 10
        cfg.runtime.device = "cpu"
        t0 = time.perf_counter()
        res = train(cfg, seed=seed, device="cpu", posthoc_probe=True, quiet=True,
                    checkpoint_path=str(CKPT_DIR / f"F10_next_s{seed}.pt"),
                    execution="sparse", use_router=True, loss_mode="joint",
                    router_ste=True, pred_mode="next",
                    loss_overrides={"detach_pred_target": 1.0,
                                    "balance": 0.02, "entropy": 0.002})
        wall = time.perf_counter() - t0
        ev = res.final["eval_all"]
        pos = res.final.get("probe_posthoc_test", {})
        rows.append({"arm": "F10_next", "seed": seed, "steps": res.steps,
                     "acc_all": round(ev["accuracy"], 4),
                     "acc_occ": round(ev.get("accuracy_occluded", float("nan")), 4),
                     "probe_posthoc_test": round(pos.get("probe_posthoc_accuracy", float("nan")), 4),
                     "pred_tail": round(res.curve["pred"][-1], 4),
                     "non_finite": res.final["non_finite_steps_skipped"],
                     "pred_mode": res.final["ablation"]["pred_mode"],
                     "router_ste": res.final["ablation"]["router_ste"],
                     "dataset_hash": res.final["dataset_hash"],
                     "wall_s": round(wall, 1),
                     "checkpoint": str(CKPT_DIR / f"F10_next_s{seed}.pt")})
        print(f"[F10_next s{seed}] occ={rows[-1]['acc_occ']} probe_te={rows[-1]['probe_posthoc_test']} "
              f"pred_tail={rows[-1]['pred_tail']} ({wall:.0f}s)", flush=True)

    dest = HERE / "next_mode_results.json"
    old = json.loads(dest.read_text(encoding="utf-8")) if dest.exists() else {"runs": []}
    merged = {f"{r['arm']}_s{r['seed']}": r for r in old.get("runs", [])}
    for r in rows:
        merged[f"{r['arm']}_s{r['seed']}"] = r
    dest.write_text(json.dumps({"runs": list(merged.values())}, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print(f"→ {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
