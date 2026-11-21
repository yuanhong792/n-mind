"""M2.2 最小附加消融：F9（只软化均衡/熵，不 detach）与 F10（F7 + 稀疏 STE 路由）。

依据 gradient_diagnostics.json 的实测：
  - 稀疏模式下 action/pred/probe 对 router 的梯度全部为 0（任务盲路由），
    只有 balance/entropy 正则在训练 router；
  - 稠密模式的 router 有任务梯度（STE 软掩码通路），且在 router 上
    action×pred cosine = -0.453（冲突证据），但仍收敛到 0.98。

因此补充两臂回答"各自贡献多少"：
  F9_soft_only  = D + 软化均衡/熵（balance 0.02, entropy 0.002），不 detach
                  —— 隔离"软化正则"本身的作用（F7 = detach + 软化，缺这半边）
  F10_ste       = F7 配置 + router_ste=True
                  —— 给稀疏路由接上任务梯度（NCell.forward_sparse 的 STE 权重通路，
                     前向数值与硬选择一致，由 tests/test_router_ste.py 钉死）

预期读法（写进报告前不下结论）：
  F9 ≈ D 且 << F7 ⇒ 软化正则单独无效，detach 是主导；
  F9 ≈ F7 ⇒ detach 可省，软化就够了；
  F10 > F7 ⇒ 任务盲路由确实是瓶颈之一；F10 ≈ F7 ⇒ 路由不是瓶颈。

用法：
  python experiments/m2_closeout/f9_f10_router_ste.py --seeds 20261121,20261122,20261123
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from nmind.config import default_config  # noqa: E402
from nmind.learning import datasets as D  # noqa: E402
from nmind.learning.checkpoint import save_checkpoint  # noqa: E402
from nmind.learning.trainer import train  # noqa: E402

OUT = ROOT / "experiments" / "m2_closeout"
CKPT_DIR = ROOT / "experiments" / "m2_ablation"

ARMS = {
    # F7 的对照半边：只软化正则，不 detach
    "F9_soft_only": dict(router_ste=False,
                         loss_overrides={"balance": 0.02, "entropy": 0.002}),
    # F7 + 稀疏 STE 路由：给 router 接任务梯度
    "F10_ste": dict(router_ste=True,
                    loss_overrides={"detach_pred_target": 1.0,
                                    "balance": 0.02, "entropy": 0.002}),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="20261121,20261122,20261123")
    ap.add_argument("--steps", type=int, default=480)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    rows: list[dict] = []
    cfg0 = default_config()
    cfg0.learning.epochs = max(1, args.steps // cfg0.learning.n_updates)
    cfg0.learning.traj_len = 24
    cfg0.learning.eval_interval = 10
    cfg0.runtime.device = "cpu"

    for seed in seeds:
        for arm, kw in ARMS.items():
            cfg = cfg0.model_copy(deep=True)
            ckpt = CKPT_DIR / f"{arm}_s{seed}.pt"
            t0 = time.perf_counter()
            res = train(cfg, seed=seed, device="cpu", posthoc_probe=True, quiet=True,
                        checkpoint_path=str(ckpt), **kw)
            wall = time.perf_counter() - t0
            ev = res.final["eval_all"]
            pos_t = res.final.get("probe_posthoc_test", {})
            row = {"arm": arm, "seed": seed, "wall_s": round(wall, 2), "steps": res.steps,
                   "acc_all": round(ev["accuracy"], 4),
                   "acc_occ": round(ev.get("accuracy_occluded", float("nan")), 4),
                   "probe_posthoc_test": round(pos_t.get("probe_posthoc_accuracy", float("nan")), 4),
                   "loss_tail": round(res.curve["total"][-1], 4),
                   "non_finite": res.final["non_finite_steps_skipped"],
                   "router_ste": kw["router_ste"],
                   "checkpoint": str(ckpt),
                   "dataset_hash": res.final["dataset_hash"]}
            rows.append(row)
            print(f"[{arm} s{seed}] occ={row['acc_occ']} all={row['acc_all']} "
                  f"probe_te={row['probe_posthoc_test']} ({wall:.0f}s)", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / "f9_f10_results.json"
    dest.write_text(json.dumps({
        "steps": args.steps, "seeds": seeds,
        "arms": {k: {"router_ste": v["router_ste"], "loss_overrides": v["loss_overrides"]}
                 for k, v in ARMS.items()},
        "reference_seed20261121": {"C_sparse_ce": 0.9563, "D_sparse_joint": 0.5719,
                                   "F1_detach": 0.8984, "F7_detach_soft_balent": 0.9234},
        "runs": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"→ {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
