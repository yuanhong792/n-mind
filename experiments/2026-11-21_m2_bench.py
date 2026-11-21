"""M2 延迟基准：用**训练后的真实权重**测单序列端到端延迟与 k 的取舍。

与 cli benchmark 的分工：
  * cli benchmark：随机初始化、固定 T=48，回答"架构本身省不省时间"；
  * 本脚本：2x2 训练出的 checkpoint，T=24（训练长度）与 T=48（外推长度），
    回答"训练后路由真的稀疏吗，k 变小能不能换来延迟"。两者都保留。

口径：
  * 单行 batch=1（批内共享计算会掩盖差异），warmup 后采样；
  * 每个样本重复 repeats 次，报 p50/p95/mean/max；
  * 同时报 mean_active_blocks（由 blocks_touched 实测，不是由 k 推的）；
  * 每个 k 都重测准确率：延迟换来的若是能力下降，要写在同一张表里。
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nmind.learning import datasets as D
from nmind.learning.checkpoint import load_checkpoint
from nmind.learning.trainer import _assemble, make_data


def _rows(seed: int, T: int, cfg) -> list[dict[str, Any]]:
    data = make_data(cfg, D.GridTask(goal_visible_steps=4, seed=seed), seed=seed)
    rows = []
    for r in data["test"]:
        rows.append(dict(r, obs=r["obs"][:T], actions=r["actions"][:T],
                         mem=r.get("mem", np.zeros((T, 16), np.float32))[:T], len=T))
    return rows


def _eval_at(model, rows, device, k: int | None) -> dict[str, float]:
    hit = tot = occ_hit = occ_n = 0
    touched = 0
    steps = 0
    model.eval()
    with torch.no_grad():
        for r in rows:
            b = _assemble([r], device)
            out = model.forward_sequence(b["obs"], b["actions"], k=k)
            pred = out["action_logits"].argmax(-1)
            hit += int((pred == b["actions"]).sum().item())
            tot += int(pred.numel())
            occ = b["obs"][:, :, 24] > 0.5
            occ_hit += int((pred[occ] == b["actions"][occ]).sum().item())
            occ_n += int(occ.sum().item())
            touched += int(out["blocks_touched"])
            steps += b["obs"].shape[1]
    return {"accuracy": hit / max(1, tot), "accuracy_occluded": occ_hit / max(1, occ_n),
            "mean_active_blocks": touched / max(1, steps), "n_occ": float(occ_n)}


def _latency(model, rows, device, k: int | None, *, repeats: int) -> dict[str, float]:
    model.eval()
    warm = [rows[0], rows[1]]
    with torch.no_grad():
        for _ in range(5):
            for r in warm:
                b = _assemble([r], device)
                model.forward_sequence(b["obs"], b["actions"], k=k)
        samples = []
        for _ in range(repeats):
            for r in rows:
                b = _assemble([r], device)
                t0 = time.perf_counter()
                model.forward_sequence(b["obs"], b["actions"], k=k)
                samples.append((time.perf_counter() - t0) * 1000.0)
    ss = sorted(samples)
    return {"n": len(ss), "p50_ms": round(statistics.median(ss), 4),
            "p95_ms": round(ss[min(len(ss) - 1, int(0.95 * (len(ss) - 1)))], 4),
            "mean_ms": round(statistics.fmean(ss), 4), "max_ms": round(ss[-1], 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="20261121")
    ap.add_argument("--arms", default="A_dense_ce,D_sparse_joint")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--tag", default="s1")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    ckpt_dir = Path(__file__).parent / "m2_ablation"
    device = torch.device("cpu")
    out: dict[str, Any] = {"repeats": args.repeats, "runs": []}

    for seed in seeds:
        for arm in arms:
            path = ckpt_dir / f"{arm}_s{seed}.pt"
            if not path.exists():
                print(f"跳过 {path.name}：不存在")
                continue
            model, meta = load_checkpoint(path, device="cpu")
            cfg = meta.get("config_hash")
            for T in (24, 48):
                rows = _rows(seed, T, _cfg_of(model))
                ks = [1, 2, 4] if model.execution == "sparse" else [None]
                for k in ks:
                    lat = _latency(model, rows, device, k, repeats=args.repeats)
                    ev = _eval_at(model, rows, device, k)
                    rec = {"arm": arm, "seed": seed, "T": T, "k": k, **lat, **ev,
                           "config_hash": cfg}
                    out["runs"].append(rec)
                    print(f"{arm} s{seed} T={T} k={k}: p50={lat['p50_ms']}ms "
                          f"p95={lat['p95_ms']}ms occ={ev['accuracy_occluded']:.3f} "
                          f"blocks={ev['mean_active_blocks']}")

    (ckpt_dir / f"m2_bench_{args.tag}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"写入 {ckpt_dir}/m2_bench_{args.tag}.json")


def _cfg_of(model):
    """从模型实例还原 cfg：make_data 只用到 model.* 与 learning 默认值。"""
    from nmind.config import default_config

    cfg = default_config()
    cfg.model = model.cfg
    return cfg


if __name__ == "__main__":
    main()
