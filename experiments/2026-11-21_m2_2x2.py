"""M2 四组控制变量实验：稠密/稀疏 × CE/联合，同一数据、同一预算、同一评测。

为什么必须是同一份 train()：
  四组如果各自手写训练循环，"配置差异"会混进训练代码差异。这里只翻三个开关
  （execution / use_router / loss_mode），其余全部来自同一个 cfg、同一个 seed。

对照矩阵：
  A dense  + ce      —— 上界参考：全激活，只学动作
  B dense  + joint   —— 隔离"联合损失"的代价（无路由干扰）
  C sparse + ce      —— 隔离"稀疏路由"的代价（无联合损失干扰）
  D sparse + joint   —— M2 报障配置
  D 与 A 的差距 = B 与 A 的差距 + C 与 A 的差距 + 交互项（可用 2×2 差值检验）。

用法：
  PYTHONPATH=src python experiments/2026-11-21_m2_2x2.py --steps 480 --seeds 20261121
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from nmind.config import default_config
from nmind.learning import datasets as D
from nmind.learning.trainer import _assemble, make_data, train

ARMS = {
    "A_dense_ce": dict(execution="dense", use_router=False, loss_mode="ce"),
    "B_dense_joint": dict(execution="dense", use_router=False, loss_mode="joint"),
    "C_sparse_ce": dict(execution="sparse", use_router=True, loss_mode="ce"),
    "D_sparse_joint": dict(execution="sparse", use_router=True, loss_mode="joint"),
}

# 联合损失内部归因：只改一个权重，其余保持 D 的默认值
ISO = {
    "I_no_pred": {"pred": 0.0},
    "I_no_var": {"var": 0.0},
    "I_no_balent": {"balance": 0.0, "entropy": 0.0},
    "I_no_probe": {"probe": 0.0},
    "I_only_pred": {"var": 0.0, "balance": 0.0, "entropy": 0.0, "probe": 0.0},
    "I_only_probe": {"pred": 0.0, "var": 0.0, "balance": 0.0, "entropy": 0.0},
}

# 归因后的候选修复：全部只改损失口径，不改结构、不改稀疏推理路径
FIX = {
    "F1_detach": {"detach_pred_target": 1.0},
    "F2_detach_novar": {"detach_pred_target": 1.0, "var": 0.0},
    "F3_pred025_novar": {"pred": 0.25, "var": 0.0},
    "F4_pred025": {"pred": 0.25},
    "F5_dense_detach": {"detach_pred_target": 1.0},
    # F1 恢复后仍比 sparse+CE 低约 6 分，试三个"软化联合项"的变体，看能否再靠近
    "F6_detach_pred025": {"detach_pred_target": 1.0, "pred": 0.25},
    "F7_detach_soft_balent": {"detach_pred_target": 1.0, "balance": 0.02, "entropy": 0.002},
    "F8_detach_probe05": {"detach_pred_target": 1.0, "probe": 0.5},
    # M2.2 机制修复候选（2026-11-21）：STE 路由。flags 里的 router_ste 由 main() 特判注入
    "F10_ste": {"detach_pred_target": 1.0, "balance": 0.02, "entropy": 0.002},
}


def perf_profile(model, rows, device, *, repeats: int = 5) -> dict[str, float]:
    """测整条序列的 CPU 延迟与路由行为；全部 no_grad。"""
    model.eval()
    b = _assemble(rows[:4], device)
    times: list[float] = []
    with torch.no_grad():
        for _ in range(repeats):
            t0 = time.perf_counter()
            out = model.forward_sequence(b["obs"], b["actions"])
            times.append((time.perf_counter() - t0) * 1000.0)
        bsz, T = b["obs"].shape[0], b["obs"].shape[1]
        active_blocks = out["blocks_touched"] / max(1, bsz * T)
        probs = out["route_probs"]
        ent = float(-(probs.clamp_min(1e-12).log() * probs).sum(-1).mean())
        top1 = float(probs.max(-1).values.mean())
    stats = model.active_block_stats() if getattr(model, "router", None) is not None else None
    med = statistics.median(times)
    return {"seq_ms_p50": round(med, 3), "seq_ms_min": round(min(times), 3),
            "mean_active_blocks_per_step": round(active_blocks, 3),
            "route_entropy": round(ent, 4), "route_top1_prob": round(top1, 4),
            "top1_block_share": None if stats is None else stats["top1_block_share"],
            "dead_blocks": None if stats is None else stats["dead_blocks"]}


def run_arm(name: str, flags: dict[str, Any], *, cfg, seed: int, device: str,
            out_dir: Path, T: int, loss_overrides: dict[str, float] | None = None) -> dict[str, Any]:
    cfg = cfg.model_copy(deep=True)
    ckpt = out_dir / f"{name}_s{seed}.pt"
    t0 = time.perf_counter()
    res = train(cfg, seed=seed, device=device, posthoc_probe=True, quiet=True,
                checkpoint_path=str(ckpt), loss_overrides=loss_overrides, **flags)
    wall = time.perf_counter() - t0
    model = res.model
    dev = next(model.parameters()).device
    # 评测行与 train() 内部同一 task/seed ⇒ 同一份 test（dataset_hash 在 JSON 里核对）
    rows = make_data(cfg, D.GridTask(goal_visible_steps=4, seed=seed), seed=seed)["test"]
    prof = perf_profile(model, rows, dev)
    npar = model.n_params()
    k = int(cfg.model.top_k)
    # 每步激活参数量的估算：稀疏时只有 k/N 的块参数参与，其余模块照常
    block_params = npar["blocks"]
    active_est = npar["total"] if flags["execution"] == "dense" else \
        int(npar["total"] - block_params * (1.0 - k / cfg.model.n_blocks))
    ev = res.final["eval_all"] if "eval_all" in res.final else res.final["eval"]
    pos_v = res.final.get("probe_posthoc_val", {})
    pos_t = res.final.get("probe_posthoc_test", {})
    row = {"arm": name, "seed": seed, "execution": flags["execution"],
           "loss_mode": flags["loss_mode"], "wall_s": round(wall, 2), "steps": res.steps,
           "params_total": npar["total"], "active_params_per_step_est": active_est,
           "loss_head": round(res.curve["total"][0], 4),
           "loss_tail": round(res.curve["total"][-1], 4),
           "acc_all": round(ev["accuracy"], 4),
           "acc_occ": round(ev.get("accuracy_occluded", float("nan")), 4),
           "n_steps_occ": int(ev.get("n_steps_occluded", 0)),
           "probe_posthoc_val": round(pos_v.get("probe_posthoc_accuracy", float("nan")), 4),
           "probe_posthoc_test": round(pos_t.get("probe_posthoc_accuracy", float("nan")), 4),
           "probe_joint": round(ev.get("probe_accuracy", float("nan")), 4)
           if "probe_accuracy" in ev else None,
           "checkpoint": str(ckpt)}
    # 曲线只留 40 个等距点：够看出"先学会再被干扰"还是"从未学会"
    act_curve = res.curve["action"]
    stride = max(1, len(act_curve) // 40)
    row["action_curve"] = [round(float(v), 4) for v in act_curve[::stride]]
    row["val_occ_trace"] = [{"ep": int(e["epoch"]),
                             "acc_occ": round(float(e["val"].get("accuracy_occluded", 0.0)), 4)}
                            for e in res.evals]
    row["non_finite_steps"] = int(res.final.get("non_finite_steps_skipped", 0))
    row.update(prof)
    return row


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按 arm 汇总：均值 ± 样本标准差；单 seed 时 std 记 0 并标注。"""
    keys = ["acc_all", "acc_occ", "probe_posthoc_val", "probe_posthoc_test", "seq_ms_p50",
            "mean_active_blocks_per_step", "wall_s"]
    out = []
    for arm in dict.fromkeys(r["arm"] for r in rows):
        mine = [r for r in rows if r["arm"] == arm]
        if not mine:
            continue
        agg = {"arm": arm, "n_seeds": len(mine)}
        for key in keys:
            vals = [float(r[key]) for r in mine if r.get(key) is not None]
            if not vals or any(v != v for v in vals):
                agg[key] = None
                continue
            agg[key] = round(statistics.mean(vals), 4)
            # 2026-11-21 M2.2 起统一样本 std（ddof=1）；此前历史 JSON 为 pstdev（ddof=0）
            agg[key + "_std"] = round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0
        out.append(agg)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=480, help="每个 arm 的优化步数（四组相同）")
    ap.add_argument("--seeds", type=str, default="20261121")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--traj-len", type=int, default=24)
    ap.add_argument("--tag", default="")
    ap.add_argument("--arms", default=",".join(ARMS),
                    help=f"逗号分隔；可选 {list(ARMS)} + {list(ISO)}")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    picked = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in picked if a not in ARMS and a not in ISO and a not in FIX]
    if unknown:
        raise SystemExit(f"未知 arm：{unknown}；可选 {list(ARMS)} + {list(ISO)} + {list(FIX)}")

    cfg = default_config()
    cfg.learning.epochs = max(1, args.steps // cfg.learning.n_updates)
    cfg.learning.traj_len = args.traj_len
    cfg.learning.eval_interval = 10      # 训练轨迹与损失不变，只降低评测频率的工程量
    cfg.runtime.device = args.device

    # 数据指纹必须四组一致：打印一次，写进 JSON
    data = make_data(cfg, D.GridTask(goal_visible_steps=4, seed=seeds[0]), seed=seeds[0])
    print(f"dataset_hash={data['dataset_hash']} split_version={data.get('split_version')} "
          f"train={len(data['train'])} val={len(data['val'])} test={len(data['test'])}")

    out_dir = Path(__file__).parent / "m2_ablation"
    out_dir.mkdir(exist_ok=True)
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for name in picked:
            if name in ARMS:
                flags, overrides = dict(ARMS[name]), None
            elif name in ISO:
                flags, overrides = dict(ARMS["D_sparse_joint"]), dict(ISO[name])
            else:
                base = ARMS["B_dense_joint"] if name.startswith("F5") else ARMS["D_sparse_joint"]
                flags, overrides = dict(base), dict(FIX[name])
                if name == "F10_ste":
                    flags["router_ste"] = True    # F10 的机制开关（train 已支持该 kwargs）
            t0 = time.perf_counter()
            row = run_arm(name, flags, cfg=cfg, seed=seed, device=args.device,
                          out_dir=out_dir, T=args.traj_len, loss_overrides=overrides)
            rows.append(row)
            print(f"[{name} s{seed}] occ={row['acc_occ']} all={row['acc_all']} "
                  f"probe_te={row['probe_posthoc_test']} {row['seq_ms_p50']}ms "
                  f"({time.perf_counter() - t0:.0f}s)")

    tag = f"_{args.tag}" if args.tag else ""
    agg = aggregate(rows)
    csv_path = Path(__file__).parent.parent / "experiments" / f"m2_results{tag}.csv"
    scalar_cols = [k for k, v in rows[0].items() if not isinstance(v, (list, dict))]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=scalar_cols)
        w.writeheader()
        w.writerows([{k: r[k] for k in scalar_cols} for r in rows])
    arms_meta = {}
    for name in picked:
        base = dict(ARMS[name]) if name in ARMS else dict(
            ARMS["B_dense_joint"] if name.startswith("F5") else ARMS["D_sparse_joint"])
        if name == "F10_ste":
            base["router_ste"] = True
        arms_meta[name] = {"flags": base,
                           "loss_overrides": (None if name in ARMS else
                                              dict(ISO[name]) if name in ISO else dict(FIX[name]))}
    payload = {"steps": args.steps, "seeds": seeds, "traj_len": args.traj_len,
               "dataset_hash": data["dataset_hash"], "split_version": data.get("split_version"),
               "n_train": len(data["train"]), "n_val": len(data["val"]), "n_test": len(data["test"]),
               "arms": arms_meta, "runs": rows, "aggregate": agg}
    (out_dir / f"m2_2x2{tag}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                               encoding="utf-8")
    print(f"\n写入 {csv_path} 与 {out_dir}/m2_2x2{tag}.json")
    for a in agg:
        print(f"{a['arm']:<16} occ={a['acc_occ']} all={a['acc_all']} "
              f"probe_te={a['probe_posthoc_test']} ms={a['seq_ms_p50']}")


if __name__ == "__main__":
    main()
