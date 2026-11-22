"""M2.2-D H3b v2 / H4b cued v2 正式重训（10-seed，任务书第 6/11 节）。

对照矩阵（全部 CPU、随机初始化、无预训练权重，与 M2 冻结基线同预算 480 updates）：
  h3b: A_dense_ce / C_sparse_ce / F10_ste × {mixed 无课程, curriculum-h3b-v1 先简后繁}
       × 10 配对 seed（20261131-40，与 M2.2 冻结基线同族）
       curriculum-h3b-v1: 40% scattered+wall_gap → 30% corridor+pocket → 30% 全混合
  h4b cued: A/C/F10 × 10 seed（normal/reversed 成对共享 start/goal 的课程内建于数据）

评测（holdout 封存，每个模型对每个冻结集只评一次）：
  - teacher-forced 遮挡段 acc（h3b 分拓扑族；h4b normal/reversed/规则冲突步）
  - 闭环策略 rollout（任务书"路径规划成功率"）：h3b 模型自走（含障碍碰撞物理）→
    成功率/首达步数/碰撞率/BFS 一致率；h4b 模型自走 → 遮挡步与当前提示规则
    oracle 的一致率（规则提示是否被使用的直接证据，同时报 normal 规则一致率对照）
  - 长度外推（仅 h3b）：holdout 同布局实例 T=36/48（teacher-forced + 闭环）
  - 配对统计：同模式 C−F10、同臂 curriculum−mixed，逐 seed + bootstrap 95% CI
  - 等价性检查：手动训练循环 vs trainer.train()（seed 20261201, C 臂, mixed）
    ——两者单步语义逐行对齐（pred_shift/非有限守卫/裁剪/detect 线程口径），
    该检查用同一 seed 验证两条路径收敛到同一测试分数。

诚实边界：smoke（3 seed × 240 步）之上扩展；ACCEPTED 判定以报告 + STATUS 复核为准，
本脚本输出不自动升级任何里程碑。

用法：
  python -u experiments/m2_2_validation/train_ood_v2_full.py --task h3b
  python -u experiments/m2_2_validation/train_ood_v2_full.py --task h4b
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from nmind.config import default_config  # noqa: E402
from nmind.learning import datasets as D  # noqa: E402
from nmind.learning import ood_layouts as O  # noqa: E402
from nmind.learning.losses import LossWeights, compute_total  # noqa: E402
from nmind.learning.trainer import (  # noqa: E402
    _assemble, _make_probe, build_model, evaluate, train)
from nmind.runtime.hardware import detect  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_SEEDS = list(range(20261131, 20261141))   # 与 M2.2 冻结基线同族配对 seed
ARMS = {
    "A_dense_ce": {"execution": "dense", "use_router": False, "loss_mode": "ce"},
    "C_sparse_ce": {"execution": "sparse", "use_router": True, "loss_mode": "ce"},
    "F10_ste": {"execution": "sparse", "use_router": True, "loss_mode": "joint",
                 "router_ste": True,
                 "loss_overrides": {"detach_pred_target": 1.0,
                                    "balance": 0.02, "entropy": 0.002}},
}
CURRICULUM_VERSION = "curriculum-h3b-v1"
BOOTSTRAP_SEED = 1234
VIS = 4                    # goal_visible_steps（与基础任务一致）


# ------------------------------------------------------------------ 数据


def _merge(parts: list[dict]) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    hashes: list[str] = []
    for p in parts:
        rows.extend(p["rows"])
        hashes.append(p["manifest"]["manifest_hash"])
    return rows, hashes


def build_data(task: str) -> dict:
    if task == "h3b":
        train_parts = [O.make_h3b_split("dev", n=32, seed=s) for s in range(3001, 3009)]
        val_parts = [O.make_h3b_split("dev", n=32, seed=s) for s in (4001, 4002)]
        test_parts = [O.make_h3b_split("holdout", n=32, seed=s) for s in (5001, 5002)]
        len_parts = {t: [O.make_h3b_split("holdout", n=32, seed=s, T=t) for s in (5001, 5002)]
                     for t in (36, 48)}
        obs_dim, traj_len = O.OBS_DIM_WITH_MAP, 24
    elif task == "h4b":
        train_parts = [O.make_h4b_cued_split("dev", n=32, seed=s) for s in range(6001, 6009)]
        val_parts = [O.make_h4b_cued_split("dev", n=32, seed=s) for s in (7001, 7002)]
        test_parts = [O.make_h4b_cued_split("holdout", n=32, seed=s) for s in (8001, 8002)]
        len_parts = {}
        obs_dim, traj_len = O.OBS_DIM_WITH_RULE_CUE, 16
    else:
        raise ValueError(f"未知任务 {task!r}")

    train_rows, train_hashes = _merge(train_parts)
    val_rows, val_hashes = _merge(val_parts)
    test_rows, test_hashes = _merge(test_parts)
    fingerprint = hashlib.sha256(
        "|".join(train_hashes + val_hashes + test_hashes).encode()).hexdigest()[:16]
    audit = {"train_n": len(train_rows), "val_n": len(val_rows), "test_n": len(test_rows),
             "train_manifests": train_hashes, "val_manifests": val_hashes,
             "test_manifests": test_hashes}
    extra = {f"test_T{t}_manifests": [p["manifest"]["manifest_hash"] for p in ps]
             for t, ps in len_parts.items()}
    audit.update(extra)
    return {"train": train_rows, "val": val_rows, "test": test_rows,
            "len_parts": len_parts, "obs_dim": obs_dim, "traj_len": traj_len,
            "dataset_hash": fingerprint,
            "split_version": f"{task}-v2-multi-layout", "audit": audit}


def action_hist(rows: list[dict]) -> dict[str, int]:
    c = Counter(int(a) for r in rows for a in r["actions"])
    return {str(k): v for k, v in sorted(c.items())}


def family_stats(rows: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for fam in sorted({r["task"] for r in rows}):
        sub = [r for r in rows if r["task"] == fam]
        out[fam] = {"n": len(sub), "oracle_action_hist": action_hist(sub)}
    return out


# ------------------------------------------------------------------ 训练


def train_manual(cfg, *, seed: int, arm_flags: dict, schedule: list[tuple[int, list]],
                 val_rows: list[dict], device: torch.device) -> tuple[nn.Module, nn.Module, dict]:
    """与 trainer.train() 单步语义逐行对齐的手动循环（支持分阶段课程）。

    对齐点：build_model(seed) 初始化、probe 联合训练（仅 joint）、pred_shift=next、
    非有限损失跳过、clip_grad_norm、AdamW(lr, wd=1e-4)、rng=default_rng(seed+999)、
    batch 采样 rng.integers(0, len(rows), bs)。差异仅允许：阶段间换训练行池。
    """
    model = build_model(cfg, seed=seed, device=device,
                        execution=arm_flags["execution"],
                        use_router=arm_flags["use_router"],
                        router_ste=bool(arm_flags.get("router_ste", False)))
    pred_mode_eff = model.pred_mode
    joint = arm_flags["loss_mode"] == "joint"
    probe = _make_probe(cfg.model.block_hidden, 36).to(device)
    weights = (LossWeights.from_cfg(cfg) if joint
               else LossWeights(pred=0.0, var=0.0, balance=0.0, entropy=0.0, probe=0.0))
    for k, v in (arm_flags.get("loss_overrides") or {}).items():
        if not hasattr(weights, k):
            raise ValueError(f"未知损失权重 {k!r}；可用：{list(weights.__dataclass_fields__)}")
        setattr(weights, k, float(v))
    params = [p for p in list(model.parameters())
              + (list(probe.parameters()) if joint else []) if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.learning.lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed + 999)
    bs = max(2, int(cfg.learning.batch_size))
    total_steps = sum(n for n, _r in schedule)
    curve: list[dict] = []
    val_curve: list[dict] = []
    non_finite = 0
    step = 0
    model.train()
    t0 = time.perf_counter()
    for n_steps, rows in schedule:
        sums = {"total": 0.0, "action": 0.0, "pred": 0.0}
        cnt = 0
        for _ in range(n_steps):
            pick = [int(i) for i in rng.integers(0, len(rows), size=bs)]
            b = _assemble([rows[i] for i in pick], device)
            out = model.forward_sequence(b["obs"], b["actions"], mem=b["mem"], depth=1)
            losses = compute_total(out, b["actions"], weights,
                                   pred_shift=(pred_mode_eff == "next"))
            if joint:
                p_logits = probe(out["ctx"].reshape(-1, out["ctx"].shape[-1]))
                codes = b["codes"].repeat_interleave(b["obs"].shape[1])
                probe_loss = F.cross_entropy(p_logits, codes)
                losses["probe"] = probe_loss
                losses["total"] = losses["total"] + weights.probe * probe_loss
            else:
                losses["probe"] = losses["total"].new_zeros(())
            if not torch.isfinite(losses["total"]):
                non_finite += 1
                opt.zero_grad(set_to_none=True)
                continue
            opt.zero_grad(set_to_none=True)
            losses["total"].backward()
            nn.utils.clip_grad_norm_(params, cfg.learning.grad_clip)
            opt.step()
            step += 1
            for k in sums:
                sums[k] += float(losses[k].detach())
            cnt += 1
            if step % 40 == 0:
                model.eval()
                ev = evaluate(model, probe if joint else None, val_rows,
                              device=device, occluded_only=True)
                model.train()
                val_curve.append({"step": step,
                                  "val_acc_occ": round(ev.get("accuracy_occluded", float("nan")), 4)})
                curve.append({"step": step,
                              **{k: round(sums[k] / max(1, cnt), 4) for k in sums}})
                sums = {k: 0.0 for k in sums}
                cnt = 0
    wall = time.perf_counter() - t0
    model.eval()
    info = {"val_curve": val_curve, "loss_curve_40step": curve,
            "non_finite_steps_skipped": non_finite, "steps_done": step,
            "steps_requested": total_steps, "wall_s": round(wall, 2)}
    return model, probe, info


# ------------------------------------------------------------------ 闭环评测


def _recover_obstacles(row: dict) -> tuple[tuple[int, int], ...]:
    grid = np.asarray(row["obs"])[0][D.OBS_DIM:D.OBS_DIM + 36]
    return tuple((int(i) // 6, int(i) % 6) for i in np.nonzero(grid)[0])


def h3b_closed_loop(model, rows: list[dict], *, device: torch.device,
                    obs_dim: int, batch: int = 64) -> dict:
    """任务书"路径规划成功率"：模型自走（观测自建含障碍图，动作自选，碰撞物理照环境）。"""
    size = 6
    model.eval()
    per_row: list[dict] = []
    with torch.no_grad():
        for cs in range(0, len(rows), batch):
            chunk = rows[cs:cs + batch]
            n = len(chunk)
            starts = [tuple(r["start"]) for r in chunk]
            goals = [tuple(r["goal"]) for r in chunk]
            obst = [_recover_obstacles(r) for r in chunk]
            max_t = max(int(r["len"]) for r in chunk)
            obs_buf = np.zeros((n, max_t, obs_dim), dtype=np.float32)
            act_buf = np.zeros((n, max_t), dtype=np.int64)
            pos = list(starts)
            collided = [False] * n
            reached = [None] * n
            n_coll = [0] * n
            agree = [0] * n
            occ = [0] * n
            for t in range(max_t):
                for i in range(n):
                    obs_buf[i, t] = O.encode_observable_map(
                        pos[i][0], pos[i][1], goals[i][0], goals[i][1],
                        occluded=t >= VIS, collided=collided[i], obstacles=obst[i])
                obs_t = torch.tensor(obs_buf[:, :t + 1], device=device)
                act_t = torch.tensor(act_buf[:, :t + 1], device=device)
                out = model.forward_sequence(obs_t, act_t, mem=None, depth=1)
                a = out["action_logits"][:, -1].argmax(-1).cpu().numpy()
                for i in range(n):
                    ai = int(a[i])
                    act_buf[i, t] = ai          # actions[t] 条件 t+1；logits[t] 不读它
                    if t >= VIS:
                        oracle = O.shortest_path_action(pos[i], goals[i], obst[i], size)
                        occ[i] += 1
                        agree[i] += int(ai == int(oracle))
                    dx, dy = D.DIRS[ai]
                    nx, ny = pos[i][0] + dx, pos[i][1] + dy
                    if 0 <= nx < size and 0 <= ny < size and (nx, ny) not in obst[i]:
                        collided[i] = False
                        pos[i] = (nx, ny)
                    else:
                        collided[i] = True
                        n_coll[i] += 1
                    if reached[i] is None and pos[i] == goals[i]:
                        reached[i] = t + 1
            for i, r in enumerate(chunk):
                per_row.append({"task": r["task"], "success": reached[i] is not None,
                                "steps_to_goal": reached[i], "collisions": n_coll[i],
                                "bfs_agreement_occ": round(agree[i] / max(1, occ[i]), 4)})
    return _agg_closed_loop(per_row)


def h4b_closed_loop(model, rows: list[dict], *, device: torch.device,
                    obs_dim: int, batch: int = 64) -> dict:
    """模型自走：每步观测末位带规则提示，量遮挡步与"当前提示规则"oracle 的一致率。"""
    size = 6
    model.eval()
    per_row: list[dict] = []
    with torch.no_grad():
        for cs in range(0, len(rows), batch):
            chunk = rows[cs:cs + batch]
            n = len(chunk)
            goals = [tuple(r["goal"]) for r in chunk]
            rev = [bool(r["task"].endswith("reversed")) for r in chunk]
            max_t = max(int(r["len"]) for r in chunk)
            obs_buf = np.zeros((n, max_t, obs_dim), dtype=np.float32)
            act_buf = np.zeros((n, max_t), dtype=np.int64)
            pos = [tuple(r["start"]) for r in chunk]
            agree_rule = [0] * n
            agree_normal = [0] * n
            occ = [0] * n
            viol = [0] * n
            for t in range(max_t):
                for i in range(n):
                    base = D._encode(pos[i][0], pos[i][1], goals[i][0], goals[i][1],
                                     t >= VIS, False)
                    obs_buf[i, t] = np.concatenate(
                        [base, np.array([1.0 if rev[i] else 0.0], dtype=np.float32)])
                obs_t = torch.tensor(obs_buf[:, :t + 1], device=device)
                act_t = torch.tensor(act_buf[:, :t + 1], device=device)
                out = model.forward_sequence(obs_t, act_t, mem=None, depth=1)
                a = out["action_logits"][:, -1].argmax(-1).cpu().numpy()
                for i in range(n):
                    ai = int(a[i])
                    act_buf[i, t] = ai
                    if t >= VIS:
                        occ[i] += 1
                        agree_rule[i] += int(ai == D.oracle_action(
                            pos[i], goals[i], tuple(), size, reversed_control=rev[i]))
                        agree_normal[i] += int(ai == D.oracle_action(
                            pos[i], goals[i], tuple(), size, reversed_control=False))
                    dx, dy = D.DIRS[ai]
                    nx, ny = pos[i][0] + dx, pos[i][1] + dy
                    if not (0 <= nx < size and 0 <= ny < size):
                        viol[i] += 1
                        nx, ny = min(max(nx, 0), size - 1), min(max(ny, 0), size - 1)
                    pos[i] = (nx, ny)
            for i, r in enumerate(chunk):
                per_row.append({"task": r["task"],
                                "rule_agreement_occ": round(agree_rule[i] / max(1, occ[i]), 4),
                                "normal_agreement_occ": round(agree_normal[i] / max(1, occ[i]), 4),
                                "oob_violations": viol[i]})
    return _agg_closed_loop(per_row)


def _agg_closed_loop(per_row: list[dict]) -> dict:
    def agg(sub: list[dict]) -> dict:
        if not sub:
            return {"n": 0}
        succ = [r for r in sub if "success" in r]
        out: dict = {"n": len(sub)}
        if succ and len(succ) == len(sub):
            out["success_rate"] = round(sum(1 for r in succ if r["success"]) / len(succ), 4)
            done = [r["steps_to_goal"] for r in succ if r["success"]]
            out["steps_to_goal_mean"] = round(float(np.mean(done)), 2) if done else None
            out["collisions_per_episode_mean"] = round(
                float(np.mean([r["collisions"] for r in succ])), 2)
            out["bfs_agreement_occ_mean"] = round(
                float(np.mean([r["bfs_agreement_occ"] for r in succ])), 4)
        else:
            out["rule_agreement_occ_mean"] = round(
                float(np.mean([r["rule_agreement_occ"] for r in sub])), 4)
            out["normal_agreement_occ_mean"] = round(
                float(np.mean([r["normal_agreement_occ"] for r in sub])), 4)
            out["oob_violations_per_episode_mean"] = round(
                float(np.mean([r["oob_violations"] for r in sub])), 2)
        return out

    overall = agg(per_row)
    by_task = {name: agg([r for r in per_row if r["task"] == name])
               for name in sorted({r["task"] for r in per_row})}
    return {"overall": overall, "by_task": by_task}


# ------------------------------------------------------------------ 统计


def paired(diffs: list[float], seeds: list[int]) -> dict:
    arr = np.array(diffs, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = [arr[rng.integers(0, len(arr), size=len(arr))].mean() for _ in range(10000)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean": round(float(arr.mean()), 4),
            "std_ddof1": round(float(arr.std(ddof=1)), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "per_seed": {str(s): round(d, 4) for s, d in zip(seeds, diffs)}}


def meanstd(values: list[float]) -> dict:
    return {"mean": round(statistics.mean(values), 4),
            "std_ddof1": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
            "worst": round(min(values), 4), "per_seed": [round(v, 4) for v in values]}


# ------------------------------------------------------------------ 主流程


def h4b_conflict_mask(rows: list[dict]) -> torch.Tensor:
    mask = None
    for r in rows:
        goal = tuple(r["goal"])
        rm = []
        for t in range(int(r["len"])):
            if t < VIS:
                rm.append(False)
                continue
            pos = O.decode_pos(r["obs"][t])
            normal = D.oracle_action(pos, goal, tuple(), 6, reversed_control=False)
            rm.append(normal != int(r["actions"][t]))
        m = torch.tensor(rm, dtype=torch.bool)
        mask = m if mask is None else torch.cat([mask, m])
    return mask


def run_task(task: str, updates: int, seeds: list[int]) -> dict:
    t0 = time.perf_counter()
    data = build_data(task)
    cfg = default_config()
    cfg = cfg.model_copy(update={
        "model": cfg.model.model_copy(update={"event_numeric_dim": data["obs_dim"]}),
        "learning": cfg.learning.model_copy(update={"traj_len": data["traj_len"],
                                                    "eval_interval": 40})})
    device = torch.device("cpu")
    _dev, hw = detect("cpu", torch_threads=cfg.runtime.torch_threads)  # 线程口径与 train() 一致

    # ---- 课程 / 无课程时间表 ----
    train_rows = data["train"]
    specs: dict[str, dict] = {}
    if task == "h3b":
        s1 = [r for r in train_rows if r["task"] in ("h3b_scattered", "h3b_wall_gap")]
        s2 = [r for r in train_rows if r["task"] in ("h3b_corridor", "h3b_pocket")]
        n1, n2 = int(round(updates * 0.4)), int(round(updates * 0.3))
        n3 = updates - n1 - n2
        schedules = {"mixed": [(updates, train_rows)],
                     CURRICULUM_VERSION: [(n1, s1), (n2, s2), (n3, train_rows)]}
        specs[CURRICULUM_VERSION] = {"fractions": [0.4, 0.3, 0.3], "stages": [
            {"steps": n1, "rows": len(s1), "families": ["scattered", "wall_gap"]},
            {"steps": n2, "rows": len(s2), "families": ["corridor", "pocket"]},
            {"steps": n3, "rows": len(train_rows), "families": "all"}]}
    else:
        schedules = {"mixed": [(updates, train_rows)]}

    # ---- 等价性检查（h3b：手动循环 vs trainer.train()，C 臂 seed 20261201）----
    equivalence = None
    if task == "h3b":
        seed_eq = 20261201
        flags = ARMS["C_sparse_ce"]
        data_override = {"train": data["train"], "test": data["test"], "val": data["val"],
                         "dataset_hash": data["dataset_hash"],
                         "split_version": data["split_version"]}
        r_t = train(cfg, epochs=max(1, updates // cfg.learning.n_updates), seed=seed_eq,
                    device="cpu", quiet=True, posthoc_probe=False,
                    data_override=data_override, **flags)
        acc_train = r_t.final["eval"]["accuracy_occluded"]
        m_m, _p_m, _i_m = train_manual(cfg, seed=seed_eq, arm_flags=flags,
                                       schedule=[(updates, data["train"])],
                                       val_rows=data["val"], device=device)
        acc_manual = evaluate(m_m, None, data["test"], device=device,
                              occluded_only=True)["accuracy_occluded"]
        equivalence = {"seed": seed_eq, "arm": "C_sparse_ce", "updates": updates,
                       "trainer_train_acc_occ": round(acc_train, 6),
                       "manual_loop_acc_occ": round(acc_manual, 6),
                       "abs_delta": round(abs(acc_train - acc_manual), 6),
                       "verdict": "equivalent" if abs(acc_train - acc_manual) < 1e-6
                       else "DIVERGENT"}
        print(f"[equivalence] train()={acc_train:.6f} manual={acc_manual:.6f} "
              f"→ {equivalence['verdict']}", flush=True)

    # ---- 正式 10-seed 重训 ----
    probe_live_of = {"A_dense_ce": None, "C_sparse_ce": None, "F10_ste": None}
    runs: list[dict] = []
    for seed in seeds:
        for mode, sched in schedules.items():
            for arm_name, flags in ARMS.items():
                model, probe, info = train_manual(cfg, seed=seed, arm_flags=flags,
                                                  schedule=sched, val_rows=data["val"],
                                                  device=device)
                probe_live = probe if flags["loss_mode"] == "joint" else None
                ev = evaluate(model, probe_live, data["test"], device=device,
                              occluded_only=True)
                ev_all = evaluate(model, probe_live, data["test"], device=device)
                row = {"task": task, "mode": mode, "arm": arm_name, "seed": seed,
                       "tf_acc_occ": round(ev["accuracy_occluded"], 4),
                       "tf_acc_all": round(ev_all["accuracy"], 4),
                       "train_info": info}
                if flags["loss_mode"] == "joint":
                    row["probe_acc_occ"] = round(ev.get("probe_accuracy", float("nan")), 4)
                if flags["loss_mode"] != "ce" or flags["use_router"]:
                    row["route"] = model.active_block_stats()
                if task == "h3b":
                    for fam in O.FAMILIES:
                        sub = [r for r in data["test"] if r["task"] == f"h3b_{fam}"]
                        row[f"tf_acc_occ_{fam}"] = round(evaluate(
                            model, probe_live, sub, device=device,
                            occluded_only=True)["accuracy_occluded"], 4)
                    row["closed_loop_T24"] = h3b_closed_loop(
                        model, data["test"], device=device, obs_dim=data["obs_dim"])
                    for tlen, parts in data["len_parts"].items():
                        rows_t, _h = _merge(parts)
                        row[f"tf_acc_occ_T{tlen}"] = round(evaluate(
                            model, probe_live, rows_t, device=device,
                            occluded_only=True)["accuracy_occluded"], 4)
                        row[f"closed_loop_T{tlen}"] = h3b_closed_loop(
                            model, rows_t, device=device, obs_dim=data["obs_dim"])
                else:
                    normal = [r for r in data["test"] if r["task"] == "h4b_normal"]
                    rev = [r for r in data["test"] if r["task"] == "h4b_reversed"]
                    row["tf_acc_occ_normal"] = round(evaluate(
                        model, probe_live, normal, device=device,
                        occluded_only=True)["accuracy_occluded"], 4)
                    row["tf_acc_occ_reversed"] = round(evaluate(
                        model, probe_live, rev, device=device,
                        occluded_only=True)["accuracy_occluded"], 4)
                    batch = _assemble(rev, device)
                    with torch.no_grad():
                        pred = model.forward_sequence(
                            batch["obs"], batch["actions"], mem=batch["mem"]
                        )["action_logits"].argmax(-1)
                    mask = h4b_conflict_mask(rev)
                    row["tf_reversed_conflict_acc"] = round(float(
                        (pred[mask] == batch["actions"][mask]).float().mean()), 4)
                    row["closed_loop_holdout"] = h4b_closed_loop(
                        model, data["test"], device=device, obs_dim=data["obs_dim"])
                runs.append(row)
                print(f"[{task} {mode} {arm_name} s{seed}] tf_occ={row['tf_acc_occ']:.3f} "
                      f"({info['wall_s']:.0f}s)", flush=True)

    # ---- 汇总与配对统计 ----
    summary: dict[str, dict] = {}
    for mode in schedules:
        for arm_name in ARMS:
            sub = [r for r in runs if r["mode"] == mode and r["arm"] == arm_name]
            summary[f"{arm_name}/{mode}"] = meanstd([r["tf_acc_occ"] for r in sub])
    by_seed = {}
    for mode in schedules:
        for arm_name in ARMS:
            by_seed[(mode, arm_name)] = {
                r["seed"]: r["tf_acc_occ"] for r in runs
                if r["mode"] == mode and r["arm"] == arm_name}
    stats: dict[str, dict] = {}
    if task == "h3b":
        for mode in schedules:
            diffs = [by_seed[("mixed", "C_sparse_ce")][s] - by_seed[(mode, "F10_ste")][s]
                     for s in seeds]
            stats[f"C_minus_F10/{mode}"] = paired(diffs, seeds)
        for arm_name in ARMS:
            if CURRICULUM_VERSION in schedules:
                diffs = [by_seed[(CURRICULUM_VERSION, arm_name)][s]
                         - by_seed[("mixed", arm_name)][s] for s in seeds]
                stats[f"curriculum_minus_mixed/{arm_name}"] = paired(diffs, seeds)
    else:
        def key(mode, arm, field):
            return {r["seed"]: r[field] for r in runs
                    if r["mode"] == mode and r["arm"] == arm}
        for field in ("tf_reversed_conflict_acc", "tf_acc_occ_reversed"):
            diffs = [key("mixed", "C_sparse_ce", field)[s]
                     - key("mixed", "F10_ste", field)[s] for s in seeds]
            stats[f"C_minus_F10/{field}"] = paired(diffs, seeds)

    payload = {
        "experiment": f"{task} v2 corrected-input retrain（10-seed 正式口径）",
        "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": "cpu", "torch_threads": int(torch.get_num_threads()),
        "updates": updates, "seeds": seeds,
        "obs_dim": data["obs_dim"], "traj_len": data["traj_len"],
        "data_audit": data["audit"], "dataset_hash": data["dataset_hash"],
        "train_distribution": family_stats(data["train"]),
        "test_distribution": family_stats(data["test"]),
        "curriculum": specs, "arms": ARMS,
        "equivalence_check": equivalence,
        "runs": runs, "summary": summary, "paired_stats": stats,
        "claim_limit": "正式 10-seed 口径；ACCEPTED 判定以报告+STATUS 复核为准，本文件不自动升级里程碑",
        "wall_s_total": round(time.perf_counter() - t0, 1),
    }
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=("h3b", "h4b"), required=True)
    ap.add_argument("--updates", type=int, default=480)
    ap.add_argument("--seeds", type=str,
                    default=",".join(map(str, DEFAULT_SEEDS)),
                    help="逗号分隔或 a-b 区间，默认 20261131-40")
    ap.add_argument("--suffix", type=str, default="full")
    args = ap.parse_args()
    if "-" in args.seeds and "," not in args.seeds:
        lo, hi = args.seeds.split("-")
        seeds = list(range(int(lo), int(hi) + 1))
    else:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    payload = run_task(args.task, args.updates, seeds)
    dest = HERE / f"{args.task}_v2_retrain_{args.suffix}.json"
    tmp = dest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(dest)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2), flush=True)
    print(f"→ {dest}（总用时 {payload['wall_s_total']:.0f}s）", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
