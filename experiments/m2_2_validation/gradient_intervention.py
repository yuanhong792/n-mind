"""M2.2-C 训练轨迹干预实验（提示词第 5 节）：从同一 fork 点分叉，比较机制开关。

设计（最小充分矩阵，为什么是这 5 臂）：
  fork 点 = D 配置（sparse+joint，无 detach、无 STE）训练 240 步后的
  模型参数 + AdamW 优化器状态 + 数据顺序 RNG 状态（深拷贝，逐 bit 相同）。

  arm0_baseline    继续 D 原配置 240 步      —— 对照：不加干预会怎样
  arm1_detach      开 detach_pred_target   —— F1 的机制是否在"中途切换"下仍成立
  arm2_ste         开 router_ste           —— 任务梯度进入 router 是否立即改善
  arm3_detach_ste  同时开两者              —— F10 组合
  arm4_pred0       pred 权重→0             —— 完全关闭 pred 路径的对照

  每臂共享相同数据顺序与后续 240 步；记录切换瞬间与之后每 40 步的 val 遮挡
  准确率、action loss、router 任务梯度范数（action-only autograd.grad）、
  最终 posthoc probe 与 pred 有效值。3 个配对 seed（20261121/22/23）。

判读（预登记）：
  arm1/arm3 在切换后 <80 步内 occ 超过 arm0 并持续拉开 ⇒ detach/STE 的收益
  发生在训练中（机制性），不是终点巧合；arm2 单独无效 ⇒ 任务梯度进 router
  本身不足以修复（pred 路径才是主因）；arm4 ≈ arm1 ⇒ detach 等价于关闭
  pred 对主干的反传。

用法：
  python -u experiments/m2_2_validation/gradient_intervention.py
"""

from __future__ import annotations

import copy
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from nmind.config import default_config  # noqa: E402
from nmind.learning import datasets as D  # noqa: E402
from nmind.learning.losses import LossWeights, compute_total, prediction_loss  # noqa: E402
from nmind.learning.trainer import _assemble, evaluate, fit_posthoc_probe, make_data  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE
SEEDS = [20261121, 20261122, 20261123]
FORK_STEP = 240
PHASE2_STEPS = 240
EVAL_EVERY = 40

ARMS = [
    ("arm0_baseline", {}),
    ("arm1_detach", {"detach_pred_target": True}),
    ("arm2_ste", {"router_ste": True}),
    ("arm3_detach_ste", {"detach_pred_target": True, "router_ste": True}),
    ("arm4_pred0", {"pred_weight": 0.0}),
]


def router_task_grad_norm(model, batch_rows, device) -> float:
    """action-only CE 对 router 的梯度范数（判别任务梯度是否进入路由器）。"""
    b = _assemble(batch_rows, device)
    out = model.forward_sequence(b["obs"], b["actions"], mem=b["mem"], depth=1)
    ce = F.cross_entropy(out["action_logits"].reshape(-1, out["action_logits"].shape[-1]),
                         b["actions"].reshape(-1))
    gs = torch.autograd.grad(ce, list(model.router.parameters()), allow_unused=True)
    total = 0.0
    for g in gs:
        if g is not None:
            total += float(g.pow(2).sum())
    return total ** 0.5


def run_seed(seed: int) -> dict:
    cfg = default_config()
    cfg.learning.epochs = 200
    cfg.learning.eval_interval = 1000        # 干预脚本自己控制评测节奏
    cfg.runtime.device = "cpu"
    dev = torch.device("cpu")
    torch.manual_seed(seed)
    np.random.seed(seed % 2**32)

    from nmind.learning.trainer import build_model, _make_probe
    model = build_model(cfg, seed=seed, device=dev)
    probe = _make_probe(cfg.model.block_hidden, 36).to(dev)
    data = make_data(cfg, D.GridTask(goal_visible_steps=4, seed=seed), seed=seed)
    rows, test_rows, val_rows = data["train"], data["test"], data["val"]
    weights = LossWeights.from_cfg(cfg)
    params = [p for p in list(model.parameters()) + list(probe.parameters()) if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.learning.lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed + 999)
    bs = max(2, cfg.learning.batch_size)
    steps_per_eval = EVAL_EVERY

    def one_step():
        pick = [int(i) for i in rng.integers(0, len(rows), size=bs)]
        b = _assemble([rows[i] for i in pick], dev)
        out = model.forward_sequence(b["obs"], b["actions"], mem=b["mem"], depth=1)
        losses = compute_total(out, b["actions"], weights)
        p_logits = probe(out["ctx"].reshape(-1, out["ctx"].shape[-1]))
        codes = b["codes"].repeat_interleave(b["obs"].shape[1])
        probe_loss = F.cross_entropy(p_logits, codes)
        losses["total"] = losses["total"] + weights.probe * probe_loss
        if not torch.isfinite(losses["total"]):
            opt.zero_grad(set_to_none=True)
            return None
        opt.zero_grad(set_to_none=True)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(params, cfg.learning.grad_clip)
        opt.step()
        return {k: float(losses[k].detach()) for k in ("total", "action", "pred")}

    def val_occ():
        ev = evaluate(model, probe, val_rows, device=dev, occluded_only=True)
        return round(ev.get("accuracy_occluded", float("nan")), 4)

    # ---- Phase 1：共同 fork 点 ----
    phase1_curve, phase1_val = [], []
    for step in range(FORK_STEP):
        losses = one_step()
        phase1_curve.append(losses)
        if (step + 1) % steps_per_eval == 0:
            phase1_val.append({"step": step + 1, "val_occ": val_occ()})
    fork_val_occ = val_occ()
    fork_router_grad = router_task_grad_norm(model, rows[:8], dev)

    # ---- fork：深拷贝参数/优化器/RNG ----
    arms = {}
    for name, switches in ARMS:
        m2 = build_model(cfg, seed=seed, device=dev)
        m2.load_state_dict(model.state_dict())
        pr2 = _make_probe(cfg.model.block_hidden, 36).to(dev)
        pr2.load_state_dict(probe.state_dict())
        opt2 = torch.optim.AdamW([p for p in list(m2.parameters()) + list(pr2.parameters())
                                  if p.requires_grad], lr=cfg.learning.lr, weight_decay=1e-4)
        opt2.load_state_dict(copy.deepcopy(opt.state_dict()))
        rng2 = np.random.default_rng(seed + 999)
        # 重放 Phase1 的采样序列到 fork 点（保证各臂数据顺序一致）
        for _ in range(FORK_STEP):
            rng2.integers(0, len(rows), size=bs)
        arms[name] = {"model": m2, "probe": pr2, "opt": opt2, "rng": rng2,
                      "switches": switches}

    results = {}
    for name, st in arms.items():
        m2, pr2, opt2, rng2 = st["model"], st["probe"], st["opt"], st["rng"]
        sw = st["switches"]
        m2.router_ste = bool(sw.get("router_ste", False))
        w2 = LossWeights.from_cfg(cfg)
        w2.detach_pred_target = bool(sw.get("detach_pred_target", False))
        if "pred_weight" in sw:
            w2.pred = float(sw["pred_weight"])
        trace = [{"step": FORK_STEP, "val_occ": fork_val_occ}]
        first_step_router_grad = None
        action_curve = []
        for i in range(PHASE2_STEPS):
            pick = [int(x) for x in rng2.integers(0, len(rows), size=bs)]
            b = _assemble([rows[k] for k in pick], dev)
            out = m2.forward_sequence(b["obs"], b["actions"], mem=b["mem"], depth=1)
            losses = compute_total(out, b["actions"], w2)
            p_logits = pr2(out["ctx"].reshape(-1, out["ctx"].shape[-1]))
            codes = b["codes"].repeat_interleave(b["obs"].shape[1])
            probe_loss = F.cross_entropy(p_logits, codes)
            total = losses["total"] + w2.probe * probe_loss
            if not torch.isfinite(total):
                opt2.zero_grad(set_to_none=True)
                continue
            opt2.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in list(m2.parameters()) + list(pr2.parameters()) if p.requires_grad],
                cfg.learning.grad_clip)
            opt2.step()
            action_curve.append(float(losses["action"]))
            if i == 0:
                first_step_router_grad = router_task_grad_norm(m2, rows[:8], dev)
            if (i + 1) % steps_per_eval == 0:
                m2.eval()
                trace.append({"step": FORK_STEP + i + 1, "val_occ": val_occ()})
                m2.train()
        m2.eval()
        pos = fit_posthoc_probe(m2, rows, test_rows, device=dev, seed=seed)
        ev = evaluate(m2, None, test_rows, device=dev, occluded_only=True)
        results[name] = {
            "switches": sw,
            "first_step_router_task_grad": round(first_step_router_grad, 6),
            "trace": trace,
            "final_val_occ": trace[-1]["val_occ"],
            "test_acc_occ": round(ev.get("accuracy_occluded", float("nan")), 4),
            "posthoc_probe_test": round(pos["probe_posthoc_accuracy"], 4),
            "action_loss_head": round(statistics.mean(action_curve[:10]), 4),
            "action_loss_tail": round(statistics.mean(action_curve[-10:]), 4),
        }
        print(f"  [s{seed} {name}] fork_val={fork_val_occ} → final_val={results[name]['final_val_occ']} "
              f"test_occ={results[name]['test_acc_occ']} router_grad0={first_step_router_grad:.4f}",
              flush=True)

    return {"seed": seed, "fork_step": FORK_STEP,
            "fork_val_occ": fork_val_occ,
            "fork_router_task_grad": round(fork_router_grad, 6),
            "arms": results}


def main() -> int:
    t0 = time.perf_counter()
    out = {"experiment": "训练轨迹干预（detach/STE/pred 权重，同一 fork 点分叉）",
           "fork_step": FORK_STEP, "phase2_steps": PHASE2_STEPS,
           "minimal_matrix_rationale":
               "5 臂覆盖：不加干预对照 / detach / STE / 组合 / 关 pred——"
               "回答'中途切换是否有效、两开关是否独立、任务梯度进 router 是否充分'",
           "seeds": SEEDS, "device": "cpu", "runs": []}
    for seed in SEEDS:
        print(f"== seed {seed} ==", flush=True)
        out["runs"].append(run_seed(seed))
    # 跨 seed 汇总
    summary = {}
    for name, _sw in ARMS:
        finals = [r["arms"][name]["test_acc_occ"] for r in out["runs"]]
        summary[name] = {"final_test_occ_mean": round(statistics.mean(finals), 4),
                         "final_test_occ_std_ddof1": round(statistics.stdev(finals), 3),
                         "per_seed": finals}
    out["summary"] = summary
    out["wall_s"] = round(time.perf_counter() - t0, 1)
    dest = OUT / "pred_interventions.json"
    tmp = dest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(dest)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"→ {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
