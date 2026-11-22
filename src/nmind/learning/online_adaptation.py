"""增量学习（neural_adapt）的唯一合法入口与权重取证工具。

三种运行模式（纠错任务书第 6 节）：
  inference    —— 只推理：无 optimizer、无权重变化（默认）。
  memory_only  —— 只写/读外部记忆（SQLite/Replay），参数完全冻结。
  neural_adapt —— 从已验证 checkpoint 出发，受限新经验 + 明确可训练参数集合 + 硬预算。

取证口径（任务书第 2 节，全部按浮点存储值逐位比较，不用容差掩盖变化）：
  * parent/child state sha256 + 逐位继承断言，可追溯权重世系；
  * 冻结参数逐位不变（torch.equal）；哪组参数、哪批经验、多少步、变化多少，全部落盘；
  * transient buffer（router.load_bias、block_load_count）单独记录漂移——它们是训练态
    统计量，不计入"参数变化"证据，也不冒充学习；评测前可复位。
  * 反向检验：lr=0 / max_updates=0 / 全冻结时参数必须逐位不变（本模块的测试钉住）。

块级冻结的实现说明：NCell 的核权重是 [N, in, out] 单张量（每块一份独立权重切片），
无法按块拆 requires_grad。core_partial 用"梯度切片掩码 + 该参数组 weight_decay=0"实现：
掩码保证未授权块的梯度恒为 0；wd=0 保证零梯度元素的 AdamW 更新恰好为 0（带 wd 时
零梯度元素仍会被 decoupled weight decay 拉动），从而冻结切片逐位不变可被验证。
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..core.architecture import NModel
from .losses import LossWeights, compute_total
from .trainer import _assemble

TRANSIENT_BUFFERS = ("block_load_count", "router.load_bias")
ADAPTER_MODULES = ("head", "action_emb", "block_to_ws")
SCOPES = ("frozen", "adapter", "core_partial", "head_only", "head_and_workspace", "selected_ncells")


# ------------------------------------------------------------------ 快照与世系


def state_hash(model: nn.Module, probe: nn.Module | None = None) -> str:
    """state_dict 的 sha256（键名字典序 + 张量原始字节）；probe 可选并入。"""
    h = hashlib.sha256()
    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    for k in sorted(state):
        h.update(f"model:{k}".encode())
        h.update(state[k].numpy().tobytes())
    if probe is not None:
        pstate = {k: v.detach().cpu().contiguous() for k, v in probe.state_dict().items()}
        for k in sorted(pstate):
            h.update(f"probe:{k}".encode())
            h.update(pstate[k].numpy().tobytes())
    return h.hexdigest()


def clone_snapshot(model: nn.Module, probe: nn.Module | None = None) -> dict[str, Any]:
    """参数 + buffer 的完整 CPU 快照（含训练态统计 buffer，便于复位与取证）。"""
    return {"model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "probe": None if probe is None else
            {k: v.detach().cpu().clone() for k, v in probe.state_dict().items()},
            "sha256": state_hash(model, probe)}


def verify_bitwise_inheritance(snapshot: dict[str, Any], model: nn.Module,
                               probe: nn.Module | None = None) -> dict[str, Any]:
    """子模型与父快照逐位比较。任何一键不等都算继承失败，绝不静默放过。"""
    mismatched: list[str] = []
    checked = 0
    child_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    for k, v in snapshot["model"].items():
        checked += 1
        if k not in child_state or not torch.equal(child_state[k], v):
            mismatched.append(f"model:{k}")
    if snapshot.get("probe") and probe is not None:
        pstate = {k: v.detach().cpu() for k, v in probe.state_dict().items()}
        for k, v in snapshot["probe"].items():
            checked += 1
            if k not in pstate or not torch.equal(pstate[k], v):
                mismatched.append(f"probe:{k}")
    return {"all_bitwise_equal": not mismatched, "mismatched_keys": mismatched,
            "checked_tensors": checked}


def reset_transient_buffers(model: NModel, snapshot_model_state: dict[str, torch.Tensor]) -> None:
    """把训练态统计 buffer 复位到快照值（评测前调用，保证跨臂统计口径一致）。"""
    buffers = dict(model.named_buffers())
    with torch.no_grad():
        for name in TRANSIENT_BUFFERS:
            if name in snapshot_model_state and name in buffers:
                buffers[name].copy_(snapshot_model_state[name].to(buffers[name].device))


def _transient_buffers(model: NModel) -> dict[str, torch.Tensor]:
    return {name: buf.detach().clone()
            for name, buf in model.named_buffers() if name in TRANSIENT_BUFFERS}


# ------------------------------------------------------------------ 预算守卫


class BudgetExceeded(RuntimeError):
    """实验预注册的 optimizer.step() 总预算超限。绝不自动放宽。"""


class AdaptationBudget:
    """一次实验内所有实验臂共享的硬预算（任务书第 5 节）。"""

    def __init__(self, max_total_steps: int, *, label: str = "") -> None:
        self.cap = int(max_total_steps)
        self.used = 0
        self.label = label
        self.log: list[dict[str, Any]] = []

    def spend(self, steps: int, arm: str) -> int:
        steps = int(steps)
        if self.used + steps > self.cap:
            raise BudgetExceeded(
                f"预算超限：已用 {self.used} + 请求 {steps} > 上限 {self.cap}（{self.label}）。"
                "按任务书要求停止并汇报，不得继续训练。")
        self.used += steps
        self.log.append({"arm": arm, "steps": steps, "used_after": self.used})
        return self.used


# ------------------------------------------------------------------ 更新范围


def apply_update_scope(model: NModel, *, scope: str = "adapter",
                       core_blocks: int = 2, core_block_ids: list[int] | None = None
                       ) -> dict[str, Any]:
    """冻结/放参数并返回 ScopePlan（trainable/frozen 清单 + 块级梯度掩码规则）。

    scope（M4 任务书第 2 节命名）：
      frozen            —— 全部冻结（memory_only / Control-0 的合法形态）
      head_only         —— 只放开决策头（最小可塑性）
      adapter           —— head + action_emb + block_to_ws（历史口径，兼容保留）
      head_and_workspace —— head + action_emb + block_to_ws + workspace
      core_partial      —— adapter 之外再放开指定 N-Cell 块的核权重（历史口径）
      selected_ncells   —— head_and_workspace + 指定 N-Cell 块（selected_ncells 语义）
    core_block_ids 显式给定放开的块号（预登记用，如父模型负载最高的 2 块）；
    未给定时默认前 core_blocks 个块。
    """
    if scope not in SCOPES:
        raise ValueError(f"scope 只能是 {SCOPES}，实际 {scope!r}")
    for p in model.parameters():
        p.requires_grad_(False)
    trainable: list[str] = []
    if scope == "head_only":
        open_modules: tuple[str, ...] = ("head",)
    elif scope == "adapter":
        open_modules = ADAPTER_MODULES
    elif scope in ("head_and_workspace", "core_partial", "selected_ncells"):
        open_modules = ADAPTER_MODULES + ("workspace",)
    else:
        open_modules = ()
    for name in open_modules:
        for pname, p in getattr(model, name).named_parameters():
            p.requires_grad_(True)
            trainable.append(f"{name}.{pname}")
    mask_rules: list[dict[str, Any]] = []
    chosen: list[int] = []
    if scope in ("core_partial", "selected_ncells"):
        nc = model.blocks
        n_blocks = int(nc.n_blocks)
        if core_block_ids is None:
            core_block_ids = list(range(max(1, int(core_blocks))))
        chosen = sorted({int(i) for i in core_block_ids if 0 <= int(i) < n_blocks})
        if not chosen:
            raise ValueError("core_partial 需要至少一个有效块号")
        for pname, p in nc.named_parameters():
            p.requires_grad_(True)
            trainable.append(f"blocks.{pname}")
            mask_rules.append({"param": f"blocks.{pname}", "keep_blocks": chosen,
                               "n_blocks": n_blocks, "shape": list(p.shape)})
    frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
    return {"scope": scope, "core_blocks_chosen": chosen,
            "trainable_parameter_names": trainable,
            "frozen_parameter_names": frozen, "grad_mask_rules": mask_rules}


def _mask_grads(model: NModel, plan: dict[str, Any]) -> None:
    """把未授权块切片的梯度置 0（裁剪前后各做一次，保证冻结切片严格零梯度）。"""
    if not plan["grad_mask_rules"]:
        return
    by_name = dict(model.named_parameters())
    for rule in plan["grad_mask_rules"]:
        p = by_name[rule["param"]]
        if p.grad is None:
            continue
        keep = torch.zeros(rule["n_blocks"], dtype=p.grad.dtype, device=p.grad.device)
        keep[torch.tensor(rule["keep_blocks"], device=p.grad.device)] = 1.0
        view = [1] * p.grad.dim()
        view[0] = rule["n_blocks"]
        p.grad.mul_(keep.view(view))


# ------------------------------------------------------------------ 参数变化取证


def parameter_delta(before: dict[str, Any], model_after: NModel,
                    plan: dict[str, Any]) -> dict[str, Any]:
    """逐张量取证：范数、delta_l2、delta_max_abs、变化元素数与占比、逐位等价。

    断言口径：trainable 参数至少一个非零变化；frozen 参数逐位不变（torch.equal）。
    state_dict 里的 buffer（如 load_bias/block_load_count）不计入 trainable/frozen
    断言，其漂移由 transient_buffer_drift 单独报告。
    """
    before_state = before["model"]
    after_state = {k: v.detach().cpu() for k, v in model_after.state_dict().items()}
    trainable = set(plan["trainable_parameter_names"])
    frozen = set(plan["frozen_parameter_names"])
    modules: dict[str, Any] = {}
    changed_trainable: list[str] = []
    frozen_violations: list[str] = []
    for k in sorted(set(before_state) | set(after_state)):
        b, a = before_state.get(k), after_state.get(k)
        if b is None or a is None or b.shape != a.shape:
            modules[k] = {"status": "presence_or_shape_mismatch"}
            continue
        diff = a - b
        entry = {"shape": list(a.shape),
                 "norm_before": round(float(b.norm()), 6),
                 "norm_after": round(float(a.norm()), 6),
                 "delta_l2": round(float(diff.norm()), 8),
                 "delta_max_abs": round(float(diff.abs().max()), 8),
                 "changed_elements": int((diff != 0).sum().item()),
                 "changed_ratio": round(float((diff != 0).float().mean().item()), 6),
                 "bitwise_equal": bool(torch.equal(a, b))}
        modules[k] = entry
        if k in trainable:
            if not entry["bitwise_equal"]:
                changed_trainable.append(k)
        elif k in frozen and not entry["bitwise_equal"]:
            frozen_violations.append(k)
    return {"modules": modules,
            "trainable_total": len(trainable),
            "trainable_changed_count": len(changed_trainable),
            "changed_trainable_params": changed_trainable,
            "frozen_total": len(frozen),
            "frozen_violations": frozen_violations,
            "frozen_all_bitwise_unchanged": not frozen_violations}


# ------------------------------------------------------------------ 增量更新入口


def fit_existing_model(model: NModel, *, cfg, rows: list[dict], device,
                       scope: str = "adapter", core_blocks: int = 2,
                       core_block_ids: list[int] | None = None,
                       max_updates: int = 8, sampling_seed: int = 0,
                       lr: float | None = None, probe: nn.Module | None = None,
                       loss_mode: str = "ce", loss_overrides: dict[str, float] | None = None,
                       budget: AdaptationBudget | None = None, arm: str = "arm") -> dict[str, Any]:
    """对**现存模型**做受限增量更新——neural_adapt 的唯一合法入口。

    合同：
      * 不重新初始化：模型以继承态进入（调用方先用 verify_bitwise_inheritance 断言）；
      * optimizer 全新建并明确记录（AdamW，动量不继承任何先前阶段）；
      * 预算硬守卫：budget.spend() 超限即抛 BudgetExceeded；
      * 训练态 buffer 漂移单独记录；frozen 参数逐位不变可被 parameter_delta 验证。
    """
    if loss_mode not in ("ce", "joint"):
        raise ValueError(f"loss_mode 只能是 'ce' 或 'joint'，实际 {loss_mode!r}")
    if budget is not None:
        budget.spend(max_updates, arm)
    before = clone_snapshot(model, probe)
    plan = apply_update_scope(model, scope=scope, core_blocks=core_blocks,
                              core_block_ids=core_block_ids)
    weights = (LossWeights.from_cfg(cfg) if loss_mode == "joint"
               else LossWeights(pred=0.0, var=0.0, balance=0.0, entropy=0.0, probe=0.0))
    for k, v in (loss_overrides or {}).items():
        if not hasattr(weights, k):
            raise ValueError(f"未知损失权重 {k!r}；可用：{list(weights.__dataclass_fields__)}")
        setattr(weights, k, float(v))

    core_names = {r["param"] for r in plan["grad_mask_rules"]}
    adapter_params = [p for n, p in model.named_parameters()
                      if p.requires_grad and n not in core_names]
    core_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and n in core_names]
    groups: list[dict[str, Any]] = []
    if adapter_params:
        groups.append({"params": adapter_params, "weight_decay": 1e-4})
    if core_params:
        # wd=0：零梯度切片的 AdamW 更新恰好为 0，冻结块切片才能逐位不变
        groups.append({"params": core_params, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups, lr=float(cfg.learning.lr if lr is None else lr)) \
        if groups else None
    rng = np.random.default_rng(int(sampling_seed))
    bs = max(2, int(cfg.learning.batch_size))
    grad_clip = float(cfg.learning.grad_clip)
    params = [p for p in model.parameters() if p.requires_grad]
    transient_before = _transient_buffers(model)

    model.train()
    action_losses: list[float] = []
    grad_norms_first: dict[str, float] = {}
    steps_done = 0
    effective_samples = 0
    pred_mode_eff = model.pred_mode
    t0 = time.perf_counter()
    for _ in range(int(max_updates)):
        if opt is None:
            break                      # frozen scope：没有可训练参数，一步都不走
        pick = [int(i) for i in rng.integers(0, len(rows), size=bs)]
        b = _assemble([rows[i] for i in pick], device)
        out = model.forward_sequence(b["obs"], b["actions"], mem=b["mem"], depth=1)
        losses = compute_total(out, b["actions"], weights,
                               pred_shift=(pred_mode_eff == "next"))
        if loss_mode == "joint" and probe is not None:
            p_logits = probe(out["ctx"].reshape(-1, out["ctx"].shape[-1]))
            codes = b["codes"].repeat_interleave(b["obs"].shape[1])
            probe_loss = F.cross_entropy(p_logits, codes)
            losses["total"] = losses["total"] + weights.probe * probe_loss
        if not torch.isfinite(losses["total"]):
            opt.zero_grad(set_to_none=True)
            continue
        opt.zero_grad(set_to_none=True)
        losses["total"].backward()
        _mask_grads(model, plan)
        if steps_done == 0:
            grad_norms_first = {n: round(float(p.grad.norm()), 6)
                                for n, p in model.named_parameters()
                                if p.requires_grad and p.grad is not None}
        if params:
            nn.utils.clip_grad_norm_(params, grad_clip)
        _mask_grads(model, plan)     # 裁剪后再补一次：冻结切片严格零梯度
        opt.step()
        steps_done += 1
        effective_samples += bs
        action_losses.append(float(losses["action"].detach()))
    wall = time.perf_counter() - t0
    model.eval()
    transient_after = _transient_buffers(model)
    drift = {k: {"changed": not torch.equal(transient_before[k], transient_after[k]),
                 "delta_l2": round(float((transient_after[k] - transient_before[k]).norm()), 6)}
             for k in transient_before}
    delta = parameter_delta(before, model, plan)
    return {"arm": arm, "scope": plan["scope"], "core_blocks_chosen": plan["core_blocks_chosen"],
            "optimizer_steps_executed": steps_done, "effective_samples": effective_samples,
            "optimizer_init": {"kind": "AdamW(new)", "lr": float(cfg.learning.lr if lr is None else lr),
                                "weight_decay": {"adapter_group": 1e-4, "core_group": 0.0},
                                "momentum_inherited": False,
                                "note": "增量阶段 optimizer 新建，动量不从任何先前阶段继承（任务书允许，须记录）"},
            "sampling_seed": int(sampling_seed), "batch_size": bs,
            "loss_action_first": round(action_losses[0], 4) if action_losses else None,
            "loss_action_last": round(action_losses[-1], 4) if action_losses else None,
            "grad_norms_first_step": grad_norms_first,
            "transient_buffer_drift": drift,
            "plan": {"scope": plan["scope"], "core_blocks_chosen": plan["core_blocks_chosen"],
                     "trainable_parameter_names": plan["trainable_parameter_names"],
                     "frozen_parameter_names": plan["frozen_parameter_names"],
                     "grad_mask_rules": plan["grad_mask_rules"]},
            "delta": delta,
            "parent_state_sha256": before["sha256"],
            "child_state_sha256": state_hash(model, probe),
            "wall_s": round(wall, 2), "device": str(device)}


__all__ = ["AdaptationBudget", "BudgetExceeded", "apply_update_scope", "clone_snapshot",
           "fit_existing_model", "parameter_delta", "reset_transient_buffers",
           "state_hash", "verify_bitwise_inheritance"]
