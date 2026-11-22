"""有界在线流更新：verified 经验 + 旧任务回放 + 参数锚定（M4 任务书第 5 节最小算法）。

    loss = CE(new_batch) + lambda_replay * CE(replay_batch) + lambda_anchor * anchor(candidate, stable)

合同：
  * model 必须是 stable 的**独立副本**（调用方先做逐位继承断言）；stable 永不被触碰；
  * budget（AdaptationBudget）硬守卫，超限抛 BudgetExceeded；
  * 非有限损失 → 立即拒绝候选（rejected="non_finite"），绝不带病继续；
  * 冻结参数逐位不变由 parameter_delta 证明（梯度切片掩码 + wd=0 组）；
  * 负对照：label_permutation_seed 给定时打乱反馈标签映射——若同样提升，
    应怀疑泄漏或评测错误（任务书第 5/8 节）。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..core.architecture import NModel
from .losses import LossWeights, compute_total
from .online_adaptation import (AdaptationBudget, apply_update_scope, clone_snapshot,
                                parameter_delta, state_hash)
from .online_adaptation import _mask_grads
from .plasticity import anchor_penalty
from .trainer import _assemble


def bounded_stream_update(model: NModel, *, cfg, rows: list[dict], device: torch.device,
                          stable_state: dict[str, torch.Tensor], scope: str = "adapter",
                          core_blocks: int = 2, core_block_ids: list[int] | None = None,
                          max_updates: int = 40, sampling_seed: int = 0,
                          lr: float | None = None, replay_rows: list[dict] | None = None,
                          replay_k: int = 4, lambda_replay: float = 0.3,
                          lambda_anchor: float = 0.1,
                          label_permutation_seed: int | None = None,
                          budget: AdaptationBudget | None = None,
                          arm: str = "arm") -> dict[str, Any]:
    if budget is not None:
        budget.spend(max_updates, arm)
    before = clone_snapshot(model)
    plan = apply_update_scope(model, scope=scope, core_blocks=core_blocks,
                              core_block_ids=core_block_ids)
    weights = LossWeights(pred=0.0, var=0.0, balance=0.0, entropy=0.0, probe=0.0)  # 纯 CE
    core_names = {r["param"] for r in plan["grad_mask_rules"]}
    adapter_params = [p for n, p in model.named_parameters()
                      if p.requires_grad and n not in core_names]
    core_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and n in core_names]
    groups: list[dict[str, Any]] = []
    if adapter_params:
        groups.append({"params": adapter_params, "weight_decay": 1e-4})
    if core_params:
        groups.append({"params": core_params, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups, lr=float(cfg.learning.lr if lr is None else lr)) \
        if groups else None
    rng = np.random.default_rng(int(sampling_seed))
    perm_rng = (np.random.default_rng(int(label_permutation_seed))
                if label_permutation_seed is not None else None)
    perm_table = (np.argsort(perm_rng.permutation(4)) if perm_rng is not None else None)
    bs = max(2, int(cfg.learning.batch_size))
    grad_clip = float(cfg.learning.grad_clip)
    params = [p for p in model.parameters() if p.requires_grad]
    transient_before = {n: b.detach().clone() for n, b in model.named_buffers()
                        if n in ("block_load_count", "router.load_bias")}

    model.train()
    losses_hist: list[float] = []
    grad_norms_first: dict[str, float] = {}
    steps_done = 0
    rejected: str | None = None
    pred_mode_eff = model.pred_mode
    t0 = __import__("time").perf_counter()
    for _ in range(int(max_updates)):
        if opt is None:
            rejected = "no_trainable_params"
            break
        pick = [int(i) for i in rng.integers(0, len(rows), size=bs)]
        batch_rows = [rows[i] for i in pick]
        b = _assemble(batch_rows, device)
        if perm_table is not None:
            # 负对照：打乱反馈标签映射（分布不变、与观测的相关性被破坏）
            shuffled = perm_table[np.asarray(b["actions"].cpu())]
            b = {**b, "actions": torch.tensor(shuffled, device=device, dtype=torch.long)}
        out = model.forward_sequence(b["obs"], b["actions"], mem=b["mem"], depth=1)
        losses = compute_total(out, b["actions"], weights,
                               pred_shift=(pred_mode_eff == "next"))
        loss = losses["total"]
        if replay_rows and lambda_replay > 0:
            rpick = [int(i) for i in rng.integers(0, len(replay_rows), size=replay_k)]
            rb = _assemble([replay_rows[i] for i in rpick], device)
            rout = model.forward_sequence(rb["obs"], rb["actions"], mem=rb["mem"], depth=1)
            rlosses = compute_total(rout, rb["actions"], weights,
                                    pred_shift=(pred_mode_eff == "next"))
            loss = loss + float(lambda_replay) * rlosses["total"]
        if lambda_anchor > 0:
            loss = loss + float(lambda_anchor) * anchor_penalty(model, stable_state)
        if not torch.isfinite(loss):
            opt.zero_grad(set_to_none=True)
            rejected = "non_finite"
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        _mask_grads(model, plan)
        if steps_done == 0:
            grad_norms_first = {n: round(float(p.grad.norm()), 6)
                                for n, p in model.named_parameters()
                                if p.requires_grad and p.grad is not None}
        if params:
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
        _mask_grads(model, plan)
        opt.step()
        steps_done += 1
        losses_hist.append(float(loss.detach()))
    wall = __import__("time").perf_counter() - t0
    model.eval()
    transient_after = {n: b.detach().clone() for n, b in model.named_buffers()
                       if n in ("block_load_count", "router.load_bias")}
    drift = {k: {"changed": not torch.equal(transient_before[k], transient_after[k]),
                 "delta_l2": round(float((transient_after[k] - transient_before[k]).norm()), 6)}
             for k in transient_before}
    return {"arm": arm, "scope": plan["scope"],
            "core_blocks_chosen": plan["core_blocks_chosen"],
            "optimizer_steps_executed": steps_done, "effective_samples": steps_done * bs,
            "rejected": rejected,
            "status": "REJECTED" if rejected else "TRAINED",
            "optimizer_init": {"kind": "AdamW(new)",
                                "lr": float(cfg.learning.lr if lr is None else lr),
                                "weight_decay": {"adapter_group": 1e-4, "core_group": 0.0},
                                "momentum_inherited": False},
            "sampling_seed": int(sampling_seed),
            "label_permutation_seed": label_permutation_seed,
            "lambda_replay": float(lambda_replay), "lambda_anchor": float(lambda_anchor),
            "replay_k": int(replay_k), "replay_rows_n": len(replay_rows or []),
            "loss_first": round(losses_hist[0], 4) if losses_hist else None,
            "loss_last": round(losses_hist[-1], 4) if losses_hist else None,
            "grad_norms_first_step": grad_norms_first,
            "transient_buffer_drift": drift,
            "plan": {"scope": plan["scope"], "core_blocks_chosen": plan["core_blocks_chosen"],
                     "trainable_parameter_names": plan["trainable_parameter_names"],
                     "frozen_parameter_names": plan["frozen_parameter_names"]},
            "delta": parameter_delta(before, model, plan),
            "parent_state_sha256": before["sha256"],
            "child_state_sha256": state_hash(model),
            "wall_s": round(wall, 2), "device": str(device)}
