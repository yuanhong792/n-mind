"""可塑性辅助：锚定惩罚（candidate 对 stable 的参数锚）+ 旧经验回放混合。

对应 M4 任务书第 5 节最小算法的 loss 项：
    loss = CE(new) + lambda_replay * CE(replay) + lambda_anchor * anchor(candidate, stable)
锚定只作用于**允许更新的参数**（requires_grad），冻结参数不参与也不受影响。
"""

from __future__ import annotations

import torch
from torch import nn


def anchor_penalty(model: nn.Module, stable_state: dict, *,
                   names: list[str] | None = None) -> torch.Tensor:
    """candidate 可训练参数相对 stable 快照的均方偏差（逐参数 mean 后再平均）。"""
    dev = next(model.parameters()).device
    total = torch.zeros((), device=dev)
    n = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if names is not None and name not in names:
            continue
        if name not in stable_state:
            continue
        sp = stable_state[name].to(dev)
        total = total + (p - sp).pow(2).mean()
        n += 1
    return total / max(1, n)


def sample_replay(replay_rows: list[dict], rng, k: int) -> list[dict]:
    """从旧任务经验池抽 k 条（rng 由调用方持有，保证可复现）。"""
    if not replay_rows or k <= 0:
        return []
    k = min(k, len(replay_rows))
    pick = [int(i) for i in rng.integers(0, len(replay_rows), size=k)]
    return [replay_rows[i] for i in pick]
