"""算子 M2-DYN：动力学稳定性分析工具。

用于诊断重复生成与状态循环：
  - state_norm：状态范数随时间的增长/爆炸检测；
  - step_distance：相邻状态变化距离；
  - cos_adjacent：相邻状态余弦（接近 1 表示陷入不动点/重复）；
  - token_entropy：输出分布熵（塌缩检测）；
  - fixed_point_ratio：连续多步状态几乎不变的比例（重复生成征兆）。
"""
from __future__ import annotations

import torch


def state_norm(states: torch.Tensor) -> torch.Tensor:
    """states:[T,B,...] -> [T] 每步范数均值。"""
    return torch.stack([s.float().pow(2).mean().sqrt() for s in states])


def step_distance(states: torch.Tensor) -> torch.Tensor:
    ds = []
    for i in range(1, len(states)):
        d = (states[i].float() - states[i - 1].float()).pow(2).mean().sqrt()
        ds.append(d)
    return torch.stack(ds) if ds else torch.tensor([])


def cos_adjacent(states: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """支持 1-D（单样本状态向量 [d]）与 2-D（[B,d]）。"""
    cs = []
    flat = states.float().reshape(len(states), -1)   # [T, D]
    for i in range(1, flat.shape[0]):
        a, b = flat[i - 1], flat[i]
        c = (a * b).sum() / (a.norm() * b.norm() + eps)
        cs.append(c)
    return torch.stack(cs) if cs else torch.tensor([])


def token_entropy(logits: torch.Tensor) -> torch.Tensor:
    """logits:[T,V] -> [T] 分布熵。"""
    p = torch.softmax(logits.float(), dim=-1)
    return -(p * p.clamp_min(1e-12).log()).sum(-1)


def fixed_point_ratio(states: torch.Tensor, tol: float = 1e-4) -> float:
    ds = step_distance(states)
    if ds.numel() == 0:
        return 0.0
    return float((ds < tol).float().mean())


def analyze(states: torch.Tensor, logits: torch.Tensor | None = None) -> dict:
    out = {
        "state_norm": [round(float(x), 5) for x in state_norm(states)],
        "step_distance": [round(float(x), 5) for x in step_distance(states)],
        "cos_adjacent": [round(float(x), 5) for x in cos_adjacent(states)],
        "fixed_point_ratio": round(fixed_point_ratio(states), 5),
    }
    if logits is not None:
        out["token_entropy"] = [round(float(x), 5) for x in token_entropy(logits)]
        out["entropy_collapse"] = round(float(token_entropy(logits)[-3:].mean()), 5)
    return out
