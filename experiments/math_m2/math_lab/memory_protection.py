"""算子 M2-PG：可学习保护门（Protection Gate）。

g_t = sigmoid(w_g · h_t + b_g) ∈ [0,1]，控制关联记忆写入时对旧记忆的保留：
    S_new = S + g_t * (candidate - S)
g 由当前隐藏状态与候选/旧记忆的相似度共同决定，使模型能自主学习
「遇到干扰写入时关闭更新、遇到目标写入时打开更新」。
本文件提供纯函数与最小可训练封装。
"""
from __future__ import annotations

import torch
from torch import nn


def protection_from_logits(logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits)


def protect_step(S_old: torch.Tensor, S_candidate: torch.Tensor,
                 gate: torch.Tensor) -> torch.Tensor:
    """S = S_old + gate*(S_candidate - S_old)，gate 可广播到 S 形状。"""
    if gate.dim() == 1:
        g = gate[:, None, None]
    elif gate.dim() == 2:
        g = gate[:, :, None] if gate.shape[-1] == S_old.shape[1] else gate[:, None, :]
    elif gate.dim() == 3:
        g = gate
    else:
        raise ValueError(f"gate 形状不支持: {tuple(gate.shape)}")
    return S_old + g * (S_candidate - S_old)


def similarity_gate(h: torch.Tensor, old_read: torch.Tensor,
                    cand_read: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """基于旧/候选读出与查询差异的自适应保护 logit。
    weight: [B,d] 或可学习向量；差异越大 gate 越小（更保护旧记忆）。"""
    delta = (cand_read - old_read)
    score = (delta * weight).sum(-1)          # [B] 或 [B,d]
    if score.dim() > 1:
        score = score.mean(-1)
    return score


class ProtectionGate(nn.Module):
    def __init__(self, hidden: int, mode: str = "scalar"):
        super().__init__()
        if mode not in ("scalar", "per_value"):
            raise ValueError("mode must be scalar|per_value")
        self.mode = mode
        out = 1 if mode == "scalar" else hidden
        self.w = nn.Linear(hidden, out)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.w(h))
