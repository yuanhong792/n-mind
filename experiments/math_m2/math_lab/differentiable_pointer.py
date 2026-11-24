"""算子 M2-PTR：可微分读写指针（Neural Read/Write Pointer）。

维护固定容量 L 的位置分布 p ∈ simplex^L。支持：
  - content addressing：p ∝ softmax(beta * cos(key, M))
  - location shift：p' = conv_shift(p, shift_kernel)，循环/截断两种模式
  - interpolation：p' = (1-i)*p + i*p_candidate
  - write/read：write 用 p 加权写入；read 用 p 加权读出
全部操作可微，且对 batch 逐样本独立。
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def l2_normalize(x, dim=-1, eps=1e-6):
    return x / x.pow(2).sum(dim=dim, keepdim=True).clamp_min(eps).sqrt()


def content_address(M: torch.Tensor, key: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """M:[B,L,d], key:[B,d], beta:[B] -> p:[B,L]。"""
    m = l2_normalize(M, dim=-1)
    k = l2_normalize(key, dim=-1).unsqueeze(-1)
    sim = torch.bmm(m, k).squeeze(-1)             # [B,L]
    return F.softmax(beta[:, None] * sim, dim=-1)


def shift_pointer(p: torch.Tensor, kernel: torch.Tensor, circular: bool = True) -> torch.Tensor:
    """p:[B,L], kernel:[B,K]（K 为奇数）-> p':[B,L]；沿位置轴做循环/截断位移。
    用离散移位叠加，避免 conv1d 的 groups/channel 歧义，且完全可微。"""
    B, L = p.shape
    K = kernel.shape[-1]
    mid = K // 2
    if not circular:
        kernel = kernel / kernel.sum(-1, keepdim=True).clamp_min(1e-6)
    out = torch.zeros_like(p)
    for off in range(K):
        shift = off - mid                          # off=mid 表示不移动
        w = kernel[:, off]                         # [B]
        if abs(shift) == 0:
            out = out + w[:, None] * p
        else:
            src = torch.roll(p, shifts=shift, dims=-1) if circular else _shift_trunc(p, shift)
            out = out + w[:, None] * src
    return out


def _shift_trunc(p: torch.Tensor, shift: int) -> torch.Tensor:
    """非循环移位，越界补零。"""
    B, L = p.shape
    out = torch.zeros_like(p)
    if shift > 0:
        out[:, shift:] = p[:, :L - shift]
    elif shift < 0:
        out[:, :shift] = p[:, -shift:]
    else:
        out = p.clone()
    return out


def interpolate(old: torch.Tensor, new: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    g = gate[:, None] if gate.dim() == 1 else gate
    return (1.0 - g) * old + g * new


def pointer_write(M: torch.Tensor, p: torch.Tensor, v: torch.Tensor,
                  erase: torch.Tensor | None = None) -> torch.Tensor:
    """M:[B,L,d], p:[B,L], v:[B,d] -> M_new:[B,L,d]。"""
    add = p.unsqueeze(-1) * v.unsqueeze(1)         # [B,L,d]
    if erase is None:
        return M + add
    es = torch.sigmoid(erase)
    if es.dim() == 1:
        es = es[:, None, None]                      # [B,1,1]
    elif es.dim() == 2:
        es = es[:, :, None]                          # [B,L,1]
    return M * (1.0 - p.unsqueeze(-1) * es) + add


def pointer_read(M: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    return torch.bmm(p.unsqueeze(1), M).squeeze(1)   # [B,d]


class PointerCell(nn.Module):
    """一个可微指针单元：从旧指针经 content/shift/interp 得到新指针。

    h_dim: 控制器隐藏维度；mem_dim: 序列记忆槽位特征维度。
    """

    def __init__(self, L: int, mem_dim: int, h_dim: int | None = None, kernel_size: int = 3):
        super().__init__()
        self.L = L
        h_dim = h_dim or mem_dim
        self.K = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        self.key_proj = nn.Linear(h_dim, mem_dim)     # 投影到记忆特征维度
        self.beta_proj = nn.Linear(h_dim, 1)
        self.interp_proj = nn.Linear(h_dim, 1)
        self.shift_logits = nn.Parameter(torch.zeros(self.K))
        self.pad = self.K // 2

    def initial_pointer(self, batch: int, device) -> torch.Tensor:
        p = torch.zeros(batch, self.L, device=device)
        p[:, 0] = 1.0
        return p

    def step(self, M: torch.Tensor, p_old: torch.Tensor, h: torch.Tensor,
             use_content: bool = True) -> torch.Tensor:
        beta = torch.sigmoid(self.beta_proj(h)).squeeze(-1)
        interp = torch.sigmoid(self.interp_proj(h)).squeeze(-1)
        shift = F.softmax(self.shift_logits, dim=-1).expand(M.size(0), -1)
        p_shift = shift_pointer(p_old, shift, circular=True)
        if use_content:
            key = self.key_proj(h)
            p_content = content_address(M, key, beta)
            p_cand = interpolate(p_shift, p_content, interp)
        else:
            p_cand = p_shift
        p_cand = p_cand / p_cand.sum(-1, keepdim=True).clamp_min(1e-8)
        return interpolate(p_old, p_cand, torch.ones_like(interp) * 0.9)
