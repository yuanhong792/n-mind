"""M3-P3：低秩增量神经计算算子与 MAC 核算。

候选公式：
    z        = V @ h          # V:[r,d], h:[d]  →  O(dr)
    delta    = U @ z          # U:[d,r]         →  O(dr)
    h_new    = h + delta      # O(d)
对比稠密： W @ h              # W:[d,d]         →  O(d²)

实现精确 MAC 计数（不是理论估计），并验证低秩等价性与复杂度优势。
"""
from __future__ import annotations

import torch
from torch import nn


def low_rank_step(h: torch.Tensor, V: torch.Tensor, U: torch.Tensor,
                  diag: torch.Tensor | None = None, x_proj: torch.Tensor | None = None,
                  bias: torch.Tensor | None = None) -> torch.Tensor:
    """h_new = diag*h + U(Vh) [+ x_proj + bias]。h:[...,d]。"""
    z = torch.matmul(h, V.transpose(-1, -2))       # [...,r]
    delta = torch.matmul(z, U.transpose(-1, -2))   # [...,d]
    out = delta
    if diag is not None:
        out = out + diag * h
    else:
        out = out + h
    if x_proj is not None:
        out = out + x_proj
    if bias is not None:
        out = out + bias
    return out


def low_rank_macs(d: int, r: int, batch: int = 1, with_diag: bool = True) -> dict:
    """精确乘加数（MAC）：Vh = d*r；U z = r*d；diag*h = d。"""
    return {"Vh": batch * d * r, "Uz": batch * r * d,
            "diag": batch * d if with_diag else 0,
            "total": batch * (2 * d * r + (d if with_diag else 0))}


def dense_macs(d: int, batch: int = 1) -> dict:
    return {"Wh": batch * d * d, "total": batch * d * d}


class LowRankCell(nn.Module):
    """可训练低秩状态更新单元，含对角状态保留与输入注入。"""

    def __init__(self, d: int, r: int, x_dim: int | None = None):
        super().__init__()
        self.d, self.r = d, r
        self.V = nn.Parameter(torch.empty(r, d))
        self.U = nn.Parameter(torch.empty(d, r))
        self.diag = nn.Parameter(torch.ones(d))
        self.bias = nn.Parameter(torch.zeros(d))
        self.x_proj = nn.Linear(x_dim, d, bias=False) if x_dim else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.V, a=5 ** 0.5)
        nn.init.zeros_(self.U)   # 初始为恒等（仅对角保留），训练稳定
        nn.init.ones_(self.diag)

    def forward(self, h: torch.Tensor, x: torch.Tensor | None = None):
        xp = self.x_proj(x) if (self.x_proj is not None and x is not None) else None
        return low_rank_step(h, self.V, self.U, self.diag, xp, self.bias)

    def macs(self, batch=1):
        m = low_rank_macs(self.d, self.r, batch, with_diag=True)
        if self.x_proj:
            m["x_proj"] = batch * self.x_proj.in_features * self.d
            m["total"] += m["x_proj"]
        return m
