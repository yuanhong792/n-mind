"""算子 M2-CONV：短因果深度可分离卷积 + 位置编码。

因果性：位置 t 的输出只依赖 0..t，未来 token 不进入窗口（左 padding）。
支持核大小 2/4/8；用于给字符级序列注入局部位置/邻接信号。
位置编码：正弦固定编码（不涉及可学习参数，便于长度外推对照）。
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class CausalDepthwiseConv(nn.Module):
    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.k = kernel_size
        self.dw = nn.Conv1d(channels, channels, kernel_size,
                            padding=0, groups=channels, bias=False)
        self.pw = nn.Conv1d(channels, channels, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,C] -> [B,C,T]
        z = x.transpose(1, 2)
        z = F.pad(z, (self.k - 1, 0), mode="constant", value=0.0)
        z = self.dw(z)
        z = F.gelu(z)
        z = self.pw(z)
        return z.transpose(1, 2)


def sinusoidal_pos(L: int, d: int, device=None) -> torch.Tensor:
    pos = torch.arange(L, device=device).float()[:, None]
    idx = torch.arange(d, device=device).float()[None, :]
    angle = pos / (10000.0 ** (idx / d))
    enc = torch.empty(L, d, device=device)
    enc[:, 0::2] = torch.sin(angle[:, 0::2])
    enc[:, 1::2] = torch.cos(angle[:, 1::2])
    return enc
