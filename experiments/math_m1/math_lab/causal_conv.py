"""算子 D：短因果深度可分离卷积。任何时刻只读当前与过去 token。"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class CausalDepthwiseConv(nn.Module):
    def __init__(self, d_model: int, kernel: int = 1):
        super().__init__()
        if kernel not in (1, 2, 4, 8):
            raise ValueError(f"kernel 必须为 1/2/4/8，实际 {kernel}")
        self.kernel = int(kernel)
        self.conv = nn.Conv1d(d_model, d_model, kernel,
                              padding=0, groups=d_model, bias=False) if kernel > 1 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B,T,D] -> [B,T,D]；只在左侧补零。"""
        if self.kernel == 1:
            return x
        xt = x.transpose(1, 2)                      # [B,D,T]
        padded = F.pad(xt, (self.kernel - 1, 0))    # 左填充，未来不入窗
        return self.conv(padded).transpose(1, 2)

    def extra_repr(self) -> str:
        return f"kernel={self.kernel}"
