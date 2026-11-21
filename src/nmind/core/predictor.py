"""WorldPredictor：在压缩状态上预测下一时刻潜状态，并给出归一化预测误差 E_t。

两条纪律：
1. 预测损失只用 MSE 会奖励"把所有 z 预测成同一个常数"的塌缩解，因此这里同时提供
   方差正则（VICReg 风格的简化形式），由 training 侧一起优化并单独记录。
2. E_t 是路由信号的代理，不是校准过的失败概率；docs/LIMITATIONS.md 明确写出这点。
"""

from __future__ import annotations

import torch
from torch import nn


class WorldPredictor(nn.Module):
    def __init__(self, z_dim: int, action_dim: int, d_w: int, hidden: int = 64) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim + action_dim + d_w, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, z_dim),
        )

    def forward(self, z: torch.Tensor, action_emb: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, action_emb, w], dim=-1)
        return self.net(x)

    # ------------------------------------------------------------------ 误差

    @staticmethod
    def normalized_error(z_next: torch.Tensor, z_hat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """逐样本的相对误差 E_t ∈ 约 [0, ~2]；分母用目标自身的范数，尺度无关。"""
        diff = (z_next - z_hat).pow(2).mean(dim=-1).sqrt()
        scale = z_next.pow(2).mean(dim=-1).sqrt() + eps
        return (diff / scale).unsqueeze(-1)

    @staticmethod
    def cosine_error(z_next: torch.Tensor, z_hat: torch.Tensor) -> torch.Tensor:
        a = nn.functional.normalize(z_next, dim=-1)
        b = nn.functional.normalize(z_hat, dim=-1)
        return (1.0 - (a * b).sum(-1)).unsqueeze(-1)

    @staticmethod
    def variance_regularizer(z: torch.Tensor, target_std: float = 1.0) -> torch.Tensor:
        """塌缩检测/惩罚：每个维度的标准差低于 target_std 就付出代价。

        若网络把所有事件编码成同一个向量，各维标准差趋 0，此项会明显上升。
        """
        std = z.std(dim=0)
        return torch.relu(target_std - std).mean()
