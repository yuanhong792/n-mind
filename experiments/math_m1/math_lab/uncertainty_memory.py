"""算子 C：Kalman 对角不确定性关联记忆（独立实现；机制来源 KDN）。

仅用于数学算子实验；未经 math_verifier 验证不得进入正式主模型。
状态：S[d_v,d_k]，每 key 通道的不确定性 p[d_k]。
"""
from __future__ import annotations

import torch
from torch import nn


def diagonal_kalman_step(S: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                         p: torch.Tensor, alpha: torch.Tensor, omega: torch.Tensor,
                         r: torch.Tensor, mu: float = 1.0):
    """返回 S_new, p_new, kappa, residual, obs_error。

    p/alpha/omega: [B,dk] 或 [dk]；r: [B] 或 标量。
    """
    unbatched = S.dim() == 2
    if unbatched:
        S, k, v = S[None], k[None], v[None]
        p, alpha, omega = p[None], alpha[None], omega[None]
        if torch.is_tensor(r) and r.dim() > 0:
            r = r[None]
    B, dv, dk = S.shape

    p_hat = alpha.pow(2) * p + omega
    k2 = k.pow(2)
    denom = r.reshape(B, 1) + (p_hat * k2).sum(dim=1, keepdim=True)
    kappa = p_hat * k / denom                                   # [B,dk]
    pred = torch.bmm(S, k.unsqueeze(-1)).squeeze(-1)             # [B,dv]
    residual = v - pred
    S_new = S + residual.unsqueeze(-1) * kappa.unsqueeze(1)      # [B,dv,dk]
    p_new = 1.0 / (1.0 / p_hat.clamp_min(1e-8) + (mu / r).reshape(B, 1) * k2)
    obs_error = (residual.pow(2).mean(dim=1, keepdim=True) / denom).squeeze(-1)
    if unbatched:
        return S_new[0], p_new[0], kappa[0], residual[0], obs_error[0]
    return S_new, p_new, kappa, residual, obs_error


class UncertaintyMemoryOperator(nn.Module):
    """可训练噪声参数的最小对角 KDN 算子。"""

    def __init__(self, dk: int, dv: int):
        super().__init__()
        self.dk, self.dv = dk, dv
        self.log_r = nn.Parameter(torch.zeros(()))
        self.log_omega = nn.Parameter(torch.full((dk,), -3.0))
        self.log_alpha = nn.Parameter(torch.zeros(dk))

    def forward(self, S, k, v, p=None, mu: float = 1.0):
        B, dv, dk = S.shape
        if p is None:
            p = torch.ones(B, dk, device=S.device, dtype=S.dtype)
        r = torch.sigmoid(self.log_r).clamp_min(1e-3).expand(B, 1)
        omega = torch.sigmoid(self.log_omega).expand(B, dk)
        alpha = torch.sigmoid(self.log_alpha).expand(B, dk)
        return diagonal_kalman_step(S, k, v, p, alpha, omega, r, mu=mu)
