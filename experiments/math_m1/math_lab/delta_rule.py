"""算子 A：Delta 矩阵关联记忆（独立实现，公式见 M1_MATHEMATICAL_SPEC.md）。

状态 S: [..., d_v, d_k]，读取 read = S @ q。
支持共享标量 beta / 标量 alpha / 逐键 alpha。
"""
from __future__ import annotations

import torch
from torch import nn


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x / x.pow(2).sum(dim=dim, keepdim=True).clamp_min(eps).sqrt()


def _with_batch(S, k, v, beta, alpha):
    if S.dim() == 2:
        return (S[None], k[None], v[None],
                beta.reshape(1) if beta.dim() == 0 else beta[None],
                None if alpha is None else (alpha.reshape(1) if alpha.dim() == 0 else alpha[None]),
                True)
    return S, k, v, beta, alpha, False


def _decay_factor(alpha: torch.Tensor, dv: int, dk: int) -> torch.Tensor:
    """alpha: [B] 或 [B, dk] -> [B, 1, dk] 或 [B, 1, 1]。"""
    if alpha.dim() == 1:
        return alpha[:, None, None]
    if alpha.dim() == 2 and alpha.shape[-1] == dk:
        return alpha[:, None, :]
    raise ValueError(f"alpha 形状不支持: {tuple(alpha.shape)}")


def delta_step(S: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
               beta: torch.Tensor, alpha: torch.Tensor | None = None) -> torch.Tensor:
    """S_new = alpha*S + beta*(v - S@k) outer k。"""
    S, k, v, beta, alpha, unbatched = _with_batch(S, k, v, beta, alpha)
    B, dv, dk = S.shape
    decayed = S if alpha is None else _decay_factor(alpha, dv, dk) * S
    pred = torch.bmm(decayed, k.unsqueeze(-1)).squeeze(-1)   # [B,dv] 基于衰减后状态
    residual = v - pred
    gain = beta.view(B, 1, 1).expand(B, dv, dk)
    update = gain * residual.unsqueeze(-1) * k.unsqueeze(1)
    out = decayed + update
    return out[0] if unbatched else out


def delta_read(S: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    if S.dim() == 2:
        return S @ q
    return torch.bmm(S, q.unsqueeze(-1)).squeeze(-1)


class DeltaMemoryOperator(nn.Module):
    """可训练门控参数的最小算子封装；正式模型中的 k/v/q 由网络投影产生。"""

    def __init__(self, dk: int, dv: int):
        super().__init__()
        self.dk, self.dv = dk, dv
        self.log_beta = nn.Parameter(torch.zeros(()))
        self.log_alpha = nn.Parameter(torch.zeros(()))

    def forward(self, S, k, v):
        beta = torch.sigmoid(self.log_beta).to(k.dtype)
        alpha = torch.sigmoid(self.log_alpha).to(k.dtype)
        return delta_step(S, k, v, beta, alpha)
