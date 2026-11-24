"""算子 M2-AM：关联记忆（Associative Memory）与可学习保护门。

状态 S: [B, d_v, d_k]，key k: [B, d_k]（L2 归一），value v: [B, d_v]。
写入分两步：先用衰减 alpha 与写入门 beta 产生候选增量，再由保护门 g 决定
「新候选替换多少旧记忆」：
    decayed = alpha * S
    candidate = decayed + beta * (v - decayed@k) outer k
    S_new = S + g * (candidate - S)          # g∈[0,1]，逐样本或逐通道
当 g=1 退化为标准 delta 写入；当 g=0 完全保护旧记忆（抗干扰）。
保护门由 memory_protection.py 提供，本文件只实现记忆算子本身。
"""
from __future__ import annotations

import torch


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x / x.pow(2).sum(dim=dim, keepdim=True).clamp_min(eps).sqrt()


def _as_batch(S, k, v, beta, alpha, protect):
    if S.dim() == 2:
        def b1(t):
            return t.reshape(1) if t is None or t.dim() == 0 else t[None]
        return (S[None], k[None], v[None], b1(beta),
                None if alpha is None else b1(alpha),
                None if protect is None else b1(protect), True)
    return S, k, v, beta, alpha, protect, False


def _expand_gate(gate: torch.Tensor, B: int, dv: int, dk: int) -> torch.Tensor:
    """把 [B] / [B,dk] / [B,dv] / [B,dv,dk] 的 gate 广播到 [B,dv,dk]。"""
    if gate.dim() == 0:
        return gate.view(1, 1, 1).expand(B, dv, dk)
    if gate.dim() == 1:
        return gate[:, None, None].expand(B, dv, dk)
    if gate.dim() == 2:
        if gate.shape[-1] == dk:
            return gate[:, None, :].expand(B, dv, dk)
        if gate.shape[-1] == dv:
            return gate[:, :, None].expand(B, dv, dk)
    if gate.dim() == 3:
        return gate
    raise ValueError(f"gate 形状不支持: {tuple(gate.shape)}")


def am_write(S: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
             beta: torch.Tensor, alpha: torch.Tensor | None = None,
             protect: torch.Tensor | None = None) -> torch.Tensor:
    """带保护门的 delta 写入。protect=1 表示完全接受候选，0 表示完全保护。"""
    S, k, v, beta, alpha, protect, unbatched = _as_batch(S, k, v, beta, alpha, protect)
    B, dv, dk = S.shape
    decayed = S if alpha is None else alpha[:, None, None] * S
    pred = torch.bmm(decayed, k.unsqueeze(-1)).squeeze(-1)   # [B,dv]
    residual = v - pred
    candidate = decayed + beta[:, None, None] * residual.unsqueeze(-1) * k.unsqueeze(1)
    if protect is None:
        out = candidate
    else:
        g = _expand_gate(protect, B, dv, dk)
        out = S + g * (candidate - S)
    return out[0] if unbatched else out


def am_read(S: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    if S.dim() == 2:
        return S @ q
    return torch.bmm(S, q.unsqueeze(-1)).squeeze(-1)


def am_erase(S: torch.Tensor, k: torch.Tensor, erase: torch.Tensor) -> torch.Tensor:
    """按 key 方向擦除：S -= erase * (S@k) outer k，erase 可为 [B]/[B,dv]/标量。"""
    unbatched = S.dim() == 2
    if unbatched:
        S, k = S[None], k[None]
        if erase.dim() == 0:
            erase = erase.reshape(1)
    B, dv, dk = S.shape
    proj = torch.bmm(S, k.unsqueeze(-1)).squeeze(-1)         # [B,dv]
    e = _expand_gate(erase, B, dv, dk) if erase.dim() > 1 else erase[:, None, None]
    out = S - e * proj.unsqueeze(-1) * k.unsqueeze(1)
    return out[0] if unbatched else out


class AssociativeMemoryOperator(torch.nn.Module):
    def __init__(self, dk: int, dv: int):
        super().__init__()
        self.dk, self.dv = dk, dv
        self.log_beta = torch.nn.Parameter(torch.zeros(()))
        self.log_alpha = torch.nn.Parameter(torch.zeros(()))

    def forward(self, S, k, v, protect=None):
        beta = torch.sigmoid(self.log_beta).to(k.dtype)
        alpha = torch.sigmoid(self.log_alpha).to(k.dtype)
        return am_write(S, k, v, beta, alpha, protect)
