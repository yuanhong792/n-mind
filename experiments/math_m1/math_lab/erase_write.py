"""算子 B：解耦擦除/写入门（独立实现；机制来源 Gated DeltaNet-2）。

论文以 S[d_k,d_v] 书写：
    S_t = (I - k (b*k)^T) D S_{t-1} + k (w*v)^T
本项目统一采用 S[d_v,d_k]、read = S @ q，因此转置后等价实现为：
    S'_t = D S'_{t-1} (I - (b*k) k^T) + (w*v) k^T
         = decayed - (decayed @ (b*k)) outer k + (w*v) outer k
共享门 b=w=beta 时与 gated delta rule 完全一致。
"""
from __future__ import annotations

import torch
from torch import nn


def _prep(S, k, v, b, w, alpha):
    if S.dim() == 2:
        return (S[None], k[None], v[None],
                b.reshape(1) if b.dim() == 0 else b[None],
                w.reshape(1) if w.dim() == 0 else w[None],
                None if alpha is None else (alpha.reshape(1) if alpha.dim() == 0 else alpha[None]),
                True)
    return S, k, v, b, w, alpha, False


def _gate_shape(g: torch.Tensor, B: int, d: int) -> torch.Tensor:
    if g.dim() == 0 or (g.dim() == 1 and g.numel() == 1):
        return g.reshape(1, 1).expand(B, d)
    if g.dim() == 1 and g.shape[0] == B:
        return g[:, None].expand(B, d)
    if g.dim() == 2 and tuple(g.shape) == (B, d):
        return g
    raise ValueError(f"门形状不支持: {tuple(g.shape)}，期望 ({B},{d}) 或标量")


def _decay_shape(alpha: torch.Tensor | None, B: int, dk: int) -> torch.Tensor:
    if alpha is None:
        return torch.ones(B, dk)
    if alpha.dim() == 0 or (alpha.dim() == 1 and alpha.numel() == 1):
        return alpha.reshape(1).expand(B, dk)
    if alpha.dim() == 1 and alpha.shape[0] == B:
        return alpha[:, None].expand(B, dk)
    if alpha.dim() == 2 and tuple(alpha.shape) == (B, dk):
        return alpha
    raise ValueError(f"alpha 形状不支持: {tuple(alpha.shape)}")


def gated_delta2_step(S: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                      b: torch.Tensor, w: torch.Tensor,
                      alpha: torch.Tensor | None = None) -> torch.Tensor:
    """独立擦除门 b[key] / 写入门 w[value]，S[d_v,d_k]。"""
    S, k, v, b, w, alpha, unbatched = _prep(S, k, v, b, w, alpha)
    B, dv, dk = S.shape
    b = _gate_shape(b, B, dk)
    w = _gate_shape(w, B, dv)
    decay = _decay_shape(alpha, B, dk)

    decayed = S * decay[:, None, :]                                  # [B,dv,dk]
    bk = b * k                                                       # [B,dk]
    proj = torch.bmm(decayed, bk.unsqueeze(-1)).squeeze(-1)           # [B,dv]
    erased = decayed - proj.unsqueeze(-1) * k.unsqueeze(1)            # [B,dv,dk]
    written = (w * v).unsqueeze(-1) * k.unsqueeze(1)                  # [B,dv,dk]
    out = erased + written
    return out[0] if unbatched else out


def shared_delta_step(S: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                      beta: torch.Tensor, alpha: torch.Tensor | None = None) -> torch.Tensor:
    """共享门版本：b=w=beta，应保持与 delta_step 数值一致。"""
    return gated_delta2_step(S, k, v, beta, beta, alpha)


class EraseWriteOperator(nn.Module):
    """最小门控算子封装；正式模型中的 b/w 由网络投影产生。"""

    def __init__(self, dk: int, dv: int):
        super().__init__()
        self.dk, self.dv = dk, dv
        self.log_b = nn.Parameter(torch.zeros(dk))
        self.log_w = nn.Parameter(torch.zeros(dv))
        self.log_alpha = nn.Parameter(torch.zeros(()))

    def forward(self, S, k, v, shared=False):
        b = torch.sigmoid(self.log_b)
        w = torch.sigmoid(self.log_w)
        if shared:
            beta = torch.sigmoid(self.log_b.mean())
            b = w = beta
        alpha = torch.sigmoid(self.log_alpha)
        return gated_delta2_step(S, k, v, b, w, alpha)
