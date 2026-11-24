"""P3：Low-Rank Information Interaction（LRII）算子实验室。

严格区分三种不同的"低秩"机制（不可混为一谈）：
  1. reduce_qk_dim：Q_r = X U_Q, K_r = X U_K, r << d，Value 保持 d。
  2. factorize_projection：W_Q = A B（对投影矩阵做低秩分解），输出维度仍为 d。
  3. approximate_score：对 softmax(A) 矩阵做低秩近似（后处理，不省 QK 乘法）。

本模块给出可执行实现、误差指标与完整图 MAC 核算。
来源：LoRA (ICLR 2022)、VN-SST (arXiv:2608.25088) 为思想背景；本项目独立实现，
不宣称原创机制。

运行：python low_rank_attention.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class AttentionStats:
    score_macs: int
    total_macs: int
    params: int
    r: int


def dense_attention(X: torch.Tensor, Wq: torch.Tensor, Wk: torch.Tensor,
                    Wv: torch.Tensor, Wo: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, d = X.shape
    h = 4
    dh = d // h
    Q = (X @ Wq).view(B, T, h, dh).transpose(1, 2)
    K = (X @ Wk).view(B, T, h, dh).transpose(1, 2)
    V = (X @ Wv).view(B, T, h, dh).transpose(1, 2)
    S = (Q @ K.transpose(-1, -2)) / math.sqrt(dh)
    S = S.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), float("-inf"))
    A = torch.softmax(S, dim=-1)
    O = (A @ V).transpose(1, 2).reshape(B, T, d)
    return O @ Wo, A


class LowRankQKAttention(nn.Module):
    """机制 1：只降低 Q/K 的有效交互维度；V、FFN、输出投影不变。"""

    def __init__(self, d: int, r: int, h: int = 4):
        super().__init__()
        assert d % h == 0 and 1 <= r <= d
        self.d, self.r, self.h, self.dh = d, r, h, d // h
        self.Uq = nn.Parameter(torch.empty(d, r))
        self.Uk = nn.Parameter(torch.empty(d, r))
        self.Wv = nn.Parameter(torch.empty(d, d))
        self.Wo = nn.Parameter(torch.empty(d, d))
        self.scale = math.sqrt(self.dh)
        self.reset_parameters()

    def reset_parameters(self):
        for p in (self.Uq, self.Uk, self.Wv, self.Wo):
            nn.init.xavier_uniform_(p)

    def forward(self, X: torch.Tensor, return_attn: bool = False):
        B, T, d = X.shape
        Q = (X @ self.Uq).view(B, T, self.h, -1).transpose(1, 2)
        K = (X @ self.Uk).view(B, T, self.h, -1).transpose(1, 2)
        V = (X @ self.Wv).view(B, T, self.h, self.dh).transpose(1, 2)
        S = (Q @ K.transpose(-1, -2)) / self.scale
        S = S.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool, device=X.device), 1),
                          float("-inf"))
        A = torch.softmax(S, dim=-1)
        O = (A @ V).transpose(1, 2).reshape(B, T, d)
        y = O @ self.Wo
        return (y, A) if return_attn else y

    def macs(self, T: int) -> AttentionStats:
        B = 1
        score = B * self.h * T * T * self.r
        total = (B * T * self.d * 2 * self.r + score * 2 +
                 B * T * self.d * self.d * 2)
        return AttentionStats(score, total, sum(p.numel() for p in self.parameters()), self.r)


class FactorizedProjectionAttention(nn.Module):
    """机制 2：投影矩阵低秩分解 W=AB，但输出维度与 QK 交互维度仍为 d。"""

    def __init__(self, d: int, r: int, h: int = 4):
        super().__init__()
        self.d, self.r, self.h, self.dh = d, r, h, d // h
        self.Aq, self.Bq = nn.Parameter(torch.empty(d, r)), nn.Parameter(torch.empty(r, d))
        self.Ak, self.Bk = nn.Parameter(torch.empty(d, r)), nn.Parameter(torch.empty(r, d))
        self.Wv, self.Wo = nn.Parameter(torch.empty(d, d)), nn.Parameter(torch.empty(d, d))
        for p in self.parameters():
            nn.init.xavier_uniform_(p)

    def forward(self, X: torch.Tensor):
        B, T, d = X.shape
        Q = (X @ (self.Aq @ self.Bq)).view(B, T, self.h, self.dh).transpose(1, 2)
        K = (X @ (self.Ak @ self.Bk)).view(B, T, self.h, self.dh).transpose(1, 2)
        V = (X @ self.Wv).view(B, T, self.h, self.dh).transpose(1, 2)
        S = (Q @ K.transpose(-1, -2)) / math.sqrt(self.dh)
        S = S.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool, device=X.device), 1), float("-inf"))
        O = (torch.softmax(S, -1) @ V).transpose(1, 2).reshape(B, T, d)
        return O @ self.Wo


def approximate_attention_matrix(A: torch.Tensor, r: int) -> torch.Tensor:
    """机制 3：softmax 后矩阵 SVD 低秩近似；不减少 QK 乘法，仅后处理。"""
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    return U[:, :, :, :r] * S[:, :, None, :r] @ Vh[:, :, :r, :]


def error_metrics(ref: torch.Tensor, got: torch.Tensor) -> dict:
    num = (ref - got).pow(2).sum().sqrt().item()
    den = ref.pow(2).sum().sqrt().item()
    return {"relative_l2": round(num / max(den, 1e-12), 6),
            "max_abs_diff": round((ref - got).abs().max().item(), 6)}


def run_self_test():
    torch.manual_seed(20261124)
    B, T, d = 2, 32, 64
    X = torch.randn(B, T, d)
    Wq, Wk, Wv, Wo = [torch.randn(d, d) * 0.1 for _ in range(4)]
    ref_out, ref_A = dense_attention(X, Wq, Wk, Wv, Wo)
    out = {"status": "COMPLETE", "mechanisms": {}}
    for r in (4, 8, 16, 32, 64):
        m = LowRankQKAttention(d, r)
        y, A = m(X, return_attn=True)
        stats = m.macs(T)
        dense = B * 4 * T * T * (d // 4) * 2 + B * T * d * d * 2
        out["mechanisms"][f"reduce_qk_dim_r{r}"] = {
            "output_error": error_metrics(ref_out, y),
            "attention_error": error_metrics(ref_A, A),
            "score_macs": stats.score_macs, "total_macs": stats.total_macs,
            "dense_reference_macs": dense,
            "full_graph_reduction": round(1 - stats.total_macs / dense, 4),
            "params": stats.params,
        }
    fm = FactorizedProjectionAttention(d, 16)
    out["mechanisms"]["factorize_projection_r16"] = {
        "output_error": error_metrics(ref_out, fm(X)),
        "note": "输出维度仍为 d，QK 交互维度未降低；MAC 不因分解而等比例下降",
    }
    out["mechanisms"]["approximate_score_matrix_r16"] = {
        "output_note": "需先算完整 QK 再 SVD，不省 QK 成本；仅作数学对照",
        "attention_error": error_metrics(ref_A, approximate_attention_matrix(ref_A, 16)),
    }
    out["conclusion"] = ("reduce_qk_dim 是唯一同时降低 score MAC 的机制；"
                         "但 Value/FFN/Head 不变时全图收益远低于 score 层收益；"
                         "factorize_projection 与 approximate_score 不得冒充省 QK 计算。")
    return out


if __name__ == "__main__":
    import json
    from pathlib import Path
    res = run_self_test()
    p = Path(__file__).resolve().parents[1] / "M4_LOW_RANK_ATTENTION.json"
    p.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[lowrank] 写出 {p}")
