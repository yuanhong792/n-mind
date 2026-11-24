"""P6/P8 辅助：信息交互的可观测代理分析。

不冒充精确互信息；只提供有效秩、奇异值谱、压缩保持率、查询区分度、
Attention 分布熵等可测量代理，用于公式初筛与状态容量分析。

运行：python information_analysis.py
"""

from __future__ import annotations

import math

import torch


def effective_rank(M: torch.Tensor, threshold: float = 1e-3) -> dict:
    s = torch.linalg.svdvals(M)
    if s.numel() == 0 or s[0] == 0:
        return {"effective_rank": 0, "top_singular": 0.0}
    return {
        "effective_rank": int((s > s[0] * threshold).sum()),
        "matrix_rank": int(torch.linalg.matrix_rank(M)),
        "top_singular": round(float(s[0]), 6),
        "tail_energy_ratio": round(float(s[-1] / s[0]), 8),
        "normalized_spectrum": [round(float(x / s[0]), 5) for x in s[:10]],
    }


def query_discriminability(A: torch.Tensor) -> dict:
    """A: [B,h,T,T] attention 矩阵，衡量不同 query 的注意力分布差异。"""
    a = A[:, :, -1]  # 最后一个 query 的分布
    flat = a.reshape(a.shape[0], -1)
    sim = torch.nn.functional.cosine_similarity(flat.unsqueeze(1), flat.unsqueeze(0), dim=-1)
    n = sim.shape[0]
    off = sim[~torch.eye(n, dtype=torch.bool)]
    return {"mean_pairwise_cosine": round(float(off.mean()), 6),
            "distinct_queries": int((off < 0.99).sum().item()),
            "note": "高相似度意味着 query 无法区分不同历史位置"}


def attention_entropy(A: torch.Tensor) -> dict:
    p = A.clamp_min(1e-12)
    h = -(p * p.log()).sum(dim=-1)
    return {"mean_entropy": round(float(h.mean()), 6),
            "max_entropy": round(float(h.max()), 6),
            "note": "熵接近 log(T) 表示近乎均匀；不能单独作为能力证据"}


def compression_retention(X: torch.Tensor, k: int) -> dict:
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    P = Vh[:k].T
    Z = X @ P
    Xh = Z @ P.T
    return {"k": k, "relative_frobenius_error":
            round(float(((X - Xh) ** 2).sum().sqrt() / (X ** 2).sum().sqrt()), 6),
            "energy_retained": round(float((S[:k] ** 2).sum() / (S ** 2).sum()), 6)}


def run_self_test():
    torch.manual_seed(20261124)
    B, h, T, d = 4, 4, 32, 64
    X = torch.randn(B * T, d)
    Q = torch.randn(B, h, T, d // h)
    K = torch.randn(B, h, T, d // h)
    s = torch.einsum("bhtd,bhsd->bhts", Q, K) / math.sqrt(d // h)
    A = torch.softmax(s, -1)
    return {
        "status": "COMPLETE",
        "disclaimer": "以下均为可观测代理，不是互信息或能力的等价测量",
        "state_effective_rank": effective_rank(X),
        "attention_query_discriminability": query_discriminability(A),
        "attention_entropy": attention_entropy(A),
        "compression_retention": {f"k{k}": compression_retention(X, k) for k in (4, 8, 16, 32)},
    }


if __name__ == "__main__":
    import json
    from pathlib import Path
    res = run_self_test()
    p = Path(__file__).resolve().parents[1] / "M4_INFORMATION_ANALYSIS.json"
    p.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[info] effective_rank={res['state_effective_rank']['effective_rank']} 写出 {p}")
