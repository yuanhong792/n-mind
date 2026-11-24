"""P5：Incremental Information Interaction（III）增量核算子实验室。

研究形式（已有线性注意力方向，非本项目原创，也**不宣称与 softmax attention 等价**）：
    S_t = S_{t-1} + phi(k_t) v_t^T
    z_t = z_{t-1} + phi(k_t)
    y_t = phi(q_t)^T S_t / (phi(q_t)^T z_t + eps)

本轮只做独立算子研究：核函数、特征维度、分母稳定性、长序列干扰、有效秩、
固定容量与准确率关系、不同长度成本。不默认加入主模型（M1-M3 固定状态记忆失败历史）。

来源：线性注意力（Katharopoulos et al. 2020）、DeltaNet、HOLA (arXiv:2607.02303) 为背景，
本项目独立实现，不复制代码。

运行：python incremental_kernel.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class KernelConfig:
    name: str
    feat_dim: int
    d: int


def phi(x: torch.Tensor, cfg: KernelConfig, proj: torch.Tensor | None = None) -> torch.Tensor:
    """核特征映射。x: [..., d] → [..., feat_dim]。"""
    if cfg.name == "elu1":
        base = torch.nn.functional.elu(x) + 1.0
    elif cfg.name == "relu":
        base = torch.nn.functional.relu(x)
    elif cfg.name == "identity":
        base = x
    else:
        raise ValueError(cfg.name)
    if base.shape[-1] == cfg.feat_dim:
        return base
    assert proj is not None, "需要随机投影到 feat_dim"
    return base @ proj


def incremental_state(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                      cfg: KernelConfig, eps: float = 1e-6, proj: torch.Tensor | None = None):
    """逐 token 递推，返回输出与累计状态。Q/K/V: [B,T,d]。"""
    B, T, d = Q.shape
    S = torch.zeros(B, cfg.feat_dim, d, device=Q.device)
    z = torch.zeros(B, cfg.feat_dim, device=Q.device)
    outs = []
    denom_min = float("inf")
    for t in range(T):
        pk = phi(K[:, t], cfg, proj)
        pv = phi(Q[:, t], cfg, proj)
        S = S + torch.einsum("bf,bd->bfd", pk, V[:, t])
        z = z + pk
        den = torch.einsum("bf,bf->b", pv, z) + eps
        denom_min = min(denom_min, float(den.min()))
        y = torch.einsum("bf,bfd->bd", pv, S) / den.unsqueeze(-1)
        outs.append(y)
    return torch.stack(outs, dim=1), S, denom_min


def softmax_attention(Q, K, V):
    d = Q.shape[-1]
    s = torch.einsum("btd,bsd->bts", Q, K) / math.sqrt(d)
    s = s.masked_fill(torch.triu(torch.ones(Q.shape[1], Q.shape[1], dtype=torch.bool), 1), float("-inf"))
    return torch.einsum("bts,bsd->btd", torch.softmax(s, -1), V)


def run_self_test():
    torch.manual_seed(20261124)
    B, T, d = 2, 64, 32
    Q, K, V = [torch.randn(B, T, d) * 0.4 for _ in range(3)]
    ref = softmax_attention(Q, K, V)
    out = {"status": "COMPLETE", "disclaimer":
           "本算子属已有线性注意力方向，不宣称与 softmax attention 等价；" 
           "以下误差是数值观察，不是等价性证明。", "configs": {}}
    for name, fd in (("elu1", 64), ("elu1", 128), ("relu", 64), ("identity", 32)):
        cfg = KernelConfig(name, fd, d)
        proj = torch.randn(d, fd) * 0.25 if fd != d else None
        y, S, den_min = incremental_state(Q, K, V, cfg, proj=proj)
        sv = torch.linalg.svdvals(S[0])
        eff_rank = float((sv > sv[0] * 1e-3).sum())
        rel = ((y - ref).norm() / ref.norm()).item()
        out["configs"][f"{name}_f{fd}"] = {
            "output_relative_l2_to_softmax": round(rel, 4),
            "min_denominator": round(den_min, 8),
            "denominator_stable": den_min > 0,
            "state_effective_rank": eff_rank,
            "state_dim": fd * d,
            "note": "误差大不等于机制无效，也不等于有效；需任务能力实验",
        }
    # 长序列信息干扰：同一查询在长度 16/64/256 下的输出漂移
    drift = {}
    base_q = torch.randn(1, d)
    for L in (16, 64, 256):
        K2 = torch.randn(1, L, d) * 0.4
        V2 = torch.randn(1, L, d) * 0.4
        cfg = KernelConfig("elu1", 64, d)
        proj = torch.randn(d, 64) * 0.25
        S = torch.zeros(1, 64, d)
        z = torch.zeros(1, 64)
        for t in range(L):
            pk = phi(K2[:, t], cfg, proj)
            S = S + torch.einsum("bf,bd->bfd", pk, V2[:, t])
            z = z + pk
        pv = phi(base_q, cfg, proj)
        drift[str(L)] = round(float(torch.einsum("bf,bfd->bd", pv, S).norm()), 4)
    out["long_sequence_interference"] = {
        "retrieval_norm_by_length": drift,
        "note": "固定容量状态随长度增长是否饱和/漂移的代理指标；非能力结论",
    }
    out["cost_scaling"] = {
        "state_bytes_fp32_d32_f64": 64 * 32 * 4,
        "per_token_macs": 2 * 64 * 32 + 2 * 64 * 32,
        "note": "状态固定，成本随长度线性；与 softmax 的 T^2 对比需实测墙钟",
    }
    out["decision"] = "NOT_RUN in main model（需授权后单独预登记）"
    return out


if __name__ == "__main__":
    import json
    from pathlib import Path
    res = run_self_test()
    p = Path(__file__).resolve().parents[1] / "M4_INCREMENTAL_KERNEL.json"
    p.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[incremental] 写出 {p}")
