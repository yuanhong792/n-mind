"""M3-P2：最小充分神经状态的可观测代理指标（不冒充互信息）。

严禁使用未经验证的代理指标冒充真实互信息 I(S;X)/I(S;Y)。
本模块只提供：
- 状态维度、有效秩、奇异值分布
- 压缩后能力保持率（调用方提供 eval_fn）
- 隐藏状态可解码性（线性探针 R²，可观测）
- 状态扰动敏感度
- 分量消融的能力变化
所有指标明确标为 PROXY，报告须写「代理指标，非互信息估计」。
"""
from __future__ import annotations

import torch
from torch import nn

from diagonal_dynamics import effective_rank, singular_spectrum


def state_proxies(states: torch.Tensor) -> dict:
    """states:[N,d] → 维度/有效秩/奇异值统计。全部为 PROXY。"""
    s = singular_spectrum(states)
    d = states.shape[-1]
    return {
        "proxy_type": "OBSERVABLE_NOT_MUTUAL_INFORMATION",
        "state_dimension": d,
        "effective_rank_95": effective_rank(states, 0.95),
        "top_singular": [round(float(x), 5) for x in s[:8]],
        "singular_ratio_max_min": round(float(s[0] / s[-1].clamp_min(1e-12)), 4),
    }


def linear_decodability(states: torch.Tensor, targets: torch.Tensor,
                        ridge: float = 1e-3) -> dict:
    """线性探针 R²（最小二乘 + ridge），衡量状态对目标的可解码性（PROXY）。"""
    X = states.float()
    Y = targets.float()
    G = X.transpose(-1, -2) @ X + ridge * torch.eye(X.shape[-1], device=X.device)
    W = torch.linalg.solve(G, X.transpose(-1, -2) @ Y)
    pred = X @ W
    ss_res = float((Y - pred).pow(2).sum())
    ss_tot = float((Y - Y.mean(0)).pow(2).sum().clamp_min(1e-12))
    return {"proxy_type": "LINEAR_DECODABILITY_R2",
            "r2": round(1.0 - ss_res / ss_tot, 5)}


def compression_retention(states: torch.Tensor, targets: torch.Tensor,
                          k_list=(16, 32, 64, 96, 128), ridge: float = 1e-3) -> dict:
    """压缩状态到 k 维后的能力保持率（PROXY）：用前 k 主成分状态做线性解码 R²。"""
    X = states.float()
    U, s, Vh = torch.linalg.svd(X, full_matrices=False)
    out = {}
    for k in k_list:
        kk = min(k, s.numel())
        Xk = (U[:, :kk] * s[:kk])
        G = Xk.transpose(-1, -2) @ Xk + ridge * torch.eye(kk, device=X.device)
        W = torch.linalg.solve(G, Xk.transpose(-1, -2) @ targets.float())
        pred = Xk @ W
        Y = targets.float()
        r2 = 1.0 - float((Y - pred).pow(2).sum()) / float((Y - Y.mean(0)).pow(2).sum().clamp_min(1e-12))
        out[str(k)] = round(r2, 5)
    return {"proxy_type": "COMPRESSED_DECODABILITY_R2", "r2_by_dim": out}
