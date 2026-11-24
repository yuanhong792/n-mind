"""M3-P5：正交保护记忆算子与成本基准（不进主模型，先做数学验证/性能基准）。

受保护键矩阵 K（每行一个已存键）。投影：
    P = I - Kᵀ (K Kᵀ)† K          # K:[n,d] 行为键
    u = P k                        # 新键的正交分量
候选更新（保持受保护键的读取不变）：
    S_new = S + (v - S k) uᵀ / (uᵀ k)
分母 uᵀk → 0 时安全退化为 0（不写入，而非除零）。

验证：
- 新关联写入准确性：读取新键 k 应得到 v
- 旧关联保持：读取 K 中受保护键应不变
- 相似键数值稳定：k≈受保护键时分母小，检查有限值
- 键空间饱和：n→d 时投影接近零
- 解除保护：可从保护集移除键
- 投影成本：实测 O(n²d + nd²) vs 收益
"""
from __future__ import annotations

import time

import torch


def protect_projection(K: torch.Tensor, ridge: float = 1e-6) -> torch.Tensor:
    """K:[n,d]（行为键）→ P = I - Kᵀ(KKᵀ+ridge I)†K，形状 [d,d]。"""
    d = K.shape[-1]
    G = K @ K.transpose(-1, -2)                       # [n,n]
    G_inv = torch.linalg.pinv(G + ridge * torch.eye(G.shape[-1], device=G.device, dtype=G.dtype))
    return torch.eye(d, device=K.device, dtype=K.dtype) - K.transpose(-1, -2) @ G_inv @ K


def orthogonal_update(S: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                      K: torch.Tensor, ridge: float = 1e-6, eps: float = 1e-6,
                      min_denom: float = 1e-4) -> tuple[torch.Tensor, dict]:
    """S:[dv,dk]（key 维 dk）；k:[dk]；K:[n,dk] 受保护键。"""
    P = protect_projection(K, ridge)
    u = P @ k                                         # [dk]
    denom = float((u @ k).item())
    stats = {"denominator": denom, "norm_u": round(float(u.norm().item()), 6),
             "projection_cost_macs": K.shape[0] ** 2 * K.shape[1] + K.shape[0] * K.shape[1] ** 2}
    if abs(denom) < min_denom:
        stats["safe_degenerate"] = True
        return S, stats                                # 分母过小 → 不写入
    stats["safe_degenerate"] = False
    pred = S @ k
    S_new = S + torch.outer(v - pred, u) / denom
    return S_new, stats


def benchmark_projection(d: int, n_list=(2, 8, 16, 32, 64), repeats=20) -> dict:
    """实测投影耗时（CPU），对比理论 MAC。"""
    torch.manual_seed(0)
    results = {}
    for n in n_list:
        if n > d:
            continue
        K = torch.linalg.qr(torch.randn(max(n, d), d))[0][:n]
        # warmup
        for _ in range(3):
            protect_projection(K)
        t0 = time.perf_counter()
        for _ in range(repeats):
            protect_projection(K)
        ms = (time.perf_counter() - t0) / repeats * 1000
        results[str(n)] = {"projection_ms": round(ms, 4),
                           "theoretical_macs": n * n * d + n * d * d}
    return results
