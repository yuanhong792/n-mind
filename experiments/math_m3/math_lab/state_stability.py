"""M3-P2/P5：神经状态稳定性分析（谱半径、Jacobian、扰动增长）。"""
from __future__ import annotations

import torch


def linear_map_spectral_radius(W: torch.Tensor) -> float:
    """对角/线性状态转移的谱半径（最大特征值模）。cpu 上用特征值。"""
    eig = torch.linalg.eigvals(W.float())
    return float(eig.abs().max().item())


def jacobian_finite_diff(fn, x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """数值 Jacobian：fn(x)[i] 对 x[j] 的偏导。x:[d] → fn:[d'] 返回 [d',d]。"""
    d = x.numel()
    base = fn(x)
    J = torch.zeros(base.numel(), d, device=x.device, dtype=x.dtype)
    for j in range(d):
        xp = x.clone(); xp[j] += eps
        xm = x.clone(); xm[j] -= eps
        J[:, j] = (fn(xp) - fn(xm)) / (2 * eps)
    return J


def perturbation_growth(fn, x: torch.Tensor, steps: int = 20,
                        eps: float = 1e-3) -> dict:
    """跟踪扰动在迭代动力系统中的增长；>1 表示发散倾向。"""
    torch.manual_seed(0)
    x1 = x.clone()
    x2 = x + torch.randn_like(x) * eps
    ratios = []
    for _ in range(steps):
        x1, x2 = fn(x1), fn(x2)
        d = float((x1 - x2).norm())
        ratios.append(d)
    return {"initial_perturbation": eps,
            "final_perturbation": round(ratios[-1], 6),
            "growth_factor": round(ratios[-1] / max(1e-12, eps), 4),
            "trajectory": [round(r, 6) for r in ratios]}


def state_norm_trajectory(states: torch.Tensor) -> list[float]:
    """states:[T,d] → 每步范数，检测爆炸/塌缩。"""
    return [round(float(s.float().norm()), 5) for s in states]
