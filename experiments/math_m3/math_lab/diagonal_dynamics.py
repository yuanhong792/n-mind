"""M3-P3/P2：对角状态动力学与状态充分性可观测代理。

对角变换 h_new = a ⊙ h + b 是 O(d)，用于保留长期信息（Mamba/SSM 的对角特例）。
提供：
- diagonal_step：对角状态更新
- effective_rank：状态奇异值有效秩（参与累计能量 95% 的奇异值数）
- singular_spectrum：奇异值分布
- compress_retain：截断到 k 维后的能力保持率代理（用输出重建误差衡量）
- perturbation_sensitivity：状态分量扰动的敏感度
- component_ablation：关闭某些分量后的能力变化（由调用方提供评测函数）
全部为可观测代理指标，不冒充互信息。
"""
from __future__ import annotations

import torch
from torch import nn


def diagonal_step(h: torch.Tensor, a: torch.Tensor, b: torch.Tensor | None = None) -> torch.Tensor:
    """h_new = a ⊙ h (+ b)。a 可为 [d] 或与 h 同形。"""
    out = a * h
    if b is not None:
        out = out + b
    return out


def singular_spectrum(states: torch.Tensor) -> torch.Tensor:
    """states:[N,d] → 奇异值降序。"""
    return torch.linalg.svdvals(states.float())


def effective_rank(states: torch.Tensor, energy: float = 0.95) -> int:
    """累计平方奇异值达到 energy 比例所需的最小奇异值数。"""
    s = singular_spectrum(states)
    var = s.pow(2)
    cum = var.cumsum(0) / var.sum().clamp_min(1e-12)
    return int((cum < energy).sum().item() + 1)


def compress_retain(states: torch.Tensor, k: int) -> dict:
    """用前 k 个主成分重建状态，返回能量保持率与最大重建误差。"""
    U, s, Vh = torch.linalg.svd(states.float(), full_matrices=False)
    k = min(k, s.numel())
    recon = (U[:, :k] * s[:k]) @ Vh[:k]
    err = (states.float() - recon).abs()
    energy = float(s[:k].pow(2).sum() / s.pow(2).sum().clamp_min(1e-12))
    return {"k": k, "energy_retained": round(energy, 5),
            "max_recon_error": round(float(err.max()), 6)}


def perturbation_sensitivity(states: torch.Tensor, eps: float = 1e-3,
                             eval_fn=None) -> dict:
    """状态扰动敏感度：若无 eval_fn，用状态相对变化衡量；有则测能力变化。"""
    torch.manual_seed(0)
    noise = torch.randn_like(states.float()) * eps
    rel = float((noise.norm() / states.float().norm().clamp_min(1e-12)))
    out = {"eps": eps, "relative_perturbation": round(rel, 6)}
    if eval_fn is not None:
        base = eval_fn(states)
        pert = eval_fn(states.float() + noise)
        out["capability_delta"] = round(float(abs(base - pert)), 6)
    return out


class DiagonalDynamics(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.a = nn.Parameter(torch.ones(d) * 0.95)
        self.b = nn.Parameter(torch.zeros(d))

    def forward(self, h):
        return diagonal_step(h, torch.sigmoid(self.a) * 2.0 - 1.0, self.b)

    def macs(self, batch=1):
        return {"diag_mul": batch * self.a.numel(), "total": batch * self.a.numel()}
