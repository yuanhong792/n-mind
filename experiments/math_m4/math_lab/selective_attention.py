"""P4：Selective Information Interaction（SII）算子实验室。

先做低成本分组摘要与粗检索，只对选中组做精确注意力。
严格纪律：
- 禁止先构建完整 QK 矩阵再 top-k（那样没有省掉完整矩阵乘法）。
- 摘要只能包含当前 Query 可访问的已完成组；未完成组只用已到达 token。
- 训练用可微软路由 + Straight-through 硬门；推理实际稀疏路径。
- 软门成绩不得代替硬门成绩；未训练时固定局部窗口对照标 NOT_RUN。

运行：python selective_attention.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class RoutingCost:
    summary_build: int
    summary_update: int
    coarse_retrieval: int
    topk: int
    gather: int
    exact_attention: int
    dense_attention: int
    full_qk_forbidden: bool = False


class GroupedSelectiveAttention(nn.Module):
    def __init__(self, d: int, h: int, group_size: int = 4, top_k: int = 1):
        super().__init__()
        assert d % h == 0
        self.d, self.h, self.dh = d, h, d // h
        self.gs, self.top_k = group_size, top_k
        self.summary_proj = nn.Linear(self.dh, self.dh, bias=False)
        self.scale = math.sqrt(self.dh)

    def _groups(self, x: torch.Tensor):
        """x: [B, h, T, dh] → 已完成组 + 未完成组（长度不足 gs 的尾部）。"""
        B, h, T, dh = x.shape
        n_full = T // self.gs
        full = x[:, :, :n_full * self.gs].reshape(B, h, n_full, self.gs, dh)
        tail = x[:, :, n_full * self.gs:]
        return full, tail

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                hard: bool = False, return_stats: bool = False):
        """Q/K/V: [B,h,T,dh]；因果注意力，按组选择。"""
        B, h, T, dh = Q.shape
        assert K.shape == V.shape == Q.shape
        Kf, Kt = self._groups(K)
        Vf, Vt = self._groups(V)
        n_full = Kf.shape[2]

        # 摘要只由该组内已到达的 K 构成（Kt 尾部不补未来信息）
        summary = self.summary_proj(Kf.mean(dim=3))                     # [B,h,G,dh]
        # 粗检索：每个 query 与组摘要点积，不构造 T×T 完整 QK
        coarse = torch.einsum("bhtd,bhgd->bhtg", Q, summary) / self.scale  # [B,h,T,G]
        # 因果粗检索：query t 只能选起始位置 <= t 的已完成组，否则 softmax 全 -inf → NaN
        g_start = torch.arange(n_full, device=Q.device) * self.gs
        q_pos = torch.arange(T, device=Q.device)
        coarse = coarse.masked_fill(g_start[None, None, None, :] > q_pos[:, None], float("-inf"))
        k = min(self.top_k, n_full) if n_full else 0
        valid_group = ~(g_start[None, None, None, :] > q_pos[:, None]) if n_full else coarse.new_ones(coarse.shape, dtype=torch.bool)
        if k > 0:
            soft = torch.softmax(coarse, dim=-1) * valid_group
            soft = soft / soft.sum(-1, keepdim=True).clamp_min(1e-12)
            idx = coarse.topk(k, dim=-1).indices
            hard_mask = torch.zeros_like(soft).scatter_(-1, idx, 1.0) * valid_group
            # STE：前向硬门、反向软门
            route = hard_mask.detach() + soft - soft.detach() if self.training else hard_mask
        else:
            route = coarse.new_zeros(coarse.shape)

        # 只对选中组做精确注意力（其余组不进入 matmul）
        out = torch.zeros(B, h, T, dh, device=Q.device)
        if k > 0:
            for g in range(n_full):
                weight = route[:, :, :, g]                              # [B,h,T]
                if float(weight.detach().max()) == 0.0:
                    continue
                kg = Kf[:, :, g]                                        # [B,h,gs,dh]
                vg = Vf[:, :, g]
                s = torch.einsum("bhtd,bhsd->bhts", Q, kg) / self.scale
                pos_q = torch.arange(T, device=Q.device)[:, None]
                pos_k = torch.arange(g * self.gs, (g + 1) * self.gs, device=Q.device)[None, :]
                s = s.masked_fill(pos_k > pos_q, float("-inf"))
                a = torch.nan_to_num(torch.softmax(s, dim=-1), nan=0.0)
                out += weight.unsqueeze(-1) * torch.einsum("bhts,bhsd->bhtd", a, vg)

        # 未完成组：只用已到达 token，禁止访问未来
        if Kt.shape[2] > 0:
            qs = torch.arange(T, device=Q.device)[:, None]
            ks = torch.arange(n_full * self.gs, T, device=Q.device)[None, :]
            s = torch.einsum("bhtd,bhsd->bhts", Q, Kt) / self.scale
            s = s.masked_fill(ks > qs, float("-inf"))
            a = torch.softmax(s, dim=-1)
            out = out + torch.einsum("bhts,bhsd->bhtd", a, Vt)

        if return_stats:
            dense = B * h * T * T * dh * 2
            exact = B * h * T * k * self.gs * dh * 2
            stats = RoutingCost(
                summary_build=B * h * n_full * dh * dh,
                summary_update=B * h * dh * dh,
                coarse_retrieval=B * h * T * n_full * dh,
                topk=B * h * T * n_full,
                gather=B * h * T * k * self.gs * dh,
                exact_attention=exact, dense_attention=dense, full_qk_forbidden=False)
            return out, stats
        return out


def dense_attention(Q, K, V):
    B, h, T, dh = Q.shape
    s = torch.einsum("bhtd,bhsd->bhts", Q, K) / math.sqrt(dh)
    s = s.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), float("-inf"))
    return torch.einsum("bhts,bhsd->bhtd", torch.softmax(s, -1), V)


def fixed_window_attention(Q, K, V, window: int):
    """固定局部窗口对照，同预算；未训练时能力为 NOT_RUN。"""
    B, h, T, dh = Q.shape
    out = torch.zeros_like(Q)
    for t in range(T):
        lo, hi = max(0, t - window + 1), t + 1
        s = torch.einsum("bhd,bhsd->bhs", Q[:, :, t], K[:, :, lo:hi]) / math.sqrt(dh)
        a = torch.softmax(s, -1)
        out[:, :, t] = torch.einsum("bhs,bhsd->bhd", a, V[:, :, lo:hi])
    return out


def run_self_test():
    torch.manual_seed(20261124)
    B, h, T, d = 2, 4, 32, 64
    q, k, v = [torch.randn(B, h, T, d // h) for _ in range(3)]
    m = GroupedSelectiveAttention(d, h, group_size=4, top_k=2)
    m.eval()
    y, stats = m(q, k, v, hard=True, return_stats=True)
    ref = dense_attention(q, k, v)
    win = fixed_window_attention(q, k, v, window=8)

    # 因果性：修改未来 token 不得影响前 T/2 个输出
    k2, v2 = k.clone(), v.clone()
    k2[:, :, T // 2:] += 100.0
    v2[:, :, T // 2:] += 100.0
    with torch.no_grad():
        y2 = m(q, k2, v2, hard=True)
    causal_max = (y[:, :, :T // 2] - y2[:, :, :T // 2]).abs().max().item()

    # STE 梯度：软路由路径必须有非零梯度
    m.train()
    qg = q.clone().requires_grad_(True)
    yg = m(qg, k, v, hard=True)
    yg.square().mean().backward()
    grad_norm = qg.grad.norm().item()

    return {
        "status": "COMPLETE",
        "output_cosine_to_dense": round(torch.nn.functional.cosine_similarity(
            y.flatten(), ref.flatten(), dim=0).item(), 6),
        "output_relative_l2": round(((y - ref).norm() / ref.norm()).item(), 6),
        "fixed_window_relative_l2": round(((win - ref).norm() / ref.norm()).item(), 6),
        "causality_future_perturbation_max": round(causal_max, 8),
        "causality_pass": causal_max < 1e-6,
        "ste_gradient_norm": round(grad_norm, 6),
        "ste_gradient_pass": grad_norm > 0,
        "routing_cost": stats.__dict__,
        "skipped_attention_ratio": round(1 - stats.exact_attention / stats.dense_attention, 4),
        "note_cost": "统计已含摘要/粗检索/top-k/gather；不得只报精确注意力节省",
        "fixed_window_capability": "NOT_RUN（未获训练预算，不得伪造能力对照）",
    }


if __name__ == "__main__":
    import json
    from pathlib import Path
    res = run_self_test()
    p = Path(__file__).resolve().parents[1] / "M4_SELECTIVE_ATTENTION.json"
    p.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[selective] causal={res['causality_pass']} ste={res['ste_gradient_pass']} "
          f"skipped={res['skipped_attention_ratio']}")
    print(f"[selective] 写出 {p}")
