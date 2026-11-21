"""ActionHead：固定动作空间的决策输出 + 运行时给定候选集的 typed 打分。

两种用法：
- `decide()`：在 n_actions 个内置动作上给分布，用于网格环境的动作选择。
- `score_candidates()`：候选项在运行时提供（每条候选编码成特征），共享一次上下文计算、
  对每条候选算一个分数再 softmax。这个接口形态参考了公开发布的 typed-decision 模型
  （vllm-sr/Decision-1.0 系列与 Cloudflare Clef）的请求/响应约定，
  不是对它们权重的复用；本模块的权重是自己训练的。

置信度是 max softmax 概率，未经校准（见 docs/LIMITATIONS.md），只能用于相对比较。
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class ActionHead(nn.Module):
    def __init__(self, z_dim: int, d_w: int, mem_dim: int, n_actions: int, hidden: int = 32) -> None:
        super().__init__()
        if n_actions < 2:
            raise ValueError("n_actions 至少为 2")
        self.n_actions = n_actions
        self.d_ctx = z_dim + d_w + mem_dim
        self.ctx_mlp = nn.Sequential(
            nn.Linear(self.d_ctx, hidden), nn.GELU(), nn.Linear(hidden, z_dim)
        )
        self.act_out = nn.Linear(z_dim, n_actions)
        # typed 打分用的交互参数：双线性 + 低秩 MLP 相加
        self.cand_norm = nn.LayerNorm(z_dim)
        self.query_norm = nn.LayerNorm(z_dim)
        self.key = nn.Linear(z_dim, hidden, bias=False)
        self.query = nn.Linear(z_dim, hidden, bias=False)
        self.cand_mlp = nn.Linear(z_dim, hidden)
        self.query_mlp = nn.Linear(z_dim, hidden, bias=False)
        self.scalar = nn.Linear(hidden, 1, bias=False)
        self.hidden = hidden

    # ------------------------------------------------------------------ 内置动作

    def context(self, z: torch.Tensor, w: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        if mem.shape[-1] != self.ctx_mlp[0].in_features - z.shape[-1] - w.shape[-1]:
            raise ValueError(
                f"mem 维度与 ActionHead 不匹配：期望 {self.ctx_mlp[0].in_features - z.shape[-1] - w.shape[-1]}，"
                f"实际 {mem.shape[-1]}"
            )
        return self.ctx_mlp(torch.cat([z, w, mem], dim=-1))

    def forward(self, z: torch.Tensor, w: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        return self.act_out(self.context(z, w, mem))

    def decide(self, z: torch.Tensor, w: torch.Tensor, mem: torch.Tensor) -> dict[str, Any]:
        logits = self.forward(z, w, mem)
        probs = torch.softmax(logits, dim=-1)
        conf, idx = probs.max(dim=-1)
        top2 = torch.topk(probs, k=min(2, self.n_actions), dim=-1).values
        margin = (top2[:, 0] - (top2[:, 1] if top2.shape[1] > 1 else torch.zeros_like(top2[:, 0])))
        return {
            "logits": logits,
            "probabilities": probs,
            # action_index 恒为"每个样本一个整数"的列表（形状与 batch 对齐，调用方别猜）；
            # action 是单人批次的便捷别名，批次 >1 时给列表，避免 int(list) 那种隐坑。
            "action_index": [int(v) for v in idx.tolist()],
            "action": int(idx[0].item()) if probs.shape[0] == 1 else [int(v) for v in idx.tolist()],
            "confidence": float(conf.mean().item()),
            "margin": float(margin.mean().item()),
        }

    # ------------------------------------------------------------------ 运行时候选集

    def score_candidates(
        self, ctx: torch.Tensor, candidates: torch.Tensor, candidate_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """ctx [B, d]，candidates [B, C, d]（或 [C, d] 自动广播）→ 分数 [B, C]。"""
        if candidates.dim() == 2:
            candidates = candidates.unsqueeze(0).expand(ctx.shape[0], -1, -1)
        if candidates.shape[-1] != ctx.shape[-1]:
            raise ValueError("候选特征维度必须与上下文一致")
        c = self.cand_norm(candidates)
        q = self.query_norm(ctx)
        bilinear = (self.key(c) * self.query(q)[:, None, :]).sum(-1) / math.sqrt(self.hidden)
        interaction = self.scalar(F.gelu(self.cand_mlp(c) + self.query_mlp(q)[:, None, :])).squeeze(-1)
        scores = bilinear + interaction
        if candidate_mask is not None:
            scores = scores.masked_fill(~candidate_mask.bool(), -torch.inf)
        return scores

    def choose(
        self,
        ctx: torch.Tensor,
        candidate_feats: torch.Tensor,
        keys: list[str],
        instructions: str = "",
    ) -> dict[str, Any]:
        """返回 typed 决策：choice / probabilities / confidence。keys 由调用方提供。"""
        if len(keys) < 2:
            raise ValueError("候选集至少 2 项")
        scores = self.score_candidates(ctx, candidate_feats)
        probs = torch.softmax(scores, dim=-1)
        conf, idx = probs.max(dim=-1)
        p0 = probs[0]
        return {
            "type": "choice",
            "choice": keys[int(idx[0].item())],
            "confidence": float(conf.mean().item()),
            "probabilities": {k: round(float(v), 6) for k, v in zip(keys, p0.tolist())},
            "instructions": instructions,
        }

    def judge(self, ctx: torch.Tensor, pos_feats: torch.Tensor, neg_feats: torch.Tensor) -> dict[str, Any]:
        """noul 型：给定"为真/为假"两条候选，返回为真的概率。"""
        feats = torch.stack([neg_feats, pos_feats], dim=0).unsqueeze(0)
        scores = self.score_candidates(ctx, feats)
        probs = torch.softmax(scores, dim=-1)
        return {
            "type": "noul",
            "noul": round(float(probs[0, 1].item()), 6),
            "probabilities": {"false": round(float(probs[0, 0].item()), 6),
                              "true": round(float(probs[0, 1].item()), 6)},
        }

    def rate(self, ctx: torch.Tensor, level_feats: torch.Tensor, levels: list[str]) -> dict[str, Any]:
        """score 型：等级分布 + 期望分数（等级序号的期望）。"""
        scores = self.score_candidates(ctx, level_feats.unsqueeze(0))
        probs = torch.softmax(scores, dim=-1)[0]
        expected = float(sum(i * float(p) for i, p in enumerate(probs)))
        return {
            "type": "score",
            "score": round(expected, 4),
            "confidence": round(float(probs.max().item()), 6),
            "probabilities": {k: round(float(v), 6) for k, v in zip(levels, probs.tolist())},
        }
