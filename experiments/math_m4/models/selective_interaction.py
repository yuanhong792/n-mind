"""M4-B：Selective Information Interaction 模型。

基于 R2，只替换信息交互范围与学习选择机制；不同时加入低秩投影、
正交压缩或新型 FFN。训练用软路由 + STE 硬门，部署用真实硬门。
固定局部窗口对照在 math_lab/selective_attention.py 中，未训练标 NOT_RUN。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from base import M2Base as M4Base, PAD  # noqa: E402
from transformer_reference import rope_rotate  # noqa: E402


class GroupedSelectiveAttention(nn.Module):
    def __init__(self, d_model, nhead, group_size=4, top_k=2):
        super().__init__()
        assert d_model % nhead == 0
        self.d, self.h, self.dh = d_model, nhead, d_model // nhead
        self.gs, self.top_k = group_size, top_k
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.summary_proj = nn.Linear(self.dh, self.dh, bias=False)
        self.scale = math.sqrt(self.dh)

    def forward(self, x, cache_k=None, cache_v=None, pos_start=0, hard_gate=False,
                collect_stats=False):
        B, T, d = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.h, self.dh)
        Q, K, V = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        pos = torch.arange(pos_start, pos_start + T, device=x.device).float()
        inv = 1.0 / (10000 ** (torch.arange(0, self.dh, 2, device=x.device).float() / self.dh))
        ang = pos[:, None] * inv[None, :]
        cos = torch.cos(ang)[None, :, None, :].expand(B, T, self.h, -1)
        sin = torch.sin(ang)[None, :, None, :].expand(B, T, self.h, -1)
        Q, K = rope_rotate(Q, cos, sin), rope_rotate(K, cos, sin)
        if cache_k is not None:
            K = torch.cat([cache_k, K], dim=1)
            V = torch.cat([cache_v, V], dim=1)
        L = K.shape[1]   # K 为 [B,T,h,dh]，序列长度在 dim1（不是 dim2=头数）
        n_full = L // self.gs
        n_new = T
        out = K.new_zeros(B, T, self.h, self.dh)
        stats = {"path_counts": {0: 0, 1: 0, 2: 0}, "executed_macs": 0,
                 "controller_macs": B * T * d * n_full * 2}
        if n_full == 0:
            out = self._exact(Q, K, V, torch.arange(L, device=x.device)[None, :])
            out = out.transpose(1, 2).reshape(B, T, d)
            return self.proj(out), K, V, stats

        Kt = K.transpose(1, 2)   # [B,h,L,dh]，先换头维再分组，避免 reshape 打乱头顺序
        Vt = V.transpose(1, 2)
        Kf = Kt[:, :, :n_full * self.gs].reshape(B, self.h, n_full, self.gs, self.dh)
        Vf = Vt[:, :, :n_full * self.gs].reshape(B, self.h, n_full, self.gs, self.dh)
        summary = self.summary_proj(Kf.mean(dim=3))
        coarse = torch.einsum("bthd,bhgd->bhtg", Q, summary) / self.scale
        g_start = torch.arange(n_full, device=x.device) * self.gs
        qpos = torch.arange(pos_start, pos_start + T, device=x.device)
        valid = ~(g_start[None, None, :] > qpos[:, None])
        coarse = coarse.masked_fill(~valid, float("-inf"))
        soft = torch.softmax(coarse, -1) * valid
        soft = soft / soft.sum(-1, keepdim=True).clamp_min(1e-12)
        k = min(self.top_k, n_full)
        idx = coarse.topk(k, -1).indices
        hard = torch.zeros_like(soft).scatter_(-1, idx, 1.0) * valid
        route = hard.detach() + soft - soft.detach() if (self.training and not hard_gate) else hard

        for g in range(n_full):
            w = route[:, :, :, g].unsqueeze(-1)   # [B,h,T,1] 才能与 attention [B,h,T,gs] 相乘
            if float(w.detach().max()) == 0.0:
                continue
            s = torch.einsum("bthd,bhsd->bhts", Q, Kf[:, :, g]) / self.scale
            pk = torch.arange(g * self.gs, (g + 1) * self.gs, device=x.device)[None, :]
            s = s.masked_fill(pk > qpos[:, None], float("-inf"))
            a = torch.nan_to_num(torch.softmax(s, -1), nan=0.0)
            out = out + torch.einsum("bhts,bhsd->bthd", w * a, Vf[:, :, g])
            stats["path_counts"][min(g, 2)] += int((w > 0.5).sum())
            stats["executed_macs"] += B * T * self.gs * self.dh * 2 * self.h
        tail = K[:, n_full * self.gs:]
        if tail.shape[1] > 0:
            s = torch.einsum("bthd,bshd->bhts", Q, tail) / self.scale
            ks = torch.arange(n_full * self.gs, L, device=x.device)[None, :]
            s = s.masked_fill(ks > qpos[:, None], float("-inf"))
            a = torch.nan_to_num(torch.softmax(s, -1), nan=0.0)
            out = out + torch.einsum("bhts,bshd->bthd", a, V[:, n_full * self.gs:])
        stats["skipped_ratio"] = round(1 - stats["executed_macs"] /
                                       max(1, B * T * L * self.dh * 2 * self.h), 4)
        return self.proj(out.transpose(1, 2).reshape(B, T, d)), K, V, stats

    def _exact(self, Q, K, V, _):
        return torch.einsum("bhts,bshd->bhtd",
                            torch.softmax(torch.einsum("bthd,bshd->bhts", Q, K) / self.scale, -1), V)


class SelBlock(nn.Module):
    def __init__(self, d_model, nhead, group_size=4, top_k=2):
        super().__init__()
        self.attn = GroupedSelectiveAttention(d_model, nhead, group_size, top_k)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(),
                                nn.Linear(d_model * 2, d_model))
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, ck=None, cv=None, pos_start=0, hard_gate=False):
        a, k, v, stats = self.attn(self.ln1(x), ck, cv, pos_start, hard_gate)
        x = x + a
        x = x + self.ff(self.ln2(x))
        return x, k, v, stats


class SelectiveInteractionM4(M4Base):
    def __init__(self, vocab_size, d_model=64, nhead=4, layers=2, group_size=4, top_k=2):
        super().__init__()
        self.cfg = {"d_model": d_model, "nhead": nhead, "layers": layers,
                    "group_size": group_size, "top_k": top_k}
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.blocks = nn.ModuleList([SelBlock(d_model, nhead, group_size, top_k)
                                     for _ in range(layers)])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, tokens, state=None, valid_mask=None, hard_gate=False, collect_stats=False):
        B, T = tokens.shape
        pos_start = state["pos"] if state else 0
        x = self.emb(tokens)
        nck, ncv, stats_list = [], [], []
        for i, blk in enumerate(self.blocks):
            ck = state["ck"][i] if state and state["ck"] is not None else None
            cv = state["cv"][i] if state and state["cv"] is not None else None
            x, k, v, s = blk(x, ck, cv, pos_start, hard_gate)
            nck.append(k); ncv.append(v); stats_list.append(s)
        logits = self.head(self.ln(x))
        stats = {"blocks": stats_list,
                 "total_executed_macs": sum(s["executed_macs"] for s in stats_list),
                 "total_controller_macs": sum(s["controller_macs"] for s in stats_list)}
        return logits, {"ck": nck, "cv": ncv, "pos": pos_start + T, "stats": stats}

    def initial_state(self, batch, device):
        return {"ck": None, "cv": None, "pos": 0}
