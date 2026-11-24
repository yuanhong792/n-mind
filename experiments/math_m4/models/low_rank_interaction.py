"""M4-A：Low-Rank Information Interaction 模型。

与 R2 完全相同的层数/Embedding/FFN/Value 维度/位置编码/输出 Head，
唯一变量是 Q/K 交互维度 r（Value 表达维度保持 d）。
不加入低秩投影分解或正交压缩（那些只在 math_lab 做算子验证）。

来源：LoRA(2022)/VN-SST(2026) 为思想背景，本项目独立实现，不宣称原创。
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


class LowRankQKAttention(nn.Module):
    def __init__(self, d_model, nhead, r):
        super().__init__()
        assert d_model % nhead == 0
        self.d, self.h, self.dh, self.r = d_model, nhead, d_model // nhead, r
        # Value 表达维度保持 d；Q/K 降到 r（按头分配 r//h）
        self.rh = max(2, r // nhead)
        assert self.rh % 2 == 0, "RoPE 要求每头 Q/K 维度为偶数（r 选 8/16/32/64）"
        self.Uq = nn.Parameter(torch.empty(d_model, nhead * self.rh))
        self.Uk = nn.Parameter(torch.empty(d_model, nhead * self.rh))
        self.Wv = nn.Parameter(torch.empty(d_model, d_model))
        self.proj = nn.Parameter(torch.empty(d_model, d_model))
        for p in (self.Uq, self.Uk, self.Wv, self.proj):
            nn.init.xavier_uniform_(p)

    def forward(self, x, cache_k=None, cache_v=None, pos_start=0,
                hard_gate=False, collect_stats=False):
        B, T, d = x.shape
        Q = (x @ self.Uq).view(B, T, self.h, self.rh)
        K = (x @ self.Uk).view(B, T, self.h, self.rh)
        V = (x @ self.Wv).view(B, T, self.h, self.dh).transpose(1, 2)
        # 相对 RoPE 在 transpose 之前作用（cos/sin 形状 [B,T,h,rh/2]，与 rope_rotate 约定一致）
        pos = torch.arange(pos_start, pos_start + T, device=x.device).float()
        inv = 1.0 / (10000 ** (torch.arange(0, self.rh, 2, device=x.device).float() / self.rh))
        ang = pos[:, None] * inv[None, :]
        cos = torch.cos(ang)[None, :, None, :].expand(B, T, self.h, -1)
        sin = torch.sin(ang)[None, :, None, :].expand(B, T, self.h, -1)
        Q, K = rope_rotate(Q, cos, sin), rope_rotate(K, cos, sin)
        Q, K = Q.transpose(1, 2), K.transpose(1, 2)
        if cache_k is not None:
            K = torch.cat([cache_k, K], dim=1)
            V = torch.cat([cache_v, V], dim=1)
        S = (Q @ K.transpose(-1, -2)) / math.sqrt(self.dh)
        L = K.shape[2]
        qpos = torch.arange(pos_start, pos_start + T, device=x.device)[:, None]
        kpos = torch.arange(0, L, device=x.device)[None, :]
        S = S.masked_fill(kpos > qpos, float("-inf"))
        A = torch.softmax(S, -1)
        O = (A @ V).transpose(1, 2).reshape(B, T, d)
        stats = {"score_macs": B * self.h * T * L * self.rh,
                 "full_graph_macs": B * T * d * 2 * self.rh + 2 * B * self.h * T * L * self.rh +
                                    B * T * d * d * 2}
        return O @ self.proj, K, V, stats


class LRBlock(nn.Module):
    def __init__(self, d_model, nhead, r):
        super().__init__()
        self.attn = LowRankQKAttention(d_model, nhead, r)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(),
                                nn.Linear(d_model * 2, d_model))
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, ck=None, cv=None, pos_start=0):
        a, k, v, stats = self.attn(self.ln1(x), ck, cv, pos_start)
        x = x + a
        x = x + self.ff(self.ln2(x))
        return x, k, v, stats


class LowRankInteractionM4(M4Base):
    def __init__(self, vocab_size, d_model=64, nhead=4, layers=2, r=16):
        super().__init__()
        self.cfg = {"d_model": d_model, "nhead": nhead, "layers": layers, "r": r}
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.blocks = nn.ModuleList([LRBlock(d_model, nhead, r) for _ in range(layers)])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, tokens, state=None, valid_mask=None, collect_stats=False):
        B, T = tokens.shape
        pos_start = state["pos"] if state else 0
        x = self.emb(tokens)
        nck, ncv, stats = [], [], {}
        for i, blk in enumerate(self.blocks):
            ck = state["ck"][i] if state and state["ck"] is not None else None
            cv = state["cv"][i] if state and state["cv"] is not None else None
            x, k, v, s = blk(x, ck, cv, pos_start)
            nck.append(k); ncv.append(v); stats[f"block{i}"] = s
        logits = self.head(self.ln(x))
        return logits, {"ck": nck, "cv": ncv, "pos": pos_start + T, "stats": stats}

    def initial_state(self, batch, device):
        return {"ck": None, "cv": None, "pos": 0}
