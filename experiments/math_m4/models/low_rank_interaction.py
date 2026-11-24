"""M4-A：Low-Rank Information Interaction 模型（M4.1 修复版）。

与 R2 完全相同的层数/Embedding/FFN/Value 维度/位置编码/输出 Head，
唯一变量是 Q/K 交互维度 r（Value 表达维度保持 d）。
不加入低秩投影分解或正交压缩（那些只在 math_lab 做算子验证）。

M4.1 修复：
- transpose 后 K/V cache 必须沿序列维 dim=2 拼接，不能沿头维 dim=1；
- QK score 缩放使用实际 reduced_head_dim，而不是 Value 维度；
- 完整统计 Q/K 投影、score、Value 聚合、输出投影与 cache 字节；
- valid_mask 屏蔽 padding key，保证 padding isolation。
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
                valid_mask=None, cache_valid=None, hard_gate=False,
                collect_stats=False):
        B, T, d = x.shape
        Q = (x @ self.Uq).view(B, T, self.h, self.rh)
        K = (x @ self.Uk).view(B, T, self.h, self.rh)
        V = (x @ self.Wv).view(B, T, self.h, self.dh).transpose(1, 2)
        pos = torch.arange(pos_start, pos_start + T, device=x.device).float()
        inv = 1.0 / (10000 ** (torch.arange(0, self.rh, 2, device=x.device).float() / self.rh))
        ang = pos[:, None] * inv[None, :]
        cos = torch.cos(ang)[None, :, None, :].expand(B, T, self.h, -1)
        sin = torch.sin(ang)[None, :, None, :].expand(B, T, self.h, -1)
        Q, K = rope_rotate(Q, cos, sin), rope_rotate(K, cos, sin)
        # 先 RoPE，再换到 [B,h,T,*]；此后序列维固定为 dim=2。
        Q, K = Q.transpose(1, 2), K.transpose(1, 2)

        cache_len = cache_k.shape[2] if cache_k is not None else 0
        if cache_k is not None:
            # 修复点：K/V 已经是 [B,h,L,*]，必须沿序列维 dim=2 拼接。
            K = torch.cat([cache_k, K], dim=2)
            V = torch.cat([cache_v, V], dim=2)

        key_ok = torch.ones(B, K.shape[2], dtype=torch.bool, device=x.device)
        if cache_valid is not None:
            key_ok[:, :cache_len] = cache_valid
        if valid_mask is not None:
            key_ok[:, cache_len:] = valid_mask

        L = K.shape[2]
        S = (Q @ K.transpose(-1, -2)) / math.sqrt(self.rh)
        qpos = torch.arange(pos_start, pos_start + T, device=x.device)[:, None]
        kpos = torch.arange(0, L, device=x.device)[None, :]
        mask = (kpos > qpos) | (~key_ok[:, None, None, :])
        S = S.masked_fill(mask, float("-inf"))
        A = torch.nan_to_num(torch.softmax(S, -1), nan=0.0)
        O = (A @ V).transpose(1, 2).reshape(B, T, d)

        qk_proj_macs = 2 * B * T * d * self.h * self.rh
        score_macs = B * self.h * T * L * self.rh
        value_proj_macs = B * T * d * d
        value_aggregation_macs = B * self.h * T * L * self.dh
        output_proj_macs = B * T * d * d
        total_macs = (qk_proj_macs + score_macs + value_proj_macs +
                      value_aggregation_macs + output_proj_macs)
        stats = {
            "reduced_head_dim": self.rh,
            "value_head_dim": self.dh,
            "qk_proj_macs": qk_proj_macs,
            "score_macs": score_macs,
            "dense_reference_score_macs": B * self.h * T * L * self.dh,
            "value_proj_macs": value_proj_macs,
            "value_aggregation_macs": value_aggregation_macs,
            "output_proj_macs": output_proj_macs,
            "total_macs": total_macs,
            "cache_elements": int(K.numel() + V.numel()),
            "cache_bytes_fp32": int((K.numel() + V.numel()) * K.element_size()),
        }
        return O @ self.proj, K, V, stats


class LRBlock(nn.Module):
    def __init__(self, d_model, nhead, r):
        super().__init__()
        self.attn = LowRankQKAttention(d_model, nhead, r)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(),
                                nn.Linear(d_model * 2, d_model))
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, ck=None, cv=None, pos_start=0,
                valid_mask=None, cache_valid=None, hard_gate=False):
        a, k, v, stats = self.attn(self.ln1(x), ck, cv, pos_start,
                                   valid_mask, cache_valid, hard_gate)
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
        cur_valid = valid_mask if valid_mask is not None else torch.ones(
            B, T, dtype=torch.bool, device=tokens.device)
        cache_valid = state.get("key_valid") if state else None
        x = self.emb(tokens)
        nck, ncv, stats = [], [], {}
        for i, blk in enumerate(self.blocks):
            ck = state["ck"][i] if state and state["ck"] is not None else None
            cv = state["cv"][i] if state and state["cv"] is not None else None
            x, k, v, s = blk(x, ck, cv, pos_start, cur_valid, cache_valid)
            nck.append(k); ncv.append(v); stats[f"block{i}"] = s
        logits = self.head(self.ln(x))
        key_valid = (torch.cat([cache_valid, cur_valid], dim=1)
                     if cache_valid is not None else cur_valid)
        cache = {"ck": nck, "cv": ncv, "pos": pos_start + T,
                 "key_valid": key_valid, "stats": stats}
        return logits, cache

    def initial_state(self, batch, device):
        return {"ck": None, "cv": None, "pos": 0, "key_valid": None}
