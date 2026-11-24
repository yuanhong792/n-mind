"""R2：微型因果 Transformer，带真正的增量 KV Cache（修复 M2 重算缺陷）。

- 完整前向：一次性计算所有位置，返回 logits 与 cache(K,V per layer)。
- 增量前向：给定上一步 cache，只计算新 token，复用缓存 K/V。
- 因果性：causal mask 保证位置 t 只看 ≤t。
- 位置编码：相对位置用缓存长度偏移，避免 max_len clamp 碰撞（修复 M2 缺陷）。
完整前向与逐 token 增量必须等价（math/self-test 验证）。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from base import M2Base as M3Base, PAD  # noqa: E402


def rope_rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """RoPE：x[...,dh]；cos/sin[...,dh/2]（每个频率一份）。"""
    x_even, x_odd = x[..., 0::2], x[..., 1::2]   # [..., dh/2]
    out = torch.stack([x_even * cos - x_odd * sin,
                       x_even * sin + x_odd * cos], dim=-1)  # [..., dh/2, 2]
    return out.flatten(-2)


class CausalSelfAttentionKV(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.d, self.h = d_model, nhead
        self.dh = d_model // nhead
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, cache_k=None, cache_v=None, pos_start=0,
                valid_mask=None, cache_valid=None):
        B, T, d = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.h, self.dh)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]   # [B,T,h,dh]
        # 相对 RoPE（位置从 pos_start 开始，支持任意长度无碰撞）
        pos = torch.arange(pos_start, pos_start + T, device=x.device).float()
        inv = 1.0 / (10000 ** (torch.arange(0, self.dh, 2, device=x.device).float() / self.dh))
        ang = pos[:, None] * inv[None, :]
        cos = torch.cos(ang)[None, :, None, :].expand(B, T, self.h, -1)
        sin = torch.sin(ang)[None, :, None, :].expand(B, T, self.h, -1)
        q, k = rope_rotate(q, cos, sin), rope_rotate(k, cos, sin)
        cache_len = cache_k.shape[1] if cache_k is not None else 0
        if cache_k is not None:
            k = torch.cat([cache_k, k], dim=1)
            v = torch.cat([cache_v, v], dim=1)
        key_ok = torch.ones(B, k.shape[1], dtype=torch.bool, device=x.device)
        if cache_valid is not None:
            key_ok[:, :cache_len] = cache_valid
        if valid_mask is not None:
            key_ok[:, cache_len:] = valid_mask
        # 注意力：q[B,T,h,dh] @ k[B,L,h,dh]^T
        qt = q.transpose(1, 2)          # [B,h,T,dh]
        kt = k.transpose(1, 2)          # [B,h,L,dh]
        vt = v.transpose(1, 2)
        scores = (qt @ kt.transpose(-1, -2)) / math.sqrt(self.dh)
        L = kt.shape[2]
        # 因果 mask：query 位置 (pos_start + i) 只能看 key 位置 j ≤ pos_start+i
        qpos = torch.arange(pos_start, pos_start + T, device=x.device)[:, None]
        kpos = torch.arange(0, L, device=x.device)[None, :]
        mask = (kpos > qpos) | (~key_ok[:, None, None, :])
        scores = scores.masked_fill(mask, float("-inf"))
        attn = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
        out = (attn @ vt).transpose(1, 2).reshape(B, T, d)
        return self.proj(out), k, v


class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.attn = CausalSelfAttentionKV(d_model, nhead)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(),
                                nn.Linear(d_model * 2, d_model))
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, ck=None, cv=None, pos_start=0,
                valid_mask=None, cache_valid=None):
        a, k, v = self.attn(self.ln1(x), ck, cv, pos_start, valid_mask, cache_valid)
        x = x + a
        x = x + self.ff(self.ln2(x))
        return x, k, v


class TinyTransformerM3(M3Base):
    def __init__(self, vocab_size, d_model=64, nhead=4, layers=2):
        super().__init__()
        self.cfg = {"d_model": d_model, "nhead": nhead, "layers": layers}
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, nhead) for _ in range(layers)])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, tokens, state=None, valid_mask=None, collect_stats=False):
        B, T = tokens.shape
        pos_start = state["pos"] if state else 0
        cur_valid = valid_mask if valid_mask is not None else torch.ones(
            B, T, dtype=torch.bool, device=tokens.device)
        cache_valid = state.get("key_valid") if state else None
        x = self.emb(tokens)
        new_ck, new_cv = [], []
        for i, blk in enumerate(self.blocks):
            ck = state["ck"][i] if state and state["ck"] is not None else None
            cv = state["cv"][i] if state and state["cv"] is not None else None
            x, k, v = blk(x, ck, cv, pos_start, cur_valid, cache_valid)
            new_ck.append(k); new_cv.append(v)
        logits = self.head(self.ln(x))
        key_valid = (torch.cat([cache_valid, cur_valid], dim=1)
                     if cache_valid is not None else cur_valid)
        cache = {"ck": new_ck, "cv": new_cv, "pos": pos_start + T,
                 "key_valid": key_valid}
        return logits, cache

    def initial_state(self, batch, device):
        return {"ck": None, "cv": None, "pos": 0, "key_valid": None}

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def state_elements(self):
        # KV cache 随长度增长，按参考上限 128 计
        c = self.cfg
        return 2 * c["layers"] * 128 * c["d_model"]

    def config_dict(self):
        return {"model_type": "R2_tiny_transformer_kvcache", **self.cfg}
