"""R2 tiny Transformer 参考：小规模自注意力，用于对照显式记忆。"""
from __future__ import annotations

import math

import torch
from torch import nn

from base import M2Base, PAD


class TinyTransformerM2(M2Base):
    def __init__(self, vocab_size: int, d_model=64, nhead=4, layers=2, max_len=160):
        super().__init__()
        self.cfg = {"d_model": d_model, "nhead": nhead, "layers": layers, "max_len": max_len}
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                           dim_feedforward=d_model * 2,
                                           batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.head = nn.Linear(d_model, vocab_size)
        self._pos_cache = {}

    def forward(self, tokens, state=None, valid_mask=None, collect_stats=False):
        B, T = tokens.shape
        dev = tokens.device
        if state is None:
            # 增量缓存：保存已处理的 token
            mem = tokens
        else:
            mem = torch.cat([state["mem"], tokens], dim=1)
        L = mem.size(1)
        pos_ids = torch.arange(L, device=dev).clamp(max=self.cfg["max_len"] - 1)
        x = self.emb(mem) + self.pos(pos_ids)[None]
        # 因果掩码
        attn_mask = torch.triu(torch.ones(L, L, device=dev, dtype=torch.bool), diagonal=1)
        out = self.encoder(x, mask=attn_mask)
        logits = self.head(out)[:, -T:, :]
        return logits, {"mem": mem.detach()}

    def state_elements(self):
        return self.cfg["max_len"] * self.cfg["d_model"]
