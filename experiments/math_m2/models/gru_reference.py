"""R1 GRU 参考：普通 GRU 基线，无显式记忆。"""
from __future__ import annotations

import torch
from torch import nn

from base import M2Base, PAD, masked_update


class GRUReferenceM2(M2Base):
    def __init__(self, vocab_size: int, d_model=64, hidden=128):
        super().__init__()
        self.cfg = {"d_model": d_model, "hidden": hidden}
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.cell = nn.GRUCell(d_model, hidden)
        self.mix = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU())
        self.head = nn.Linear(hidden, vocab_size)

    def initial_state(self, batch, device):
        return {"h": torch.zeros(batch, self.cfg["hidden"], device=device)}

    def forward(self, tokens, state=None, valid_mask=None, collect_stats=False):
        B, T = tokens.shape
        dev = tokens.device
        st = state or self.initial_state(B, dev)
        h = st["h"]
        logits = []
        for t in range(T):
            z = self.emb(tokens[:, t])
            h_new = self.cell(z, h)
            if valid_mask is not None:
                h = masked_update(h_new, h, valid_mask[:, t])
            else:
                h = h_new
            logits.append(self.head(self.mix(h)))
        return torch.stack(logits, dim=1), {"h": h}

    def state_elements(self):
        return self.cfg["hidden"]
