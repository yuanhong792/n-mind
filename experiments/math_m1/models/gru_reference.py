"""M1-A：参数量接近的普通 GRU 对照，不作为最终神经核心发布。"""
from __future__ import annotations

import torch
from torch import nn

PAD = 0


class GRUReferenceM1(nn.Module):
    def __init__(self, vocab_size: int, d_model=64, hidden=128, layers=1):
        super().__init__()
        self.cfg = {"d_model": d_model, "hidden": hidden, "layers": layers}
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.cell = nn.GRUCell(d_model, hidden)
        self.mix = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden))
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, tokens, state=None, valid_mask=None, collect_stats=False):
        B, T = tokens.shape
        dev = tokens.device
        h = torch.zeros(B, self.cfg["hidden"], device=dev) if state is None else state
        logits = []
        for t in range(T):
            z = self.emb(tokens[:, t])
            h_new = self.cell(z, h)
            if valid_mask is not None:
                v = valid_mask[:, t].to(h.dtype).unsqueeze(-1)
                h = v * h_new + (1.0 - v) * h
            else:
                h = h_new
            logits.append(self.head(self.mix(h)))
        return torch.stack(logits, dim=1), h

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def state_elements(self):
        return self.cfg["hidden"]

    def config_dict(self):
        return {"model_type": "M1-A_gru_reference", **self.cfg}
