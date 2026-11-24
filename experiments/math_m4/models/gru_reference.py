"""R1 GRU 参考：标准 GRU，随机初始化，与 M3 同 tokenizer/数据。"""

import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from base import M2Base as M3Base, PAD


class GRUReferenceM3(M3Base):
    def __init__(self, vocab_size, d_model=64, hidden=128):
        super().__init__()
        self.cfg = {"d_model": d_model, "hidden": hidden}
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.cell = nn.GRUCell(d_model, hidden)
        self.mix = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU())
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, tokens, state=None, valid_mask=None, collect_stats=False):
        B, T = tokens.shape
        dev = tokens.device
        h = torch.zeros(B, self.cfg["hidden"], device=dev) if state is None else state["h"]
        logits = []
        for t in range(T):
            z = self.emb(tokens[:, t])
            h_new = self.cell(z, h)
            if valid_mask is not None:
                v = valid_mask[:, t].to(h.dtype).unsqueeze(-1)
                h = v * h_new + (1 - v) * h
            else:
                h = h_new
            logits.append(self.head(self.mix(h)))
        return torch.stack(logits, dim=1), {"h": h}

    def initial_state(self, batch, device):
        return {"h": torch.zeros(batch, self.cfg["hidden"], device=device)}

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def state_elements(self):
        return self.cfg["hidden"]

    def config_dict(self):
        return {"model_type": "R1_gru_reference", **self.cfg}
