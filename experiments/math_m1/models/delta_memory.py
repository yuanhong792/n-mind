"""M1-B：轻量控制器 + 共享门 Delta 矩阵关联记忆。所有参数随机初始化。"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
LAB = HERE.parent / "math_lab"
sys.path.insert(0, str(LAB))

from delta_rule import delta_step, l2_normalize

PAD = 0


class DeltaMemoryLM(nn.Module):
    def __init__(self, vocab_size: int, d_model=64, hidden=96, dk=48, dv=48,
                 gate="shared", conv_kernel=1, uncertainty=False):
        super().__init__()
        if gate != "shared" or conv_kernel != 1 or uncertainty:
            raise ValueError("M1-B 固定 shared gate / conv=1 / uncertainty=False")
        self.cfg = {"d_model": d_model, "hidden": hidden, "dk": dk, "dv": dv,
                    "gate": gate, "conv_kernel": conv_kernel, "uncertainty": uncertainty}
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.controller = nn.GRUCell(d_model, hidden)
        self.k_proj = nn.Linear(hidden, dk)
        self.q_proj = nn.Linear(hidden, dk)
        self.v_proj = nn.Linear(hidden, dv)
        self.beta_proj = nn.Linear(hidden, 1)
        self.alpha_proj = nn.Linear(hidden, 1)
        self.fusion = nn.Sequential(nn.Linear(hidden + dv, d_model), nn.GELU())
        self.head = nn.Linear(d_model, vocab_size)

    def initial_state(self, batch, device):
        return {"h": torch.zeros(batch, self.cfg["hidden"], device=device),
                "S": torch.zeros(batch, self.cfg["dv"], self.cfg["dk"], device=device)}

    def forward(self, tokens, state=None, valid_mask=None, collect_stats=False):
        B, T = tokens.shape
        dev = tokens.device
        st = state if state is not None else self.initial_state(B, dev)
        h, S = st["h"], st["S"]
        logits = []
        for t in range(T):
            z = self.emb(tokens[:, t])
            h_new = self.controller(z, h)
            k = l2_normalize(self.k_proj(h_new))
            q = l2_normalize(self.q_proj(h_new))
            v = self.v_proj(h_new)
            beta = torch.sigmoid(self.beta_proj(h_new)).squeeze(-1)
            alpha = torch.sigmoid(self.alpha_proj(h_new)).squeeze(-1)
            S_new = delta_step(S, k, v, beta, alpha)
            if valid_mask is not None:
                vh = valid_mask[:, t].to(h.dtype).unsqueeze(-1)
                vs = vh.unsqueeze(1)
                h = vh * h_new + (1.0 - vh) * h
                S = vs * S_new + (1.0 - vs) * S
            else:
                h, S = h_new, S_new
            read = torch.bmm(S, q.unsqueeze(-1)).squeeze(-1)
            fused = self.fusion(torch.cat([h, read], dim=-1))
            logits.append(self.head(fused))
        return torch.stack(logits, dim=1), {"h": h, "S": S}

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def state_elements(self):
        c = self.cfg
        return c["hidden"] + c["dk"] * c["dv"]

    def config_dict(self):
        return {"model_type": "M1-B_delta_memory", **self.cfg}
