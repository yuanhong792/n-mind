"""M3-A：低秩神经计算（对角状态保留 + 低秩增量 + 轻量读出）。

状态 h:[d]；更新：
    candidate = tanh( U(Vh) + diag*a*h + Wx*x + b )
    h_new = (1-g)*h + g*candidate
其中 V:[r,d], U:[d,r]，r<<d，理论更新成本 O(dr+d)。
无复杂记忆模块；目标：检验固定低秩计算能否保持相同能力。
"""

import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from base import M2Base as M3Base, PAD  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "math_lab"))
from low_rank_operator import low_rank_step  # noqa: E402


class LowRankNeuralM3(M3Base):
    def __init__(self, vocab_size, d=96, r=16, n_layers=1):
        super().__init__()
        self.cfg = {"d": d, "r": r, "n_layers": n_layers}
        self.emb = nn.Embedding(vocab_size, d, padding_idx=PAD)
        self.V = nn.Parameter(torch.empty(r, d))
        self.U = nn.Parameter(torch.empty(d, r))
        self.diag = nn.Parameter(torch.ones(d) * 0.9)
        self.bias = nn.Parameter(torch.zeros(d))
        self.Wx = nn.Linear(d, d, bias=False)
        self.gate_proj = nn.Linear(d, d)
        self.head = nn.Linear(d, vocab_size)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.V, a=5 ** 0.5)
        nn.init.zeros_(self.U)

    def step(self, h, x):
        z = torch.matmul(h, self.V.transpose(-1, -2))
        delta = torch.matmul(z, self.U.transpose(-1, -2))
        candidate = torch.tanh(delta + self.diag * h + self.Wx(x) + self.bias)
        g = torch.sigmoid(self.gate_proj(h))
        return (1 - g) * h + g * candidate

    def forward(self, tokens, state=None, valid_mask=None, collect_stats=False,
                head_every=1):
        B, T = tokens.shape
        dev = tokens.device
        h = torch.zeros(B, self.cfg["d"], device=dev) if state is None else state["h"]
        logits = []
        for t in range(T):
            x = self.emb(tokens[:, t])
            h_new = self.step(h, x)
            if valid_mask is not None:
                v = valid_mask[:, t].to(h.dtype).unsqueeze(-1)
                h = v * h_new + (1 - v) * h
            else:
                h = h_new
            # head_every: 每 N 步投影一次，推理只需末步时可设大，跳过中间投影
            if (t + 1) % head_every == 0 or t == T - 1:
                logits.append(self.head(h))
            else:
                logits.append(None)
        # 用零占位保证形状，评测只取有效位置
        if any(x is None for x in logits):
            logits = [x if x is not None else torch.zeros(B, self.head.out_features, device=dev)
                      for x in logits]
        return torch.stack(logits, dim=1), {"h": h}

    def initial_state(self, batch, device):
        return {"h": torch.zeros(batch, self.cfg["d"], device=device)}

    def state_elements(self):
        return self.cfg["d"]

    def config_dict(self):
        return {"model_type": "M3-A_low_rank_neural", **self.cfg,
                "state_update_macs": 2 * self.cfg["d"] * self.cfg["r"] + 4 * self.cfg["d"]}
