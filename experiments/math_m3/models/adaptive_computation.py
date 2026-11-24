"""M3-B：自适应最小计算（轻量基础路径 + 按需扩展路径）。

三条路径（由可训练神经控制器从 (x,h) 决定，不含任务字符串规则）：
    C0：保持/轻量更新（对角）
    C1：低秩更新
    C2：扩展神经计算（额外 MLP）

训练：软门 routing_probs 对三条路径加权（稠密训练成本，如实记录）。
推理（hard_gate=True）：argmax 选路径，只执行选中路径 → 真正跳过昂贵计算。
always_expensive=True 对照模式：强制始终走 C2，用于测量跳过收益。

计算成本核算：
- controller 自身成本必须计入。
- 软门训练成本 = C0+C1+C2 全部执行；硬门推理成本 = 仅选中路径 + controller。
"""

import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from base import M2Base as M3Base, PAD  # noqa: E402


class AdaptiveComputationM3(M3Base):
    def __init__(self, vocab_size, d=96, r=16, hidden_mult=2):
        super().__init__()
        self.cfg = {"d": d, "r": r, "hidden_mult": hidden_mult}
        self.emb = nn.Embedding(vocab_size, d, padding_idx=PAD)
        # 基础轻量路径 C0：对角保留
        self.diag = nn.Parameter(torch.ones(d) * 0.9)
        # C1 低秩
        self.V = nn.Parameter(torch.empty(r, d))
        self.U = nn.Parameter(torch.empty(d, r))
        # C2 扩展
        self.ff = nn.Sequential(nn.Linear(d, d * hidden_mult), nn.GELU(),
                                nn.Linear(d * hidden_mult, d))
        # 控制器（神经状态驱动）
        self.controller = nn.Linear(d * 2, 3)
        # 读出
        self.gate_read = nn.Linear(d, d)
        self.head = nn.Linear(d, vocab_size)
        nn.init.kaiming_uniform_(self.V, a=5 ** 0.5)
        nn.init.zeros_(self.U)

    def paths(self, h, x, execute="all"):
        """返回 (h_c0, h_c1_or_none, h_c2_or_none)。"""
        hc0 = self.diag * h + x
        hc1 = hc0 + torch.matmul(torch.matmul(h, self.V.T), self.U.T)
        if execute == "c1":
            return hc0, hc1, None
        hc2 = hc1 + self.ff(hc1)
        return hc0, hc1, hc2

    def forward(self, tokens, state=None, valid_mask=None, collect_stats=False,
                hard_gate=False, always_expensive=False, threshold=None):
        B, T = tokens.shape
        dev = tokens.device
        h = torch.zeros(B, self.cfg["d"], device=dev) if state is None else state["h"]
        logits = []
        stats = {"path_counts": {0: 0, 1: 0, 2: 0}, "controller_macs": 0,
                 "executed_macs": 0} if collect_stats else None
        d, r = self.cfg["d"], self.cfg["r"]
        for t in range(T):
            x = self.emb(tokens[:, t])
            logits_ctrl = self.controller(torch.cat([x, h], dim=-1))
            probs = F.softmax(logits_ctrl, dim=-1)   # [B,3]
            if stats is not None:
                stats["controller_macs"] += B * (2 * d * 3)
            if always_expensive:
                choices = torch.full((B,), 2, device=dev)
            elif hard_gate:
                choices = probs.argmax(-1)
            else:
                choices = None   # 软门：全部执行
            if choices is None:
                hc0, hc1, hc2 = self.paths(h, x, "all")
                h_new = (probs[:, 0:1] * hc0 + probs[:, 1:2] * hc1 + probs[:, 2:3] * hc2)
                if stats is not None:
                    stats["executed_macs"] += B * (d + 2 * d * r + d * d * self.cfg["hidden_mult"] * 2)
                    for k in range(3):
                        stats["path_counts"][k] += B
            else:
                h_new = torch.zeros_like(h)
                for b in range(B):
                    c = int(choices[b].item())
                    if stats is not None:
                        stats["path_counts"][c] += 1
                    xb, hb = x[b:b + 1], h[b:b + 1]
                    if c == 0:
                        h_new[b] = (self.diag * hb + xb).squeeze(0)
                        if stats is not None:
                            stats["executed_macs"] += d
                    elif c == 1:
                        _, hc1, _ = self.paths(hb, xb, "c1")
                        h_new[b] = hc1.squeeze(0)
                        if stats is not None:
                            stats["executed_macs"] += d + 2 * d * r
                    else:
                        _, _, hc2 = self.paths(hb, xb, "all")
                        h_new[b] = hc2.squeeze(0)
                        if stats is not None:
                            stats["executed_macs"] += d + 2 * d * r + d * d * self.cfg["hidden_mult"] * 2
            # 读出门控（基于状态，非答案）
            g = torch.sigmoid(self.gate_read(h_new))
            h_read = g * h_new
            if valid_mask is not None:
                v = valid_mask[:, t].to(h.dtype).unsqueeze(-1)
                h = v * h_new + (1 - v) * h
            else:
                h = h_new
            logits.append(self.head(h_read))
        out = torch.stack(logits, dim=1), {"h": h}
        if collect_stats:
            return out, stats
        return out

    def initial_state(self, batch, device):
        return {"h": torch.zeros(batch, self.cfg["d"], device=device)}

    def state_elements(self):
        return self.cfg["d"]

    def config_dict(self):
        return {"model_type": "M3-B_adaptive_computation", **self.cfg}
