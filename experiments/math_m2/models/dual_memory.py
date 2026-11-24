"""M2-A/B/C/D/E：可配置双记忆模型。

- M2-A: GRU + AM(共享门 delta)，无 SM
- M2-B: GRU + AM(保护门)，无 SM
- M2-C: GRU + SM(序列记忆+指针)，无 AM
- M2-D: GRU + AM(保护门) + SM，独立拼接
- M2-E: GRU + AM + SM + 双记忆融合门 + 短因果卷积（完整模型）
所有参数随机初始化。
"""
from __future__ import annotations

import torch
from torch import nn

from base import M2Base, PAD, masked_update
from associative_memory import am_write, am_read, l2_normalize
from differentiable_pointer import PointerCell, pointer_read, pointer_write, content_address
from causal_conv import CausalDepthwiseConv


class DualMemoryM2(M2Base):
    VARIANTS = {
        "M2-A": dict(use_am=True, protect=False, use_sm=False, fuse=False, conv=False),
        "M2-B": dict(use_am=True, protect=True, use_sm=False, fuse=False, conv=False),
        "M2-C": dict(use_am=False, protect=False, use_sm=True, fuse=False, conv=False),
        "M2-D": dict(use_am=True, protect=True, use_sm=True, fuse=False, conv=False),
        "M2-E": dict(use_am=True, protect=True, use_sm=True, fuse=True, conv=True),
    }

    def __init__(self, vocab_size: int, variant="M2-E", d_model=64, hidden=96,
                 dk=32, dv=32, seq_len=16, d_seq=32, conv_kernel=4):
        super().__init__()
        if variant not in self.VARIANTS:
            raise ValueError(f"unknown variant {variant}")
        flags = self.VARIANTS[variant]
        self.variant = variant
        self.cfg = {"variant": variant, "d_model": d_model, "hidden": hidden,
                    "dk": dk, "dv": dv, "seq_len": seq_len, "d_seq": d_seq,
                    "conv_kernel": conv_kernel, **flags}
        self.use_am, self.use_protect, self.use_sm = flags["use_am"], flags["protect"], flags["use_sm"]
        self.use_fuse, self.use_conv = flags["fuse"], flags["conv"]

        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.controller = nn.GRUCell(d_model, hidden)
        if self.use_conv:
            self.conv = CausalDepthwiseConv(d_model, conv_kernel)
        else:
            self.conv = None

        if self.use_am:
            self.k_proj = nn.Linear(hidden, dk)
            self.q_proj = nn.Linear(hidden, dk)
            self.v_proj = nn.Linear(hidden, dv)
            self.beta_proj = nn.Linear(hidden, 1)
            self.alpha_proj = nn.Linear(hidden, 1)
            if self.use_protect:
                self.protect_proj = nn.Linear(hidden, 1)
            self.am_proj = nn.Linear(dv, hidden)

        if self.use_sm:
            self.pointer = PointerCell(seq_len, d_seq, h_dim=hidden, kernel_size=3)
            self.sv_proj = nn.Linear(hidden, d_seq)
            self.sm_proj = nn.Linear(d_seq, hidden)
            self.step_proj = nn.Linear(hidden, 1)
            self.erase_proj = nn.Linear(hidden, 1)

        # fuse 时 AM/SM 读出先融合成一个向量，故记忆特征只算 1 个
        if self.use_fuse and self.use_am and self.use_sm:
            feat = 1
        else:
            feat = int(self.use_am) + int(self.use_sm)
        self.am_gate_proj = nn.Linear(hidden, 1) if (self.use_fuse and self.use_am and self.use_sm) else None
        self.fusion = nn.Sequential(nn.Linear(hidden * (1 + feat), d_model), nn.GELU())
        self.head = nn.Linear(d_model, vocab_size)

    def initial_state(self, batch, device):
        st = {"h": torch.zeros(batch, self.cfg["hidden"], device=device)}
        if self.use_am:
            st["S"] = torch.zeros(batch, self.cfg["dv"], self.cfg["dk"], device=device)
        if self.use_sm:
            L, ds = self.cfg["seq_len"], self.cfg["d_seq"]
            st["H"] = torch.zeros(batch, L, ds, device=device)
            st["pw"] = self.pointer.initial_pointer(batch, device)
            st["pr"] = self.pointer.initial_pointer(batch, device)
        return st

    def forward(self, tokens, state=None, valid_mask=None, collect_stats=False):
        B, T = tokens.shape
        dev = tokens.device
        st = state if state is not None else self.initial_state(B, dev)
        h = st["h"]
        S = st.get("S")
        H = st.get("H")
        pw = st.get("pw")
        pr = st.get("pr")

        z_all = self.emb(tokens)                       # [B,T,d_model]
        if self.conv is not None:
            z_all = z_all + self.conv(z_all)

        logits = []
        stats = {"protect": [], "fuse": [], "pr": []} if collect_stats else None
        for t in range(T):
            valid = valid_mask[:, t] if valid_mask is not None else torch.ones(B, device=dev, dtype=torch.bool)
            z = z_all[:, t]
            h_new = self.controller(z, h)
            h = masked_update(h_new, h, valid)

            reads = [h]
            if self.use_am:
                k = l2_normalize(self.k_proj(h))
                q = l2_normalize(self.q_proj(h))
                v = self.v_proj(h)
                beta = torch.sigmoid(self.beta_proj(h)).squeeze(-1)
                alpha = torch.sigmoid(self.alpha_proj(h)).squeeze(-1)
                protect = None
                if self.use_protect:
                    protect = torch.sigmoid(self.protect_proj(h)).squeeze(-1)
                    if stats is not None:
                        stats["protect"].append(protect.detach())
                S_new = am_write(S, k, v, beta, alpha, protect)
                S = masked_update(S_new, S, valid)
                am_read_v = self.am_proj(am_read(S, q))
                if self.use_fuse and self.use_sm:
                    reads.append(am_read_v)
                else:
                    reads.append(am_read_v)

            if self.use_sm:
                erase = torch.sigmoid(self.erase_proj(h)).squeeze(-1)
                pw_new = self.pointer.step(H, pw, h, use_content=True)
                pw = masked_update(pw_new, pw, valid)
                sv = self.sv_proj(h)
                H_new = pointer_write(H, pw, sv, erase)
                H = masked_update(H_new, H, valid)
                step_gate = torch.sigmoid(self.step_proj(h)).squeeze(-1) * valid.to(h.dtype)
                pr_new = (1.0 - step_gate[:, None]) * pr + step_gate[:, None] * torch.roll(pr, 1, dims=-1)
                pr = pr_new
                sm_read_v = self.sm_proj(pointer_read(H, pr))
                if self.use_fuse and self.use_am:
                    gamma = torch.sigmoid(self.am_gate_proj(h)).squeeze(-1)
                    if stats is not None:
                        stats["fuse"].append(gamma.detach())
                        stats["pr"].append(pr.detach())
                    fused_mem = gamma[:, None] * am_read_v + (1.0 - gamma[:, None]) * sm_read_v
                    reads[-1] = fused_mem   # 用融合结果替换单独 AM 读出
                else:
                    reads.append(sm_read_v)

            fused = self.fusion(torch.cat(reads, dim=-1))
            logits.append(self.head(fused))
        out_state = {"h": h}
        if self.use_am:
            out_state["S"] = S
        if self.use_sm:
            out_state.update({"H": H, "pw": pw, "pr": pr})
        result = torch.stack(logits, dim=1), out_state
        if collect_stats:
            return result, stats
        return result

    def state_elements(self):
        c = self.cfg
        n = c["hidden"]
        if self.use_am:
            n += c["dk"] * c["dv"]
        if self.use_sm:
            n += c["seq_len"] * c["d_seq"] + 2 * c["seq_len"]
        return n


def build_model(variant, vocab_size):
    return DualMemoryM2(vocab_size, variant=variant)
