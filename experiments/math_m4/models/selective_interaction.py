"""M4-B：Selective Information Interaction 模型（M4.1 因果修复版）。

M4.1 关键修复：
- 已完成历史分组：group_end < query_pos 才允许用完整摘要；
- 当前未完成分组：只使用已到达位置，且仍受因果 mask；
- 未来分组完全不可见；
- padding key 不进摘要、不进精确注意力；
- 选中位置之间联合 softmax，不再按分组独立 softmax 后相加；
- 训练与部署执行同样的硬路由；STE 梯度只给已选路径；
- 输出保持 [B,T,d]，禁止 transpose 后错误 reshape 造成位置混合；
- stats 区分理论选择率、真实张量执行率、摘要/粗检索/top-k/gather 成本。
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


class GroupedSelectiveAttention(nn.Module):
    def __init__(self, d_model, nhead, group_size=4, top_k=2):
        super().__init__()
        assert d_model % nhead == 0
        self.d, self.h, self.dh = d_model, nhead, d_model // nhead
        self.gs, self.top_k = group_size, top_k
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.summary_proj = nn.Linear(self.dh, self.dh, bias=False)
        self.scale = math.sqrt(self.dh)

    @staticmethod
    def _gather(Kt, Vt, pos):
        """pos: [B,h,T,S]，返回 K/V [B,h,T,S,dh]。"""
        B, h, T, S = pos.shape
        L, dh = Kt.shape[2], Kt.shape[-1]
        idx = pos.reshape(B * h * T, S).unsqueeze(-1).expand(-1, -1, dh)
        k_exp = Kt.unsqueeze(2).expand(B, h, T, L, dh).reshape(B * h * T, L, dh)
        v_exp = Vt.unsqueeze(2).expand(B, h, T, L, dh).reshape(B * h * T, L, dh)
        k_flat = k_exp.gather(1, idx)
        v_flat = v_exp.gather(1, idx)
        return k_flat.reshape(B, h, T, S, dh), v_flat.reshape(B, h, T, S, dh)

    def forward(self, x, cache_k=None, cache_v=None, pos_start=0,
                valid_mask=None, cache_valid=None, hard_gate=False,
                collect_stats=False):
        B, T, d = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.h, self.dh)
        Q, K, V = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        pos = torch.arange(pos_start, pos_start + T, device=x.device).float()
        inv = 1.0 / (10000 ** (torch.arange(0, self.dh, 2, device=x.device).float() / self.dh))
        ang = pos[:, None] * inv[None, :]
        cos = torch.cos(ang)[None, :, None, :].expand(B, T, self.h, -1)
        sin = torch.sin(ang)[None, :, None, :].expand(B, T, self.h, -1)
        Q, K = rope_rotate(Q, cos, sin), rope_rotate(K, cos, sin)

        cache_len = cache_k.shape[1] if cache_k is not None else 0
        if cache_k is not None:
            K = torch.cat([cache_k, K], dim=1)
            V = torch.cat([cache_v, V], dim=1)

        key_ok = torch.ones(B, K.shape[1], dtype=torch.bool, device=x.device)
        if cache_valid is not None:
            key_ok[:, :cache_len] = cache_valid
        if valid_mask is not None:
            key_ok[:, cache_len:] = valid_mask

        L = K.shape[1]
        q_abs = torch.arange(pos_start, pos_start + T, device=x.device)  # [T]
        Qt = Q.transpose(1, 2)                                          # [B,h,T,dh]
        Kt = K.transpose(1, 2)                                          # [B,h,L,dh]
        Vt = V.transpose(1, 2)                                          # [B,h,L,dh]

        G = L // self.gs
        k = min(self.top_k, G)
        summary_macs = coarse_macs = topk_comparisons = 0
        sel_K = sel_V = None
        sel_key_ok = None
        route = None

        if G > 0 and k > 0:
            Kc = Kt[:, :, :G * self.gs].reshape(B, self.h, G, self.gs, self.dh)
            gv = key_ok[:, :G * self.gs].reshape(B, G, self.gs).to(Qt.dtype)
            cnt = gv.sum(-1).clamp_min(1.0)                              # [B,G]
            summary = (Kc * gv[:, None, :, :, None]).sum(3) / cnt[:, None, :, None]
            summary = self.summary_proj(summary)                         # [B,h,G,dh]
            summary_macs = B * self.h * G * self.gs * self.dh + B * self.h * G * self.dh * self.dh

            coarse = torch.einsum("bhtd,bhgd->bhtg", Qt, summary) / self.scale
            g_end = torch.arange(G, device=x.device) * self.gs + self.gs - 1
            allowed = (g_end[None, :] < q_abs[:, None])                  # [T,G]
            allowed = allowed[None, :, :] & (cnt > 0)[:, None, :]        # [B,T,G]
            coarse = coarse.masked_fill(~allowed[:, None, :, :], float("-inf"))
            scores, idx = coarse.topk(k, dim=-1)
            sel_group_ok = torch.isfinite(scores)
            idx = idx.masked_fill(~sel_group_ok, 0)
            topk_comparisons = B * self.h * T * G * k

            soft = torch.nan_to_num(torch.softmax(
                scores.masked_fill(~sel_group_ok, float("-inf")), dim=-1), nan=0.0)
            hard_full = torch.zeros(B, self.h, T, G, device=soft.device, dtype=soft.dtype)
            hard_full.scatter_(-1, idx, 1.0)
            hard = hard_full.gather(-1, idx) * sel_group_ok.to(soft.dtype)
            route = (hard.detach() + soft - soft.detach()) if (
                self.training and not hard_gate) else hard
            route = route * sel_group_ok.to(route.dtype)
            coarse_macs = B * self.h * T * G * self.dh

            pos = idx[..., None] * self.gs + torch.arange(
                self.gs, device=x.device).view(1, 1, 1, 1, self.gs)
            pos = pos.clamp(max=max(0, L - 1)).reshape(B, self.h, T, k * self.gs)
            sel_K, sel_V = self._gather(Kt, Vt, pos)
            gv_sel = key_ok.gather(1, pos.reshape(B, -1)).reshape(B, self.h, T, k * self.gs)
            sel_group_mask = (sel_group_ok[..., None]
                              .expand(B, self.h, T, k, self.gs)
                              .reshape(B, self.h, T, k * self.gs))
            sel_key_ok = (gv_sel & sel_group_mask).reshape(
                B, self.h, T, k, self.gs)

        # 当前分组：无论前面是否选中，都必须单独处理已到达位置。
        g_cur = (q_abs // self.gs).clamp(max=max(0, (L - 1) // self.gs))
        loc_pos_raw = g_cur[:, None] * self.gs + torch.arange(
            self.gs, device=x.device)[None, :]
        loc_pos_gather = loc_pos_raw.clamp(max=max(0, L - 1))
        loc_pos_bh = loc_pos_gather[None, None, :, :].expand(
            B, self.h, T, self.gs).reshape(B, self.h, T, self.gs)
        loc_K, loc_V = self._gather(Kt, Vt, loc_pos_bh)
        loc_key_ok = key_ok[:, loc_pos_gather].reshape(B, 1, T, self.gs)
        arrived = (loc_pos_raw <= q_abs[:, None]) & (loc_pos_raw < L)
        loc_key_ok = loc_key_ok & arrived[None, :, :]

        if sel_K is None:
            sel_slots = self.gs
            sel_K = torch.zeros(B, self.h, T, sel_slots, self.dh,
                                device=x.device, dtype=Qt.dtype)
            sel_V = torch.zeros_like(sel_K)
            sel_key_ok = torch.zeros(B, self.h, T, 1, self.gs,
                                     dtype=torch.bool, device=x.device)
            route = torch.zeros(B, self.h, T, 1, device=x.device, dtype=Qt.dtype)
            k_eff = 1
        else:
            k_eff = k
        if True:
            sel_K_flat = sel_K.reshape(B, self.h, T, k_eff * self.gs, self.dh)
            sel_scores = (torch.einsum("bhtd,bhtsd->bhts", Qt, sel_K_flat) /
                          self.scale).reshape(B, self.h, T, k_eff, self.gs)
            log_route = torch.log(route.clamp_min(1e-6))[..., None]
            sel_scores = sel_scores + log_route
            sel_scores = sel_scores.masked_fill(~sel_key_ok, float("-inf"))
            loc_scores = torch.einsum("bhtd,bhtsd->bhts", Qt, loc_K) / self.scale
            loc_scores = loc_scores.masked_fill(~loc_key_ok, float("-inf"))
            flat_scores = torch.cat([sel_scores.reshape(B, self.h, T, -1),
                                     loc_scores.reshape(B, self.h, T, -1)], dim=-1)
            flat_ok = torch.cat([sel_key_ok.reshape(B, self.h, T, -1),
                                 loc_key_ok.expand(B, self.h, T, self.gs).reshape(B, self.h, T, -1)], dim=-1)
            flat_scores = flat_scores.masked_fill(~flat_ok, float("-inf"))
            attn = torch.nan_to_num(torch.softmax(flat_scores, dim=-1), nan=0.0)
            flat_K = torch.cat([sel_K.reshape(B, self.h, T, -1, self.dh),
                                loc_K.reshape(B, self.h, T, -1, self.dh)], dim=-2)
            flat_V = torch.cat([sel_V.reshape(B, self.h, T, -1, self.dh),
                                loc_V.reshape(B, self.h, T, -1, self.dh)], dim=-2)
            out = torch.einsum("bhts,bhtsd->bhtd", attn, flat_V)
            out = out.transpose(1, 2).reshape(B, T, d)
            selected_positions = int(sel_key_ok.sum() + loc_key_ok.sum())
            S = int(sel_key_ok.shape[-1] + loc_key_ok.shape[-1])
            actual_positions = B * self.h * T * S
            actual_macs = 2 * actual_positions * self.dh
            dense_macs = 2 * B * self.h * T * L * self.dh
            stats = {
                "theoretical_selected_positions": selected_positions,
                "theoretical_dense_positions": B * self.h * T * L,
                "theoretical_selection_ratio": round(
                    selected_positions / max(1, B * self.h * T * L), 4),
                "actual_tensor_slots_per_query": S,
                "actual_executed_positions": actual_positions,
                "actual_score_macs": actual_positions * self.dh,
                "actual_value_macs": actual_positions * self.dh,
                "actual_attention_macs": actual_macs,
                "dense_attention_macs": dense_macs,
                "actual_compute_ratio": round(actual_macs / max(1, dense_macs), 4),
                "summary_macs": summary_macs,
                "coarse_retrieval_macs": coarse_macs,
                "topk_comparisons": topk_comparisons,
                "gather_elements": actual_positions,
                "controller_macs": summary_macs + coarse_macs,
                "executed_macs": actual_macs + summary_macs + coarse_macs,
                "full_qk_forbidden": True,
                "joint_normalization": True,
                "train_deploy_route_identical": True,
                "unselected_path_gradient": False,
                "future_groups_visible": False,
                "padding_in_summary": False,
                "hard_gate": bool(hard_gate or not self.training),
            }
            return self.proj(out), K, V, stats



class SelBlock(nn.Module):
    def __init__(self, d_model, nhead, group_size=4, top_k=2):
        super().__init__()
        self.attn = GroupedSelectiveAttention(d_model, nhead, group_size, top_k)
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


class SelectiveInteractionM4(M4Base):
    def __init__(self, vocab_size, d_model=64, nhead=4, layers=2, group_size=4, top_k=2):
        super().__init__()
        self.cfg = {"d_model": d_model, "nhead": nhead, "layers": layers,
                    "group_size": group_size, "top_k": top_k}
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.blocks = nn.ModuleList([SelBlock(d_model, nhead, group_size, top_k)
                                     for _ in range(layers)])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, tokens, state=None, valid_mask=None, hard_gate=False,
                collect_stats=False):
        B, T = tokens.shape
        pos_start = state["pos"] if state else 0
        cur_valid = valid_mask if valid_mask is not None else torch.ones(
            B, T, dtype=torch.bool, device=tokens.device)
        cache_valid = state.get("key_valid") if state else None
        x = self.emb(tokens)
        nck, ncv, stats_list = [], [], []
        for i, blk in enumerate(self.blocks):
            ck = state["ck"][i] if state and state["ck"] is not None else None
            cv = state["cv"][i] if state and state["cv"] is not None else None
            x, k, v, s = blk(x, ck, cv, pos_start, cur_valid, cache_valid, hard_gate)
            nck.append(k); ncv.append(v); stats_list.append(s)
        logits = self.head(self.ln(x))
        key_valid = (torch.cat([cache_valid, cur_valid], dim=1)
                     if cache_valid is not None else cur_valid)
        stats = {"blocks": stats_list,
                 "total_executed_macs": sum(s["executed_macs"] for s in stats_list),
                 "total_controller_macs": sum(s["controller_macs"] for s in stats_list),
                 "theoretical_selection_ratio": round(
                     sum(s["theoretical_selection_ratio"] for s in stats_list) /
                     max(1, len(stats_list)), 4),
                 "actual_compute_ratio": round(
                     sum(s["actual_compute_ratio"] for s in stats_list) /
                     max(1, len(stats_list)), 4)}
        cache = {"ck": nck, "cv": ncv, "pos": pos_start + T,
                 "key_valid": key_valid, "stats": stats}
        return logits, cache

    def initial_state(self, batch, device):
        return {"ck": None, "cv": None, "pos": 0, "key_valid": None}
