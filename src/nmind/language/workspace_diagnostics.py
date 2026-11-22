"""Workspace 读写诊断（Gate-B 004 第三阶段）：槽位占用、读取注意力、内容漂移。

无训练、无权重副作用：只读 forward 期间的 Workspace 状态序列，回答
"事实写进了哪个槽、被谁读走、候选更新后槽内容漂移了多少"。
"""
from __future__ import annotations

from typing import Any

import torch

from .neural_dialogue import LMState, NMindLM
from .tokenizer import CharTokenizer


def workspace_trajectory(model: NMindLM, tok: CharTokenizer, prompt: str) -> dict[str, Any]:
    """单条 prompt 的 Workspace 状态轨迹（每 token 一步）。

    记录：各槽占用度、写入分配（哪块被写进哪个槽的近似——由 write 的 assign
    softmax 无法直接取用，这里用各步 payload 与最终槽内容的余弦近似）、读取
    注意力熵。"""
    model.eval()
    ids = tok.encode(prompt)
    dev = next(model.parameters()).device
    st: LMState = model.initial_state(1, dev)
    err = torch.full((1, 1), 0.5, device=dev)
    unc = torch.full((1, 1), 0.5, device=dev)
    rows = torch.arange(1, device=dev)[:, None].expand(-1, model.top_k).reshape(-1)
    steps: list[dict[str, Any]] = []
    with torch.no_grad():
        for step, tid in enumerate(ids):
            z = model.tok_emb(torch.tensor([tid], dtype=torch.long, device=dev))
            w_read = model.workspace.read(st.ws, z)
            route = model.router.route(z, st.h, w_read.value, err, unc, k=model.top_k)
            h_new, _g, _m, _p, _n = model.blocks.forward_sparse(z, st.h, w_read.value,
                                                                 route.indices)
            payload = model.block_to_ws(h_new[rows, sel := route.indices.reshape(-1)]).view(
                1, model.top_k, -1)
            wr = model.workspace.write(st.ws, payload)
            attn = w_read.attn[0]
            ent = float(-(attn * attn.clamp_min(1e-12).log()).sum())
            steps.append({
                "step": step, "token": tok.id2tok.get(tid, "<unk>"),
                "occupied": [round(float(v), 4) for v in wr.occupied[0]],
                "read_attn_entropy": round(ent, 4),
                "payload_slot_cosine": [round(float(torch.nn.functional.cosine_similarity(
                    payload[0, m], wr.state["content"][0, s], dim=0)), 4)
                    for m in range(payload.shape[1]) for s in range(wr.state["content"].shape[1])],
                "written_by_route": [int(i) for i in route.indices[0]],
            })
            st = LMState(h=h_new, ws={"content": wr.state["content"],
                                       "occupied": wr.occupied, "utility": wr.utility})
    final_occ = [round(float(v), 4) for v in st.ws["occupied"][0]]
    return {"prompt": prompt, "n_tokens": len(ids), "final_occupied": final_occ,
             "total_occupied": round(sum(final_occ), 4),
             "mean_read_attn_entropy": round(sum(s["read_attn_entropy"] for s in steps)
                                              / max(1, len(steps)), 4),
             "steps": steps}


def ws_content_drift(model_a: NMindLM, model_b: NMindLM, tok: CharTokenizer,
                      prompt: str) -> dict[str, Any]:
    """同 prompt 下两个模型的 Workspace 末态漂移（逐槽余弦 + 占用差）。"""
    outs = []
    for m in (model_a, model_b):
        m.eval()
        with torch.no_grad():
            _lg, st = m(torch.tensor([tok.encode(prompt)], dtype=torch.long))
        outs.append(st.ws)
    content_a, content_b = outs[0]["content"][0], outs[1]["content"][0]
    occ_a, occ_b = outs[0]["occupied"][0], outs[1]["occupied"][0]
    per_slot = []
    for s in range(content_a.shape[0]):
        per_slot.append({
            "slot": s,
            "cosine": round(float(torch.nn.functional.cosine_similarity(
                content_a[s], content_b[s], dim=0)), 4),
            "occupied_diff": round(float(occ_b[s] - occ_a[s]), 4),
        })
    return {"prompt": prompt, "slots": per_slot,
             "note": "候选更新后槽内容漂移——漂移大说明 Workspace 写入层承载了更新主效应"}
