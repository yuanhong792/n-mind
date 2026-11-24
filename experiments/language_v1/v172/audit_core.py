"""V1.7.2 P0/P2：Router 稀疏性与 Workspace 信息保持的冻结参数诊断。

重要口径：
- 本地不存在 experiments/language_v1/v15/candidate_v15_baseline.pt（.gitignore *.pt
  且 V171_FAIR_COMPARISON.json 记录 LOST_weights_equal_parent），因此**无法在真实
  父权重上验证**。本脚本先按历史 parent_params=279363 反推真实配置，再用同配置
  随机权重做架构级行为诊断。所有涉及权重的结论标注 ARCHITECTURE_LEVEL /
  NOT_YET_VERIFIED_ON_PARENT，绝不冒充父模型实测。
- 不更新任何磁盘 checkpoint；临时副本上的一步梯度检查单独计数。
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nmind.language.neural_dialogue import LMConfig, NMindLM
from v172_protocol import build_tokenizer, encode_autoregressive, write_json

OUT = Path(__file__).resolve().parent
HISTORICAL_PARENT_PARAMS = 279363


def cfg_params(cfg: LMConfig) -> int:
    """按 neural_dialogue.NMindLM 的实际层计算参数量。"""
    D, N, H, W, S, V = (cfg.d_model, cfg.n_blocks, cfg.block_hidden,
                        cfg.workspace_dim, cfg.workspace_slots, cfg.vocab_size)
    tok = V * D
    ncell = N * (2 * H * (D + H + W) + 2 * H)
    hidden = max(16, D // 2)
    # load_bias 是 register_buffer，不计入 parameters()，因此第二层 bias 只加一次 N
    router = (D + W + N + 2) * hidden + hidden + hidden * N + N
    ws = S * W + (D * W + W) + W * W + W * W + W * W + (W + 1)
    b2ws = H * W + W
    head = H * V + V
    readout = (H + D + W) * H + H if cfg.query_readout else 0
    return tok + ncell + router + ws + b2ws + head + readout


def infer_configs() -> list[dict]:
    """枚举与历史参数量精确匹配的候选配置。"""
    hits = []
    for D, H, W in itertools.product((32, 48, 64, 96, 128), (32, 48, 64, 96), (16, 32, 48, 64)):
        for N, S in itertools.product((2, 4, 8), (2, 4, 8)):
            for K in (1, 2, 4):
                if K > N:
                    continue
                cfg = LMConfig(vocab_size=4096, d_model=D, n_blocks=N,
                               block_hidden=H, workspace_dim=W,
                               workspace_slots=S, top_k=K)
                n = cfg_params(cfg)
                if n == HISTORICAL_PARENT_PARAMS:
                    hits.append(cfg.to_dict())
    return hits


def probe_router(model: NMindLM, tok, prompts: list[str]) -> dict:
    """六项冻结诊断：评分逐 token 变化、选中集合变化、评分改前向、顺序改写入、
    梯度非零、临时副本一步更新后路由变化。"""
    torch.manual_seed(20261124)
    model.eval()
    model.use_router_load_update = False
    model.router_gradient_path_enabled = True

    # 1/2. 逐 token 的路由评分与选中块集合
    traces = []
    for p in prompts:
        ids = tok.encode(p)
        st = model.initial_state(1, torch.device("cpu"))
        err = torch.full((1, 1), 0.5)
        unc = torch.full((1, 1), 0.5)
        rows = []
        with torch.inference_mode():
            for t, tid in enumerate(ids):
                z = model.tok_emb(torch.tensor([tid]))
                w = model.workspace.read(st.ws, z).value
                r = model.router.route(z, st.h, w, err, unc, k=model.top_k)
                rows.append({"t": t, "token": tid,
                             "scores": [round(x, 5) for x in r.scores[0].tolist()],
                             "indices": r.indices[0].tolist()})
                h, _, _, _, _ = model.blocks.forward_sparse(
                    z, st.h, w, r.indices, weights=None)
                wr = model.workspace.write(st.ws, model.block_to_ws(
                    h[torch.arange(1)[:, None].expand(-1, model.top_k).reshape(-1),
                      r.indices.reshape(-1)]).view(1, model.top_k, -1))
                st = type(st)(h=h, ws={"content": wr.state["content"],
                                       "occupied": wr.occupied, "utility": wr.utility})
        traces.append({"prompt": p[:60], "steps": rows})

    score_vectors = [torch.tensor(r["scores"]) for tr in traces for r in tr["steps"]]
    score_var = float(torch.stack(score_vectors).std(dim=0).mean()) if len(score_vectors) > 1 else 0.0
    selected_sets = {tuple(r["indices"]) for tr in traces for r in tr["steps"]}

    # 3. 修改 router 评分是否改变前向 logits（冻结参数干预，内存副本）
    ids = tok.encode(prompts[0])
    x = torch.tensor([ids])
    with torch.inference_mode():
        base_logits, _ = model(x)
    tmp = copy.deepcopy(model)
    with torch.no_grad():
        tmp.router.score_mlp[0].weight.add_(0.5)
        tmp.router.score_mlp[2].weight.mul_(-1.0)
    with torch.inference_mode():
        pert_logits, _ = tmp(x)
    forward_delta = float((base_logits - pert_logits).abs().mean())

    # 4. 交换选中块顺序是否改变 Workspace 写入（直接干预 route.indices）
    with torch.inference_mode():
        st0 = model.initial_state(1, torch.device("cpu"))
        # 先跑若干真实 token 让 h/workspace 非零：全零初始态下交换顺序必然无差异，
        # 那样的"无影响"结论是假阴性，不能作为证据。
        for tid in ids[:6]:
            zz = model.tok_emb(torch.tensor([tid]))
            ww = model.workspace.read(st0.ws, zz).value
            rr = model.router.route(zz, st0.h, ww, torch.full((1, 1), 0.5),
                                    torch.full((1, 1), 0.5), k=model.top_k)
            hh, _, _, _, _ = model.blocks.forward_sparse(zz, st0.h, ww, rr.indices, weights=None)
            ridx = torch.arange(1)[:, None].expand(-1, model.top_k).reshape(-1)
            pp = model.block_to_ws(hh[ridx, rr.indices.reshape(-1)]).view(1, model.top_k, -1)
            ww2 = model.workspace.write(st0.ws, pp)
            st0 = type(st0)(h=hh, ws={"content": ww2.state["content"],
                                      "occupied": ww2.occupied, "utility": ww2.utility})
        z = model.tok_emb(torch.tensor([ids[0]]))
        w = model.workspace.read(st0.ws, z).value
        r = model.router.route(z, st0.h, w, torch.full((1, 1), 0.5),
                               torch.full((1, 1), 0.5), k=model.top_k)
        payload = model.block_to_ws(st0.h[0, r.indices[0]]).unsqueeze(0)
        wr_a = model.workspace.write(st0.ws, payload)
        swapped = r.indices.clone()
        if swapped.shape[-1] > 1:
            swapped[0] = swapped[0].flip(0)
        payload_b = model.block_to_ws(st0.h[0, swapped[0]]).unsqueeze(0)
        wr_b = model.workspace.write(st0.ws, payload_b)
    order_effect = float((wr_a.state["content"] - wr_b.state["content"]).abs().mean())
    indices_changed = bool(not torch.equal(r.indices[0], swapped[0]))

    # 5. Router 梯度是否非零（只反传不更新）
    ts = encode_autoregressive(tok, prompts[0], "红色")
    x = torch.tensor([list(ts.sequence[:-1])])
    model.train()
    model.use_router_load_update = False
    logits, _ = model(x)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), torch.tensor(list(ts.sequence[1:])).reshape(-1))
    model.zero_grad(set_to_none=True)
    loss.backward()
    router_grads = {n: (None if p.grad is None else round(float(p.grad.abs().sum()), 8))
                    for n, p in model.named_parameters() if n.startswith("router.")}
    model.eval()
    model.zero_grad(set_to_none=True)

    # 6. 临时副本一步梯度更新后路由是否变化（父模型权重零改动）
    tmp2 = copy.deepcopy(model)
    tmp2.train()
    tmp2.use_router_load_update = False
    opt = torch.optim.SGD([p for n, p in tmp2.named_parameters() if n.startswith("router.")], lr=0.1)
    before = []
    with torch.inference_mode():
        tmp2.eval()
        for tid in ids[:8]:
            zz = tmp2.tok_emb(torch.tensor([tid]))
            ss = tmp2.initial_state(1, torch.device("cpu"))
            ww = tmp2.workspace.read(ss.ws, zz).value
            before.append(tmp2.router.route(zz, ss.h, ww, torch.full((1, 1), 0.5),
                                             torch.full((1, 1), 0.5), k=tmp2.top_k).indices[0].tolist())
        tmp2.train()
    lg2, _ = tmp2(x)
    l2 = torch.nn.functional.cross_entropy(
        lg2.reshape(-1, lg2.shape[-1]), torch.tensor(list(ts.sequence[1:])).reshape(-1))
    l2.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    after = []
    with torch.inference_mode():
        tmp2.eval()
        for tid in ids[:8]:
            zz = tmp2.tok_emb(torch.tensor([tid]))
            ss = tmp2.initial_state(1, torch.device("cpu"))
            ww = tmp2.workspace.read(ss.ws, zz).value
            after.append(tmp2.router.route(zz, ss.h, ww, torch.full((1, 1), 0.5),
                                            torch.full((1, 1), 0.5), k=tmp2.top_k).indices[0].tolist())

    cfg = model.cfg
    sparse_capacity = cfg.top_k < cfg.n_blocks
    return {
        "config": cfg.to_dict(),
        "top_k_equals_n_blocks": cfg.top_k == cfg.n_blocks,
        "has_sparse_selection_space": sparse_capacity,
        "activation_ratio": round(cfg.top_k / cfg.n_blocks, 4),
        "router_score_std_across_tokens": round(score_var, 8),
        "distinct_selected_block_sets": [list(s) for s in sorted(selected_sets)],
        "distinct_selected_set_count": len(selected_sets),
        "router_score_change_changes_forward_logits": forward_delta > 1e-8,
        "forward_logit_mean_abs_delta": round(forward_delta, 8),
        "swap_selected_order_changes_workspace": order_effect > 1e-8 and indices_changed,
        "workspace_content_mean_abs_delta_after_swap": round(order_effect, 8),
        "router_grad_abs_sum": router_grads,
        "router_has_nonzero_gradient": any(v is not None and v > 0 for v in router_grads.values()),
        "temporary_copy_one_step_route_changed": before != after,
        "route_before_temporary_step": before,
        "route_after_temporary_step": after,
        "parent_checkpoint_updated": False,
        "traces": traces,
    }


def probe_workspace(model: NMindLM, tok, n_tokens_list=(0, 8, 16, 32, 64)) -> dict:
    """P2：占用、槽内容、注意力、分配、溢出频率、信息恢复、N-Cell 兜底。"""
    model.eval()
    model.use_router_load_update = False
    fact = "已知：小明有红色苹果。"
    query = "问：小明的苹果是什么颜色？答："
    distract = "今天天气不错，路上的车很多， shop 开门了。"
    rows = []
    for n_distract in n_tokens_list:
        text = fact + distract * max(0, n_distract // 8) + query
        ids = tok.encode(text)
        st = model.initial_state(1, torch.device("cpu"))
        overflow_events = 0
        attn_last = None
        occ_trace = []
        assignments_sum = 0.0
        with torch.inference_mode():
            for tid in ids:
                z = model.tok_emb(torch.tensor([tid]))
                w_read = model.workspace.read(st.ws, z)
                attn_last = w_read.attn[0].tolist()
                err = torch.full((1, 1), 0.5)
                unc = torch.full((1, 1), 0.5)
                r = model.router.route(z, st.h, w_read.value, err, unc, k=model.top_k)
                h, _, _, _, _ = model.blocks.forward_sparse(z, st.h, w_read.value, r.indices, weights=None)
                rows_idx = torch.arange(1)[:, None].expand(-1, model.top_k).reshape(-1)
                payload = model.block_to_ws(h[rows_idx, r.indices.reshape(-1)]).view(1, model.top_k, -1)
                wr = model.workspace.write(st.ws, payload)
                overflow_events += len(wr.overflow)
                assignments_sum += float(wr.assignments.sum())
                occ_trace.append([round(v, 4) for v in wr.occupied[0].tolist()])
                st = type(st)(h=h, ws={"content": wr.state["content"],
                                       "occupied": wr.occupied, "utility": wr.utility})
            final_occ = st.ws["occupied"][0]
            final_content = st.ws["content"][0]
            # Workspace 暂停写入对照：保留 N-Cell 状态，ws 冻结为全零
            ws_off = model.workspace.initial_state(1, torch.device("cpu"))
            h_off = st.h
            zq = model.tok_emb(torch.tensor([tok.encode(query)[-1]]))
            on_read = model.workspace.read(st.ws, zq).value
            off_read = model.workspace.read(ws_off, zq).value
            logit_on = model.head(h_off.mean(dim=1) if not model.query_readout else
                                  model.query_readout_proj(torch.cat(
                                      [h_off.mean(dim=1), zq, on_read], dim=-1)))
            logit_off = model.head(h_off.mean(dim=1) if not model.query_readout else
                                   model.query_readout_proj(torch.cat(
                                       [h_off.mean(dim=1), zq, off_read], dim=-1)))
            ws_effect = float((logit_on - logit_off).abs().mean())
        rows.append({
            "distractor_tokens": n_distract,
            "input_tokens": len(ids),
            "overflow_events": overflow_events,
            "assignments_total": round(assignments_sum, 4),
            "occupied_first": occ_trace[0] if occ_trace else [],
            "occupied_last": occ_trace[-1] if occ_trace else [],
            "used_slots_last": int((final_occ > 1e-6).sum()),
            "max_occupied_last": round(float(final_occ.max()), 4),
            "slot_content_norm_last": [round(float(v), 6) for v in final_content.norm(dim=-1).tolist()],
            "last_read_attn": [round(v, 6) for v in (attn_last or [])],
            "logit_delta_when_workspace_read_zeroed": round(ws_effect, 8),
            "overflow_consumed_by_language_model": False,
        })
    return {
        "overflow_field_consumed_by_language_model": False,
        "overflow_dropped_at": "src/nmind/language/neural_dialogue.py:wr = self.workspace.write(ws, payload)",
        "capacity": model.workspace.capacity,
        "decay": model.workspace.decay,
        "slots": model.workspace.slots,
        "cases": rows,
        "note": "overflow 记录被丢弃；N-Cell h 仍保留全部历史输入的信息（未做可解码性断言）。",
    }


def main():
    torch.set_num_threads(1)
    tok = build_tokenizer()
    hits = infer_configs()
    # 若反推无唯一解，退到源码默认 LanguageV2Config 并把配置状态标为 UNRESOLVED
    fallback = LMConfig(
        vocab_size=4096, d_model=32, n_blocks=2, block_hidden=32,
        workspace_dim=16, workspace_slots=2, top_k=2).to_dict()
    # top_k 不改变参数量，无法从参数量反推；按任务书指明值 + 源码默认取 2。
    chosen = next((h for h in hits if h["top_k"] == 2), hits[0]) if hits else fallback
    topk_status = ("TOP_K_NOT_IDENTIFIABLE_FROM_PARAM_COUNT__USED_TASKBOOK_AND_SOURCE_DEFAULT_2"
                   if hits else "CONFIG_NOT_IDENTIFIED__USED_FALLBACK")
    cfg = LMConfig(**chosen)
    torch.manual_seed(20261124)
    model = NMindLM(cfg)
    if model.n_params() != HISTORICAL_PARENT_PARAMS:
        config_status = "UNRESOLVED_RECONSTRUCTED_FROM_DEFAULTS"
    else:
        config_status = "INFERRED_FROM_HISTORICAL_PARAM_COUNT"

    prompts = [
        "已知：小明有红色苹果，小红有绿色苹果。问：小明的苹果是什么颜色？答：",
        "已知：小明有红色苹果，小红有绿色苹果。问：小红的苹果是什么颜色？答：",
        "已知：小明有绿色苹果，小红有红色苹果。问：小明的苹果是什么颜色？答：",
    ]
    router = probe_router(model, tok, prompts)
    workspace = probe_workspace(model, tok)

    payload = {
        "status": "ARCHITECTURE_LEVEL_DIAGNOSIS",
        "parent_checkpoint_available": False,
        "parent_checkpoint_path": "experiments/language_v1/v15/candidate_v15_baseline.pt",
        "parent_checkpoint_missing_reason": "*.pt ignored by .gitignore; V171_FAIR_COMPARISON records LOST_weights_equal_parent",
        "historical_parent_params": HISTORICAL_PARENT_PARAMS,
        "config_inference_status": config_status,
        "top_k_inference_status": topk_status,
        "matching_config_count": len(hits),
        "matching_configs": hits[:20],
        "config_used": chosen,
        "model_params_actual": model.n_params(),
        "weights_source": "random_init_same_config_ARCHITECTURE_ONLY",
        "router": router,
        "workspace": workspace,
        "limitations": [
            "无法验证真实父权重的路由行为；本诊断只证明架构与前向路径性质。",
            "router 梯度检查只反传，未更新父模型；一步更新在 deepcopy 临时副本上进行。",
        ],
    }
    write_json(OUT / "V172_ROUTER_PROBE.json", payload)
    write_json(OUT / "V172_WORKSPACE_PROBE.json", workspace)
    arch = {
        "status": "ARCHITECTURE_AUDIT_DONE",
        "config": chosen,
        "config_status": config_status,
        "has_sparse_space": router["has_sparse_selection_space"],
        "router_findings": {k: router[k] for k in (
            "top_k_equals_n_blocks", "has_sparse_selection_space", "activation_ratio",
            "router_score_std_across_tokens", "distinct_selected_set_count",
            "router_score_change_changes_forward_logits", "swap_selected_order_changes_workspace",
            "router_has_nonzero_gradient", "temporary_copy_one_step_route_changed")},
        "workspace_findings": {"overflow_dropped": True,
                               "overflow_events": [c["overflow_events"] for c in workspace["cases"]]},
        "parent_checkpoint_available": False,
    }
    write_json(OUT / "V172_ARCHITECTURE_AUDIT.json", arch)
    print(json.dumps({"config_status": config_status, "config": chosen,
                      "sparse": router["has_sparse_selection_space"],
                      "grad": router["router_has_nonzero_gradient"],
                      "workspace_overflow": [c["overflow_events"] for c in workspace["cases"]]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
