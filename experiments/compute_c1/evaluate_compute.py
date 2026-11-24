"""C1.0 评测：自由生成 + 完整答案条件概率 + Router/Workspace 审计。

- 禁止 teacher forcing 代替自由生成；主指标来自贪心逐步生成。
- 完整答案评分包含全部 token 与 EOS。
- 立即 EOS / 空输出单独统计。
"""
from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

from tokenizer_compute import ComputeTokenizer, EOS, PAD, BOS, UNK
from model_compute import ComputeLM


def valid_ids(tok: ComputeTokenizer) -> list[int]:
    return [EOS] + [i for i in range(4, tok.vocab_size)]


@torch.inference_mode()
def generate(model: ComputeLM, tok: ComputeTokenizer, prompt: str,
             max_new: int = 16, device=None) -> dict:
    """贪心自回归自由生成；不读任何标签。"""
    dev = device or torch.device("cpu")
    was_training = model.training
    old_load = model.use_router_load_update
    model.eval()
    model.use_router_load_update = False
    ids = tok.encode(prompt, add_bos=False)
    x = torch.tensor([ids], dtype=torch.long, device=dev)
    logits, state = model(x, collect_stats=True)
    allowed = torch.tensor(valid_ids(tok), device=dev)
    generated: list[int] = []
    finish = "max_tokens"
    t0 = time.perf_counter()
    for _ in range(max_new):
        tail = logits[0, -1].index_select(0, allowed)
        tid = int(allowed[int(torch.argmax(tail))])
        if tid == EOS:
            finish = "eos"
            break
        generated.append(tid)
        logits, state = model(torch.tensor([[tid]], dtype=torch.long, device=dev), state,
                              collect_stats=True)
    model.train(was_training)
    model.use_router_load_update = old_load
    return {"text": tok.decode(generated), "token_ids": generated, "finish_reason": finish,
            "seconds": round(time.perf_counter() - t0, 5),
            "route": model.last_route_indices, "workspace_occ": model.last_workspace_occupied}


@torch.inference_mode()
def full_answer_lp(model: ComputeLM, tok: ComputeTokenizer, prompt: str, answer: str,
                   device=None) -> float:
    """完整答案条件 logprob（含 EOS），自回归逐 token。"""
    dev = device or torch.device("cpu")
    was_training = model.training
    model.eval()
    seq = tok.encode(prompt) + tok.encode(answer) + [EOS]
    x = torch.tensor([seq[:-1]], dtype=torch.long, device=dev)
    logits, _ = model(x)
    lp = torch.log_softmax(logits[0], -1)
    target = torch.tensor(seq[1:], dtype=torch.long, device=dev)
    total = float(lp[torch.arange(len(seq)-1, device=dev), target].sum())
    model.train(was_training)
    return total


def evaluate(model, tok, rows, *, limit=0, free_n=0, device=None) -> dict:
    """主评测：自由生成。limit=0 表示全部；free_n 限制生成样本数（0=全部）。"""
    dev = device or torch.device("cpu")
    if limit:
        rows = rows[:limit]
    per_task = defaultdict(lambda: {"n": 0, "free_ok": 0, "immediate_eos": 0,
                                    "empty": 0, "legal": 0, "seconds": 0.0})
    details = []
    gen_rows = rows if not free_n else rows[:free_n]
    for r in gen_rows:
        g = generate(model, tok, r["prompt"], device=dev)
        ok = g["text"].strip() == r["answer"].strip()
        d = per_task[r["task"]]
        d["n"] += 1
        d["free_ok"] += int(ok)
        d["immediate_eos"] += int(g["finish_reason"] == "eos" and not g["text"].strip())
        d["empty"] += int(not g["text"].strip())
        d["legal"] += int(g["finish_reason"] in ("eos", "max_tokens"))
        d["seconds"] += g["seconds"]
        details.append({"task": r["task"], "prompt": r["prompt"], "expected": r["answer"],
                        "predicted": g["text"], "ok": ok, "finish": g["finish_reason"],
                        "tokens": g["token_ids"]})
    return {
        "per_task": {t: {"n": d["n"],
                         "free_generation_accuracy": round(d["free_ok"] / max(1, d["n"]), 4),
                         "immediate_eos_rate": round(d["immediate_eos"] / max(1, d["n"]), 4),
                         "empty_rate": round(d["empty"] / max(1, d["n"]), 4),
                         "utf8_legal_rate": round(d["legal"] / max(1, d["n"]), 4),
                         "avg_infer_seconds": round(d["seconds"] / max(1, d["n"]), 5)}
                     for t, d in per_task.items()},
        "overall_accuracy": round(sum(d["free_ok"] for d in per_task.values()) /
                                  max(1, sum(d["n"] for d in per_task.values())), 4),
        "details": details,
    }


def evaluate_counterfactual(model, tok, pairs, *, device=None) -> dict:
    """反事实成对：同一 key 的 orig/swap 两条都答对才算一对通过。"""
    dev = device or torch.device("cpu")
    ok = 0
    detail = []
    for p in pairs[:500]:
        g1 = generate(model, tok, p["orig"]["prompt"], device=dev)
        g2 = generate(model, tok, p["swap"]["prompt"], device=dev)
        pair_ok = (g1["text"].strip() == p["orig"]["answer"].strip() and
                   g2["text"].strip() == p["swap"]["answer"].strip())
        ok += int(pair_ok)
        detail.append({"key": p["key"], "orig_expected": p["orig"]["answer"],
                       "orig_got": g1["text"], "swap_expected": p["swap"]["answer"],
                       "swap_got": g2["text"], "pair_ok": pair_ok})
    return {"pairs": len(detail), "paired_accuracy": round(ok / max(1, len(detail)), 4),
            "details": detail}


def router_audit(model, tok, rows, *, device=None) -> dict:
    """冻结评测：路由实际选择空间、评分变化影响、块激活率、Workspace 占用。"""
    dev = device or torch.device("cpu")
    import copy
    was_training = model.training
    old_load = model.use_router_load_update
    model.eval(); model.use_router_load_update = False
    sets = set()
    load = Counter()
    occ = []
    score_vectors = []
    for r in rows[:120]:
        ids = tok.encode(r["prompt"])
        st = model.initial_state(1, dev)
        err = torch.full((1, 1), 0.5, device=dev)
        unc = torch.full((1, 1), 0.5, device=dev)
        with torch.inference_mode():
            for tid in ids:
                z = model.tok_emb(torch.tensor([tid], device=dev))
                w = model.workspace.read(st["ws"], z).value
                route = model.router.route(z, st["h"], w, err, unc, k=model.top_k)
                sets.add(tuple(sorted(route.indices[0].tolist())))
                for b in route.indices[0].tolist():
                    load[b] += 1
                score_vectors.append(route.scores[0].detach())
                h, _, _, _, _ = model.blocks.forward_sparse(z, st["h"], w, route.indices, weights=None)
                ri = torch.arange(1, device=dev)[:, None].expand(-1, model.top_k).reshape(-1)
                payload = model.block_to_ws(h[ri, route.indices.reshape(-1)]).view(1, model.top_k, -1)
                wr = model.workspace.write(st["ws"], payload)
                occ.append(float(wr.occupied[0].mean()))
                st = {"h": h, "ws": {"content": wr.state["content"],
                                     "occupied": wr.occupied, "utility": wr.utility}}
    # 改评分后前向是否变化
    perturb = copy.deepcopy(model)
    with torch.no_grad():
        perturb.router.score_mlp[2].weight.mul_(-1.5)
    x = torch.tensor([tok.encode(rows[0]["prompt"])], device=dev)
    with torch.inference_mode():
        base, _ = model(x)
        changed, _ = perturb(x)
    delta = float((base - changed).abs().mean())
    model.train(was_training); model.use_router_load_update = old_load
    total = sum(load.values())
    return {
        "n_blocks": model.cfg.n_blocks, "top_k": model.cfg.top_k,
        "has_sparse_space": model.cfg.top_k < model.cfg.n_blocks,
        "activation_ratio": round(model.cfg.top_k / model.cfg.n_blocks, 4),
        "distinct_selected_sets": [list(s) for s in sorted(sets)],
        "distinct_set_count": len(sets),
        "router_load_distribution": {str(k): round(v / max(1, total), 4) for k, v in sorted(load.items())},
        "router_score_change_changes_forward": delta > 1e-8,
        "forward_delta": round(delta, 8),
        "workspace_mean_occupied": round(sum(occ) / max(1, len(occ)), 4),
        "workspace_occupied_first": round(occ[0], 4) if occ else None,
        "workspace_occupied_last": round(occ[-1], 4) if occ else None,
    }


def router_gradient_check(model, tok, rows, device=None) -> dict:
    dev = device or torch.device("cpu")
    was_training = model.training
    model.train(); model.use_router_load_update = False
    r = rows[0]
    seq = tok.encode(r["prompt"]) + tok.encode(r["answer"]) + [EOS]
    x = torch.tensor([seq[:-1]], dtype=torch.long, device=dev)
    logits, _ = model(x)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                             torch.tensor(seq[1:], device=dev))
    model.zero_grad(set_to_none=True)
    loss.backward()
    grads = {n: (None if p.grad is None else round(float(p.grad.abs().sum()), 8))
             for n, p in model.named_parameters() if n.startswith("router.")}
    model.zero_grad(set_to_none=True)
    model.train(was_training)
    return {"router_grad_abs_sum": grads,
            "has_nonzero_router_grad": any(v and v > 0 for v in grads.values())}
