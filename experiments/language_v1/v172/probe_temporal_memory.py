"""V1.7.2 P1：N-Cell 在中文人物-属性绑定上的时序信息保持探针。

冻结主模型（无权重更新），按任务书案例 A/B/C × 干扰 0/8/16/32/64 token
逐 token 记录：Block 隐藏状态、块平均状态、Workspace 读出、LM Head 输出、
正确/错误答案条件概率差、状态相似度、梯度沿时间回传强度。

注意：h.mean(dim=1) 是**块维度**平均，不是时间维度；本脚本同时保留原始
Block 状态，用可区分性比较定位信息开始丢失的层。
独立线性探针只作诊断，参数更新单独计数，不计入主模型能力。
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nmind.language.neural_dialogue import LMConfig, NMindLM
from v172_protocol import (build_tokenizer, encode_autoregressive, full_answer_conditional,
                           valid_vocab, write_json)

OUT = Path(__file__).resolve().parent

CASES = {
    "A": {"fact": "已知：小明有红色苹果，小红有绿色苹果。", "who": "小明", "correct": "红色", "wrong": "绿色"},
    "B": {"fact": "已知：小明有红色苹果，小红有绿色苹果。", "who": "小红", "correct": "绿色", "wrong": "红色"},
    "C": {"fact": "已知：小明有绿色苹果，小红有红色苹果。", "who": "小明", "correct": "绿色", "wrong": "红色"},
}
DISTRACTOR = "今天天气不错，街上的车很多，店门已经打开。"
DISTANCES = (0, 8, 16, 32, 64)


def build_prompt(case: dict, n_distract: int, tok) -> tuple[str, int]:
    q = f"问：{case['who']}的苹果是什么颜色？\n答："
    if n_distract <= 0:
        return case["fact"] + "\n" + q, 0
    text = case["fact"]
    used = 0
    while used < n_distract:
        text += DISTRACTOR
        used += len(tok.encode(DISTRACTOR))
    return text + "\n" + q, used


def trace_forward(model: NMindLM, tok, prompt: str):
    """逐 token 捕获四层表示。"""
    ids = tok.encode(prompt)
    st = model.initial_state(1, torch.device("cpu"))
    rows = []
    err = torch.full((1, 1), 0.5)
    unc = torch.full((1, 1), 0.5)
    with torch.inference_mode():
        for t, tid in enumerate(ids):
            z = model.tok_emb(torch.tensor([tid]))
            w_read = model.workspace.read(st.ws, z)
            r = model.router.route(z, st.h, w_read.value, err, unc, k=model.top_k)
            h, _, _, _, _ = model.blocks.forward_sparse(z, st.h, w_read.value, r.indices, weights=None)
            rows_idx = torch.arange(1)[:, None].expand(-1, model.top_k).reshape(-1)
            payload = model.block_to_ws(h[rows_idx, r.indices.reshape(-1)]).view(1, model.top_k, -1)
            wr = model.workspace.write(st.ws, payload)
            h_mean = h.mean(dim=1)                       # 块维度平均（不是时间维）
            if model.query_readout:
                decoded = model.query_readout_proj(torch.cat([h_mean, z, w_read.value], dim=-1))
            else:
                decoded = h_mean
            logits = model.head(decoded)
            rows.append({
                "t": t,
                "h_blocks": h[0].detach().clone(),
                "h_mean": h_mean[0].detach().clone(),
                "ws_read": w_read.value[0].detach().clone(),
                "ws_attn": w_read.attn[0].detach().clone(),
                "occupied": wr.occupied[0].detach().clone(),
                "logits": logits[0].detach().clone(),
                "route": r.indices[0].tolist(),
            })
            st = type(st)(h=h, ws={"content": wr.state["content"],
                                   "occupied": wr.occupied, "utility": wr.utility})
    return rows


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a.reshape(-1), b.reshape(-1), dim=0))


def layer_distinguishability(rows_a, rows_b, keys=("h_blocks", "h_mean", "ws_read", "logits")) -> dict:
    if not rows_a or not rows_b:
        return {}
    out = {}
    for key in keys:
        a, b = rows_a[-1][key], rows_b[-1][key]
        out[key] = {
            "cosine": round(cos(a, b), 6),
            "l2": round(float((a.reshape(-1) - b.reshape(-1)).norm()), 6),
        }
    # 逐 Block 的平均可区分度（定位是全部块还是个别块保留信息）
    per_block = []
    for i in range(rows_a[-1]["h_blocks"].shape[0]):
        per_block.append(round(cos(rows_a[-1]["h_blocks"][i], rows_b[-1]["h_blocks"][i]), 6))
    out["per_block_cosine"] = per_block
    return out


def temporal_gradient(model: NMindLM, tok, prompt: str, answer: str):
    """对答案 first token CE 反传，记录梯度沿时间步的衰减。"""
    ts = encode_autoregressive(tok, prompt, answer)
    x = torch.tensor([list(ts.sequence[:-1])], dtype=torch.long)
    y = torch.tensor(list(ts.sequence[1:]), dtype=torch.long)
    model.eval()
    model.use_router_load_update = False
    model.zero_grad(set_to_none=True)
    logits, _ = model(x)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y)
    loss.backward()
    norms = []
    for t in range(x.shape[1]):
        total = 0.0
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            # 用 token embedding 的梯度幅度作为该时间步接收监督的代理
            if name == "tok_emb.weight":
                total += float(p.grad[y[t]].abs().sum())
        norms.append(round(total, 8))
    model.zero_grad(set_to_none=True)
    return {"loss": round(float(loss.detach()), 6), "embedding_grad_abs_by_time": norms}


def independent_linear_probe(reps_a, reps_b, labels) -> dict:
    """闭式 ridge 线性探针（独立参数，不计入主模型能力）。"""
    X = torch.stack(reps_a + reps_b, dim=0).reshape(len(reps_a) + len(reps_b), -1).double()
    X = torch.cat([X, torch.ones(X.shape[0], 1, dtype=X.dtype)], dim=1)
    y = torch.tensor(labels, dtype=X.dtype)
    ridge = 1e-3 * torch.eye(X.shape[1], dtype=X.dtype)
    try:
        w = torch.linalg.solve(X.T @ X + ridge, X.T @ y)
        pred = (X @ w > 0).long()
        acc = float((pred == y.long()).float().mean())
    except torch.linalg.LinAlgError:
        acc = 0.0
    return {"probe_parameters": int(X.shape[1]), "probe_optimizer_steps": 0,
            "probe_train_accuracy": round(acc, 4),
            "note": "独立闭式 ridge 探针；主模型参数未更新，不代表主模型能力。"}


def main():
    torch.set_num_threads(1)
    tok = build_tokenizer()
    # 与 P0 同配置（父权重缺失时的架构级诊断）
    cfg_path = OUT / "V172_ROUTER_PROBE.json"
    if cfg_path.exists():
        cfg_dict = json.loads(cfg_path.read_text(encoding="utf-8"))["router"]["config"]
    else:
        cfg_dict = dict(vocab_size=260, d_model=64, n_blocks=4, block_hidden=64,
                        workspace_dim=32, workspace_slots=4, top_k=2, query_readout=False)
    cfg = LMConfig(**cfg_dict)
    torch.manual_seed(20261124)
    model = NMindLM(cfg)
    model.eval()
    model.use_router_load_update = False
    valid, _ = valid_vocab(tok)

    cases = []
    reps_a, reps_b, labels = [], [], []
    for name, case in CASES.items():
        for n in DISTANCES:
            prompt, used = build_prompt(case, n, tok)
            tr_a = trace_forward(model, tok, prompt)
            # 对照：把查询人物换成另一个人，观察表示差异（同事实、同干扰长度）
            other = "小红" if case["who"] == "小明" else "小明"
            prompt_b = prompt.replace(f"问：{case['who']}", f"问：{other}")
            tr_b = trace_forward(model, tok, prompt_b)

            p_correct = full_answer_conditional(model, tok, prompt, case["correct"], valid=valid)
            p_wrong = full_answer_conditional(model, tok, prompt, case["wrong"], valid=valid)
            fact_boundary = min(len(tok.encode(case["fact"])), len(prompt) - 1)
            sim_q = cos(tr_a[-1]["h_mean"], tr_a[fact_boundary]["h_mean"]) if fact_boundary < len(tr_a) else None
            dist = layer_distinguishability(tr_a, tr_b)
            grad = temporal_gradient(model, tok, prompt, case["correct"])

            reps_a.append(tr_a[-1]["h_mean"].double())
            reps_b.append(tr_b[-1]["h_mean"].double())
            labels.extend([1, 0])
            cases.append({
                "case": name,
                "expected_answer": case["correct"],
                "distractor_tokens_requested": n,
                "distractor_tokens_actual": used,
                "input_tokens": len(tok.encode(prompt)),
                "correct_answer_logprob": round(p_correct, 6),
                "wrong_answer_logprob": round(p_wrong, 6),
                "correct_minus_wrong": round(p_correct - p_wrong, 6),
                "fact_to_query_h_mean_cosine": None if sim_q is None else round(sim_q, 6),
                "query_distinguishability": dist,
                "temporal_gradient": grad,
                "last_step_router": tr_a[-1]["route"],
            })

    probe = independent_linear_probe(reps_a, reps_b, labels)
    first_loss_layers = {}
    for key in ("h_blocks", "h_mean", "ws_read", "logits"):
        vals = [abs(c["query_distinguishability"][key]["cosine"]) for c in cases if key in c["query_distinguishability"]]
        first_loss_layers[key] = round(sum(vals) / max(1, len(vals)), 6)

    payload = {
        "status": "NCELL_MEMORY_PROBE_DONE",
        "weights_source": "random_init_same_config_ARCHITECTURE_ONLY",
        "parent_checkpoint_available": False,
        "config": cfg_dict,
        "cases": cases,
        "mean_query_distinguishability_by_layer": first_loss_layers,
        "independent_linear_probe": probe,
        "main_model_optimizer_steps": 0,
        "probe_optimizer_steps": probe["probe_optimizer_steps"],
        "note": "h.mean(dim=1) 为块维平均。余弦越低表示问小明/问小红的表示越可区分。",
    }
    write_json(OUT / "V172_NCELL_MEMORY_PROBE.json", payload)
    print(json.dumps({"cases": len(cases), "probe": probe,
                      "mean_cos": first_loss_layers}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
