"""隐藏状态与输出映射探针（Gate-B 004 第三阶段，无训练诊断）。

* trace_forward：逐行复刻 NMindLM.forward 的单条序列计算并收集中间量——
  N-Cell 各块状态 / Workspace 槽内容与占用 / block_to_ws 输出 / Router 选择 /
  head 输入向量（h.mean(dim=1)）与末位 logits。忠实性由断言保证：trace 的
  logits 必须与原 forward 逐位一致（torch.equal），不一致直接抛错。
* answer_logits_ranking：正确答案每个字符 token 的 logit 与全词表排名 + Top-5，
  EOS/高频 token 偏置一并记录（无训练、无标签泄漏——只读模型输出分布）。
* LinearProbe：torch 闭式 ridge 线性分类（多类 one-hot 最小二乘 + argmax），
  独立 train/test 划分。探针监督结果只作表示诊断（"表示里能不能线性读出
  实体—位置"），绝不计入 N-MIND 自主语言能力成绩——探针是有监督的外部读出，
  与自主生成的 EM 是两种性质完全不同的证据。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .neural_dialogue import LMState, NMindLM
from .tokenizer import CharTokenizer, EOS


@dataclass
class TokenTrace:
    step: int
    token: str
    route_indices: list[int]
    route_weights: list[float]
    h_mean: list[float]                    # head 输入向量（h.mean(dim=1)）
    b2w_payload_norm: float                # block_to_ws 输出向量 L2
    ws_occupied: list[float]               # 各槽占用度
    read_attn_entropy: float               # workspace.read 注意力熵
    logit_top: list[tuple[str, float]]     # top-5 (token, logit)
    eos_logit: float
    target_token_info: dict[str, Any] | None = None   # 答案字符的 logit/rank（可选）


@dataclass
class TraceResult:
    prompt: str
    tokens_trace: list[TokenTrace] = field(default_factory=list)
    final_ws_content_norm: list[float] = field(default_factory=list)
    final_block_states_norm: list[float] = field(default_factory=list)
    logits_match_original: bool = False    # trace 与原 forward 的逐位一致性断言


def trace_forward(model: NMindLM, tok: CharTokenizer, prompt: str, *,
                  target: str | None = None) -> TraceResult:
    """复刻 forward 逐步收集中间量；结尾用原 forward 校验 logits 逐位一致。"""
    model.eval()
    ids = tok.encode(prompt)
    dev = next(model.parameters()).device
    st: LMState = model.initial_state(1, dev)
    err = torch.full((1, 1), 0.5, device=dev)
    unc = torch.full((1, 1), 0.5, device=dev)
    rows = torch.arange(1, device=dev)[:, None].expand(-1, model.top_k).reshape(-1)
    trace = TraceResult(prompt=prompt)
    tid_seq: list[int] = []
    with torch.no_grad():
        for step, tid in enumerate(ids):
            tid_seq.append(tid)
            z = model.tok_emb(torch.tensor([tid], dtype=torch.long, device=dev))
            w_read = model.workspace.read(st.ws, z)
            route = model.router.route(z, st.h, w_read.value, err, unc, k=model.top_k)
            h_new, _g, _m, _p, _n = model.blocks.forward_sparse(z, st.h, w_read.value,
                                                                 route.indices)
            sel = route.indices.reshape(-1)
            payload = model.block_to_ws(h_new[rows, sel]).view(1, model.top_k, -1)
            wr = model.workspace.write(st.ws, payload)
            ws = {"content": wr.state["content"], "occupied": wr.occupied,
                   "utility": wr.utility}
            head_in = h_new.mean(dim=1)                     # [1, block_hidden]
            logits = model.head(head_in)
            last = logits[0]                                # [V]：本 token 位置的输出分布
            topv = torch.topk(last, 5)
            top = [(tok.id2tok.get(int(i), "<unk>"), round(float(v), 4))
                   for v, i in zip(topv.values, topv.indices)]
            attn = w_read.attn[0]
            attn_entropy = float(-(attn * attn.clamp_min(1e-12).log()).sum()) if attn.numel() else 0.0
            tinfo = None
            if target is not None and step >= len(ids) - 1:
                # 只在 prompt 的末位（即将生成答案的位置）记录答案 token 排序
                tinfo = _answer_ranking(last, tok, target)
            trace.tokens_trace.append(TokenTrace(
                step=step, token=tok.id2tok.get(tid, "<unk>"),
                route_indices=[int(i) for i in route.indices[0]],
                route_weights=[round(float(w), 4) for w in route.weights[0]],
                h_mean=[round(float(v), 5) for v in head_in[0]],
                b2w_payload_norm=round(float(payload.norm()), 4),
                ws_occupied=[round(float(v), 4) for v in wr.occupied[0]],
                read_attn_entropy=round(attn_entropy, 4),
                logit_top=top, eos_logit=round(float(last[EOS]), 4),
                target_token_info=tinfo))
            st = LMState(h=h_new, ws=ws)
        trace.final_ws_content_norm = [round(float(v), 4) for v in st.ws["content"][0].norm(dim=-1)]
        trace.final_block_states_norm = [round(float(v), 4) for v in st.h[0].norm(dim=-1)]
        # 忠实性断言：同一 prompt 走原 forward，末位 logits 必须逐位一致
        orig_logits, _orig_state = model(torch.tensor([ids], dtype=torch.long, device=dev))
        trace.logits_match_original = bool(torch.equal(orig_logits[0, -1], logits[0]))
    if not trace.logits_match_original:
        raise RuntimeError("trace_forward 与原 forward 不一致：诊断复刻不忠实，禁止使用")
    return trace


def _answer_ranking(last_logits: torch.Tensor, tok: CharTokenizer,
                    target: str) -> dict[str, Any]:
    """正确答案首字符的 logit 与全词表排名 + Top-5（生成即将开始的位置）。"""
    out: dict[str, Any] = {"target_first_char": None, "rank": None, "logit": None,
                            "prob": None, "top5": None, "eos_minus_top": None}
    chars = [c for c in target if c in tok.vocab]
    if not chars:
        out["target_first_char"] = None
        return out
    tid = tok.vocab[chars[0]]
    probs = torch.softmax(last_logits, dim=-1)
    rank = int((last_logits > last_logits[tid]).sum().item()) + 1
    out.update({"target_first_char": chars[0], "rank": rank,
                 "logit": round(float(last_logits[tid]), 4),
                 "prob": round(float(probs[tid]), 6),
                 "top5": [(tok.id2tok.get(int(i), "<unk>"), round(float(v), 4))
                           for v, i in zip(*torch.topk(last_logits, 5))],
                 "eos_minus_top": round(float(last_logits[EOS] - last_logits.max()), 4)})
    return out


# ---------------------------------------------------------------- 线性探针

def _ridge_fit(X: torch.Tensor, Y: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
    """多类 one-hot ridge 闭式解：W = (X^T X + λI)^-1 X^T Y；预测取 argmax。"""
    d = X.shape[1]
    A = X.T @ X + lam * torch.eye(d)
    return torch.linalg.solve(A, X.T @ Y)


def linear_probe(features: list[list[float]], labels: list[int], *,
                 train_frac: float = 0.6, seed: int = 20261701,
                 n_classes: int | None = None) -> dict[str, Any]:
    """闭式 ridge 线性探针：独立 train/test 划分，返回 test 准确率与多数类基线。

    探针监督只作表示诊断（可解码性），不计入 N-MIND 自主语言能力成绩。"""
    g = torch.Generator().manual_seed(seed)
    n = len(labels)
    idx = torch.randperm(n, generator=g).tolist()
    n_train = max(2, int(n * train_frac))
    tr, te = idx[:n_train], idx[n_train:]
    X = torch.tensor(features, dtype=torch.float64)
    y = torch.tensor(labels)
    n_classes = n_classes or int(y.max().item()) + 1
    Y = torch.zeros(n, n_classes, dtype=torch.float64)
    Y[torch.arange(n), y] = 1.0
    W = _ridge_fit(X[tr], Y[tr])
    pred = (X[te] @ W).argmax(dim=-1)
    truth = y[te]
    acc = float((pred == truth).float().mean())
    counts = torch.bincount(truth, minlength=n_classes).float()
    baseline = float(counts.max() / counts.sum()) if te else None
    return {"n_train": len(tr), "n_test": len(te), "test_accuracy": round(acc, 4),
             "majority_baseline": round(baseline, 4) if baseline is not None else None,
             "note": "探针监督=外部线性读出，只作表示诊断，不计入自主语言能力成绩"}


def extract_head_input(model: NMindLM, tok: CharTokenizer, prompt: str) -> list[float]:
    """head 输入向量（h.mean(dim=1)，prompt 末位）——探针特征源之一。"""
    model.eval()
    with torch.no_grad():
        _logits, st = model(torch.tensor([tok.encode(prompt)], dtype=torch.long))
    return [round(float(v), 6) for v in st.h.mean(dim=1)[0]]


def extract_ws_content(model: NMindLM, tok: CharTokenizer, prompt: str) -> list[float]:
    """Workspace 槽内容拼接（slots×d_w，prompt 末位）——探针特征源之二。"""
    model.eval()
    with torch.no_grad():
        _logits, st = model(torch.tensor([tok.encode(prompt)], dtype=torch.long))
    return [round(float(v), 6) for v in st.ws["content"][0].flatten()]


def counterfactual_representation_shift(model: NMindLM, tok: CharTokenizer,
                                          prompt_a: str, prompt_b: str) -> dict[str, float]:
    """反事实表示变化：同上下文不同查询实体，末位 head 输入向量的余弦/L2 距离。"""
    a = torch.tensor(extract_head_input(model, tok, prompt_a))
    b = torch.tensor(extract_head_input(model, tok, prompt_b))
    cos = float(torch.nn.functional.cosine_similarity(a, b, dim=0))
    return {"cosine": round(cos, 4), "l2": round(float((a - b).norm()), 4)}
