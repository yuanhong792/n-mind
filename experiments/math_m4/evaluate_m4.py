"""M4 评测协议：Answer-only 监督 + 严格生成评测。

- Protocol A（主实验）：prompt/SEP/PAD 权重 0，答案 token 与 EOS 权重 1，
  按样本有效监督权重归一化后对 batch 平均。
- Protocol B（仅消融）：全序列监督，权重 1（首 token 无法预测，权重 0）。
- 严格正确 = 生成内容逐字符一致 且 自然 EOS 结束。
- teacher-forced 逐位置准确率与自由生成逐位置准确率分别统计。

运行：python evaluate_m4.py（自检）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(HERE))
from tokenizer_m4 import M4Tokenizer, SEP, EOS, PAD  # noqa: E402

DATA = HERE / "data"
GEN_LIMIT = 40


def load_rows(split: str) -> list[dict]:
    p = DATA / f"m4_{split}.jsonl"
    if not p.exists():
        raise FileNotFoundError(p)
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def build_fixed_panel(rows: list[dict], per_task: int = 20) -> list[dict]:
    from collections import defaultdict
    by = defaultdict(list)
    for r in rows:
        by[r["task"]].append(r)
    out = []
    for t in sorted(by):
        out.extend(by[t][:per_task])
    return out


def make_batch(tok: M4Tokenizer, rows: list[dict], protocol: str = "A", max_len: int = 96):
    """返回 input_ids, valid, weights, targets。weights[j] 对应目标 seq[j]。"""
    seqs = []
    for r in rows:
        s, _ = tok.encode_pair(r["prompt"], r["answer"])
        if len(s) > max_len:
            raise ValueError(f"静默截断禁止：序列 {len(s)} > {max_len}, prompt={r['prompt']!r}")
        seqs.append(s)
    T = max(len(s) for s in seqs)
    B = len(seqs)
    x = torch.full((B, T), PAD, dtype=torch.long)
    valid = torch.zeros(B, T, dtype=torch.bool)
    w = torch.zeros(B, T)
    tgt = torch.full((B, T), PAD, dtype=torch.long)
    for i, s in enumerate(seqs):
        L = len(s)
        x[i, :L] = torch.tensor(s)
        valid[i, :L] = True
        tgt[i, :L] = torch.tensor(s)
        sep = s.index(SEP)
        if protocol == "A":
            for j in range(sep + 1, L):   # answer tokens + EOS
                w[i, j] = 1.0
        elif protocol == "B":
            for j in range(1, L):         # 首 token 无前驱，不监督
                w[i, j] = 1.0
        else:
            raise ValueError(protocol)
    return x, valid, w, tgt


def grouped_loss(logits, target, weights):
    """按样本归一化后对 batch 取平均（M4 要求）。"""
    V = logits.shape[-1]
    ce = torch.nn.functional.cross_entropy(
        logits.reshape(-1, V), target.reshape(-1), reduction="none").reshape(target.shape)
    den = weights.sum(dim=1).clamp_min(1.0)
    return ((ce * weights).sum(dim=1) / den).mean()


@torch.no_grad()
def generate(model, tok: M4Tokenizer, prompt: str, gen_limit=GEN_LIMIT, hard_gate=True):
    ids = tok.encode_inference(prompt)
    x = torch.tensor([ids], dtype=torch.long)
    valid = torch.ones(1, x.size(1), dtype=torch.bool)
    state = model.initial_state(1, x.device)
    kw = {"hard_gate": hard_gate} if "hard_gate" in model.forward.__code__.co_varnames else {}
    logits, state = model(x, state=state, valid_mask=valid, **kw)
    out = []
    finished = False
    for _ in range(gen_limit):
        last = logits[0, -1].clone()
        last[PAD] = float("-inf")
        last[SEP] = float("-inf")
        tid = int(last.argmax().item())
        out.append(tid)
        if tid == EOS:
            finished = True
            break
        x = torch.tensor([[tid]], dtype=torch.long)
        valid = torch.ones(1, 1, dtype=torch.bool)
        logits, state = model(x, state=state, valid_mask=valid, **kw)
    return {"text": tok.decode(out, skip_special=True), "ids": out,
            "finish_reason": "eos" if finished else "length",
            "immediate_eos": len(out) == 1 and out[0] == EOS}


def eval_one(model, tok, row, hard_gate=True):
    g = generate(model, tok, row["prompt"], hard_gate=hard_gate)
    got, exp = g["text"], row["answer"]
    return {"task": row["task"], "prompt": row["prompt"], "expected": exp,
            "predicted": got, "content_ok": got == exp,
            "full_ok": got == exp and g["finish_reason"] == "eos",
            "finish_reason": g["finish_reason"], "immediate_eos": g["immediate_eos"],
            "overlong": g["finish_reason"] == "length"}


def eval_rows(model, tok, rows, hard_gate=True):
    per, all_results = {}, []
    for r in rows:
        e = eval_one(model, tok, r, hard_gate)
        all_results.append(e)
        d = per.setdefault(e["task"], {"n": 0, "ok": 0, "full_ok": 0,
                                        "immediate_eos": 0, "overlong": 0})
        d["n"] += 1
        d["ok"] += int(e["content_ok"])
        d["full_ok"] += int(e["full_ok"])
        d["immediate_eos"] += int(e["immediate_eos"])
        d["overlong"] += int(e["overlong"])
    n = len(all_results)
    ok = sum(e["content_ok"] for e in all_results)
    full = sum(e["full_ok"] for e in all_results)
    macro = sum(d["ok"] / d["n"] for d in per.values()) / len(per) if per else 0.0
    return {"n": n, "ok": ok, "content_accuracy": round(ok / max(1, n), 4),
            "full_answer_natural_eos": round(full / max(1, n), 4),
            "macro_accuracy": round(macro, 4),
            "immediate_eos_rate": round(sum(e["immediate_eos"] for e in all_results) / max(1, n), 4),
            "overlong_rate": round(sum(e["overlong"] for e in all_results) / max(1, n), 4),
            "per_task": per, "results": all_results}


def tf_position_accuracy(model, tok, rows, protocol="A"):
    """Teacher-forced 逐位置准确率（答案位置），与自由生成分开报告。"""
    correct = total = 0
    for r in rows:
        seq, _ = tok.encode_pair(r["prompt"], r["answer"])
        sep = seq.index(SEP)
        x = torch.tensor([seq[:-1]])
        logits, _ = model(x)
        pred = logits[0].argmax(-1).tolist()
        for j in range(sep + 1, len(seq) - 1):
            total += 1
            correct += int(pred[j - 1] == seq[j])
    return {"n": total, "ok": correct, "accuracy": round(correct / max(1, total), 4)}


def counterfactual_eval(model, tok, pairs):
    """四类分别统计；配对两条都正确才计通过。"""
    by_type = {}
    for p in pairs:
        t = p["type"]
        d = by_type.setdefault(t, {"n": 0, "ok": 0, "query_switch_n": 0, "query_switch_ok": 0})
        base_ok = all(eval_one(model, tok, b)["content_ok"] for b in p["base"])
        cf_ok = all(eval_one(model, tok, c)["content_ok"] for c in p["counterfactual"])
        d["ok"] += int(base_ok and cf_ok)
        d["n"] += 1
        if p.get("query_switch"):
            d["query_switch_ok"] += int(all(eval_one(model, tok, q)["content_ok"]
                                             for q in p["query_switch"]))
            d["query_switch_n"] += 1
    return {k: {"n": v["n"], "ok": v["ok"], "pair_rate": round(v["ok"] / max(1, v["n"]), 4),
                "query_switch_rate": round(v["query_switch_ok"] / max(1, v["query_switch_n"]), 4)}
            for k, v in by_type.items()}


def distance_eval(model, tok, rows):
    by = {}
    for r in rows:
        d = r["meta"]["distance"]
        e = eval_one(model, tok, r)
        cell = by.setdefault(str(d), {"n": 0, "ok": 0})
        cell["n"] += 1
        cell["ok"] += int(e["content_ok"])
    return {k: round(v["ok"] / v["n"], 4) for k, v in sorted(by.items(), key=lambda x: int(x[0]))}


ISO_TARGETS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]


def iso_record(curve):
    out = {}
    for tgt in ISO_TARGETS:
        reached = None
        for i, pt in enumerate(curve):
            if pt.get("macro", 0) >= tgt:
                if i + 1 < len(curve) and curve[i + 1].get("macro", 0) >= tgt:
                    reached = pt; break
                if i == len(curve) - 1:
                    reached = pt
        out[f"{int(tgt*100)}%"] = ({"status": "TARGET_NOT_REACHED"} if reached is None else
                                   {"status": "REACHED", "step": reached.get("step"),
                                    "macro": reached.get("macro")})
    return out


if __name__ == "__main__":
    tok = M4Tokenizer()
    rows = load_rows("train")[:4]
    x, valid, w, tgt = make_batch(tok, rows, "A")
    sep_ok = True
    for i, r in enumerate(rows):
        seq, _ = tok.encode_pair(r["prompt"], r["answer"])
        sep = seq.index(SEP)
        if not (w[i, :sep + 1].sum() == 0 and w[i, sep + 1:].sum() > 0 and seq[-1] == EOS):
            sep_ok = False
    print(json.dumps({"status": "SELF_TEST", "answer_only_alignment": sep_ok,
                      "batch": list(x.shape), "supervised_tokens": int(w.sum())}, ensure_ascii=False))
