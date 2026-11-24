"""C1.1 统一评测协议：固定验证面板、自由生成、完整答案条件概率。

修复 P5：full_answer_lp 只算答案+EOS，不含 prompt 自身预测概率。
修复 P10：所有模型用相同样本/任务/顺序，任务等权宏平均。
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import torch

import sys
HERE = Path(__file__).resolve().parent
C1 = HERE.parent / "compute_c1"
sys.path.insert(0, str(C1))
sys.path.insert(0, str(HERE))

from tokenizer_compute import ComputeTokenizer, EOS
from model_compute import ComputeLM, load_compute

PANEL_TASKS = ["A1_copy1", "A2_copyN", "A3_index", "A4_reverse",
               "B2_bind", "B4_bind", "B2_counter", "B5_reassign", "B6_indirect"]


def valid_ids(tok):
    return [EOS] + list(range(4, tok.vocab_size))


@torch.inference_mode()
def generate(model, tok, prompt, max_new=12, device=None):
    """贪心自回归；记录第一个生成 token 是否 EOS。"""
    dev = device or torch.device("cpu")
    was_train = model.training
    old_load = getattr(model, "use_router_load_update", False)
    model.eval(); model.use_router_load_update = False
    ids = tok.encode(prompt)
    logits, state = model(torch.tensor([ids], device=dev))
    allowed = torch.tensor(valid_ids(tok), device=dev)
    out = []
    finish = "max_tokens"
    t0 = time.perf_counter()
    for _ in range(max_new):
        tail = logits[0, -1].index_select(0, allowed)
        tid = int(allowed[int(torch.argmax(tail))])
        if tid == EOS:
            finish = "eos"; break
        out.append(tid)
        logits, state = model(torch.tensor([[tid]], device=dev), state)
    dt = time.perf_counter() - t0
    model.train(was_train); model.use_router_load_update = old_load
    return {"text": tok.decode(out), "token_ids": out, "finish_reason": finish,
            "seconds": dt}


@torch.inference_mode()
def full_answer_lp(model, tok, prompt, answer, device=None):
    """P5 修复：log P(answer_tokens + EOS | prompt_tokens)。

    只对答案区 target 计算，prompt 区域的预测不计入。
    """
    dev = device or torch.device("cpu")
    was_train = model.training
    model.eval()
    p_ids = tok.encode(prompt)
    a_ids = tok.encode(answer)
    seq = p_ids + a_ids + [EOS]
    x = torch.tensor([seq[:-1]], device=dev)
    logits, _ = model(x)
    logp = torch.log_softmax(logits[0], -1)
    # 答案第一个 token 由 prompt 最后一个位置预测；下标 = len(p_ids)-1
    start = len(p_ids) - 1
    total = 0.0
    for j in range(len(a_ids) + 1):  # +1 = EOS
        pos = start + j
        tid = seq[pos + 1]
        total += float(logp[pos, tid])
    model.train(was_train)
    return total


def build_fixed_panel(rows, per_task=60, seed=20261124):
    """P10：固定验证面板；按任务等额取样，顺序固定。"""
    import random
    rng = random.Random(seed)
    panel = []
    for task in PANEL_TASKS:
        task_rows = [r for r in rows if r["task"] == task]
        rng.shuffle(task_rows)
        panel.extend(task_rows[:per_task])
    return panel


def evaluate_panel(model, tok, panel, *, collect_details=False):
    """统一评测：返回分任务 + 宏平均 + 立即EOS + 完整答案准确率。"""
    per = defaultdict(lambda: {"n": 0, "ok": 0, "eos0": 0, "empty": 0, "lp_ok": 0})
    details = []
    for r in panel:
        g = generate(model, tok, r["prompt"])
        ok = g["text"].strip() == r["answer"].strip()
        d = per[r["task"]]
        d["n"] += 1
        d["ok"] += int(ok)
        d["eos0"] += int(g["finish_reason"] == "eos" and not g["text"].strip())
        d["empty"] += int(not g["text"].strip())
        if collect_details:
            details.append({"task": r["task"], "prompt": r["prompt"],
                            "expected": r["answer"], "predicted": g["text"],
                            "ok": ok, "finish": g["finish_reason"],
                            "tokens": g["token_ids"]})
    per_task = {t: {"n": d["n"], "accuracy": round(d["ok"] / max(1, d["n"]), 4),
                    "immediate_eos_rate": round(d["eos0"] / max(1, d["n"]), 4),
                    "empty_rate": round(d["empty"] / max(1, d["n"]), 4)}
                for t, d in per.items()}
    macro = round(sum(d["accuracy"] for d in per_task.values()) / max(1, len(per_task)), 4)
    total_n = sum(d["n"] for d in per.values())
    total_ok = sum(d["ok"] for d in per.values())
    out = {"per_task": per_task, "macro_accuracy": macro,
           "micro_accuracy": round(total_ok / max(1, total_n), 4),
           "immediate_eos_rate": round(sum(d["eos0"] for d in per.values()) / max(1, total_n), 4),
           "total_n": total_n, "total_ok": total_ok}
    if collect_details:
        out["details"] = details
    return out


def evaluate_counterfactual(model, tok, cf_path, limit=300):
    results = []
    lines = [json.loads(l) for l in Path(cf_path).read_text(encoding="utf-8").splitlines() if l.strip()]
    for p in lines[:limit]:
        binds = [q for q in p["queries"] if q["task"] == "B2_bind"]
        counters = [q for q in p["queries"] if q["task"] == "B2_counter"]
        if len(binds) < 2 or len(counters) < 2:
            continue
        # 反事实配对：两个 counter 题（swap 后不同查询）都要对
        oks = [generate(model, tok, q["prompt"])["text"].strip() == q["answer"] for q in counters]
        results.append({"key": p["key"], "pair_ok": all(oks),
                        "details": [{"expected": q["answer"],
                                     "got": generate(model, tok, q["prompt"])["text"]}
                                    for q in counters]})
    return {"pairs": len(results),
            "paired_accuracy": round(sum(r["pair_ok"] for r in results) / max(1, len(results)), 4),
            "results": results[:50]}


def evaluate_generalization(model, tok, gen_path, rename_path=None, limit=300):
    rows = [json.loads(l) for l in Path(gen_path).read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = rows[:limit]
    per = defaultdict(lambda: {"n": 0, "ok": 0})
    for r in rows:
        g = generate(model, tok, r["prompt"])
        d = per[r["task"]]
        d["n"] += 1
        d["ok"] += int(g["text"].strip() == r["answer"])
    return {t: {"n": d["n"], "accuracy": round(d["ok"] / max(1, d["n"]), 4)}
            for t, d in per.items()}
