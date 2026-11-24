"""M1 统一评测协议：自由生成为主，固定面板，同任务同样本同顺序。"""
from __future__ import annotations

import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from dataset_m1 import M1Tokenizer, EOS, PAD
from curriculum_m1 import PANEL_TASKS

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
SEED = 20261124
W_FIRST, W_CONTENT, W_EOS = 2.0, 1.0, 0.25


def load_rows(name):
    return [json.loads(l) for l in (DATA / f"m1_{name}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


def build_fixed_panel(rows, per_task=30, seed=SEED):
    rng = random.Random(seed)
    panel = []
    for task in PANEL_TASKS:
        task_rows = [r for r in rows if r["task"] == task]
        rng.shuffle(task_rows)
        panel.extend(task_rows[:per_task])
    return panel


def make_batch(tok, rows):
    seqs = [tok.encode(r["prompt"]) + tok.encode(r["answer"]) + [EOS] for r in rows]
    n = max(len(s) for s in seqs)
    B = len(rows)
    x = torch.full((B, n), PAD, dtype=torch.long)
    valid = torch.zeros(B, n, dtype=torch.bool)
    weights = torch.zeros(B, n - 1, dtype=torch.float32)
    for j, (r, s) in enumerate(zip(rows, seqs)):
        x[j, :len(s)] = torch.tensor(s)
        valid[j, :len(s)] = True
        p_len = len(tok.encode(r["prompt"]))
        a_len = len(tok.encode(r["answer"]))
        for k in range(a_len):
            weights[j, p_len - 1 + k] = W_FIRST if k == 0 else W_CONTENT
        weights[j, p_len + a_len - 1] = W_EOS
    return x, valid, weights


def grouped_loss(logits, target, weights):
    B = weights.size(0)
    token_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                 target.reshape(-1), reduction="none", ignore_index=PAD)
    w = weights.reshape(-1)
    per = (token_loss * w).reshape(B, -1).sum(1)
    denom = w.reshape(B, -1).sum(1).clamp_min(1e-6)
    return (per / denom).mean()


def allowed_ids(tok):
    return [EOS] + list(range(4, tok.vocab_size))


@torch.inference_mode()
def generate(model, tok, prompt, max_new=12):
    was_train = model.training
    old_load = getattr(model, "use_router_load_update", False)
    old_grad = getattr(model, "router_gradient_path_enabled", False)
    model.eval()
    model.use_router_load_update = False
    model.router_gradient_path_enabled = False
    ids = tok.encode(prompt)
    logits, state = model(torch.tensor([ids]))
    allowed = torch.tensor(allowed_ids(tok))
    out = []
    finish = "max_tokens"
    t0 = time.perf_counter()
    for _ in range(max_new):
        tail = logits[0, -1].index_select(0, allowed)
        tid = int(allowed[int(torch.argmax(tail))])
        if tid == EOS:
            finish = "eos"
            break
        out.append(tid)
        logits, state = model(torch.tensor([[tid]]), state)
    seconds = time.perf_counter() - t0
    model.train(was_train)
    model.use_router_load_update = old_load
    model.router_gradient_path_enabled = old_grad
    return {"text": tok.decode(out), "token_ids": out, "finish_reason": finish,
            "seconds": seconds}


@torch.inference_mode()
def full_answer_lp(model, tok, prompt, answer):
    """log P(answer + EOS | prompt)，不含 prompt 自身概率。"""
    was_train = model.training
    model.eval()
    p = tok.encode(prompt)
    a = tok.encode(answer)
    seq = p + a + [EOS]
    logits, _ = model(torch.tensor([seq[:-1]]))
    logp = torch.log_softmax(logits[0], -1)
    start = len(p) - 1
    total = 0.0
    for j in range(len(a) + 1):
        pos = start + j
        total += float(logp[pos, seq[pos + 1]])
    model.train(was_train)
    return total


def evaluate_panel(model, tok, panel, collect_details=False):
    per = defaultdict(lambda: {"n": 0, "ok": 0, "eos0": 0, "empty": 0})
    details = []
    for r in panel:
        g = generate(model, tok, r["prompt"])
        ok = g["text"].strip() == str(r["answer"]).strip()
        d = per[r["task"]]
        d["n"] += 1; d["ok"] += int(ok)
        d["eos0"] += int(g["finish_reason"] == "eos" and not g["text"].strip())
        d["empty"] += int(not g["text"].strip())
        if collect_details:
            details.append({"task": r["task"], "prompt": r["prompt"],
                            "expected": r["answer"], "predicted": g["text"],
                            "ok": ok, "finish": g["finish_reason"], "tokens": g["token_ids"]})
    per_task = {t: {"n": d["n"], "ok": d["ok"],
                    "accuracy": round(d["ok"] / max(1, d["n"]), 4),
                    "immediate_eos_rate": round(d["eos0"] / max(1, d["n"]), 4),
                    "empty_rate": round(d["empty"] / max(1, d["n"]), 4)}
                for t, d in sorted(per.items())}
    n = sum(d["n"] for d in per.values())
    ok = sum(d["ok"] for d in per.values())
    eos0 = sum(d["eos0"] for d in per.values())
    out = {"per_task": per_task,
           "macro_accuracy": round(sum(d["accuracy"] for d in per_task.values()) /
                                   max(1, len(per_task)), 4),
           "micro_accuracy": round(ok / max(1, n), 4),
           "immediate_eos_rate": round(eos0 / max(1, n), 4),
           "total_n": n, "total_ok": ok}
    if collect_details:
        out["details"] = details
    return out


def evaluate_rows(model, tok, rows):
    per = defaultdict(lambda: {"n": 0, "ok": 0, "eos0": 0})
    for r in rows:
        g = generate(model, tok, r["prompt"])
        d = per[r["task"]]
        d["n"] += 1
        d["ok"] += int(g["text"].strip() == str(r["answer"]).strip())
        d["eos0"] += int(g["finish_reason"] == "eos" and not g["text"].strip())
    return {t: {"n": d["n"], "ok": d["ok"],
                "accuracy": round(d["ok"] / max(1, d["n"]), 4),
                "immediate_eos_rate": round(d["eos0"] / max(1, d["n"]), 4)}
            for t, d in sorted(per.items())}


def evaluate_counterfactual(model, tok, limit=400):
    path = DATA / "m1_counterfactual.jsonl"
    pairs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()][:limit]
    oks = []
    for p in pairs:
        sw = [q for q in p["queries"] if q["task"] == "B_swap"][:2]
        if len(sw) < 2:
            continue
        vals = [generate(model, tok, q["prompt"])["text"].strip() == str(q["answer"]).strip()
                for q in sw]
        oks.append(all(vals))
    n = len(oks); ok = sum(oks)
    return {"pairs": n, "ok": ok, "paired_accuracy": round(ok / max(1, n), 4)}


def shortcut_baselines(panel, train_rows):
    counts = Counter(str(r["answer"]) for r in train_rows)
    fixed = counts.most_common(1)[0][0]
    def rate(key):
        vals = [r for r in panel if str(r.get("meta", {}).get(key, ""))]
        return {"n": len(vals),
                "ok": sum(1 for r in vals if str(r["meta"][key]) == str(r["answer"])),
                "accuracy": round(sum(1 for r in vals if str(r["meta"][key]) == str(r["answer"])) /
                                  max(1, len(vals)), 4)}
    return {"fixed_answer": rate("answer") if False else {
                "n": len(panel),
                "ok": sum(1 for r in panel if str(r["answer"]) == fixed),
                "accuracy": round(sum(1 for r in panel if str(r["answer"]) == fixed) / max(1, len(panel)), 4)},
            "first_digit": rate("first_digit"), "last_digit": rate("last_digit"),
            "nearest_assign": rate("nearest_assign"),
            "random_guess": {"n": len(panel), "ok": None,
                             "accuracy": round(1.0 / max(1, len({str(r["answer"]) for r in panel})), 4)}}
