"""M2 统一评测协议：SEP 边界自由生成主指标，固定面板，同任务同样本同顺序。

- 推理输入 = prompt_ids + [SEP]，模型从 SEP 位置开始自回归生成答案。
- 主指标：自然 EOS 的逐字符精确匹配；立即 EOS/空输出/超长单独统计。
- full_answer_lp：log P(answer+EOS | prompt+SEP)，不含 prompt/SEP 概率。
"""
from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from tokenizer_m2 import M2Tokenizer, EOS, SEP, PAD, allowed_ids, vocab_sha
from curriculum_m2 import PANEL_TASKS

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
SEED = 20261201
W_FIRST, W_CONTENT, W_EOS = 2.0, 1.0, 0.25


def load_rows(name):
    return [json.loads(l) for l in (DATA / f"m2_{name}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


def build_fixed_panel(rows, per_task=25, seed=SEED):
    rng = random.Random(seed)
    panel = []
    for task in PANEL_TASKS:
        tr = [r for r in rows if r["task"] == task]
        rng.shuffle(tr)
        panel.extend(tr[:per_task])
    return panel


def make_batch(tok: M2Tokenizer, rows):
    """序列 = prompt + SEP + answer + EOS；权重首2.0/内容1.0/EOS0.25，按样本归一。"""
    seqs, metas = [], []
    for r in rows:
        s, meta = tok.encode_pair(r["prompt"], r["answer"])
        seqs.append(s); metas.append(meta)
    n = max(len(s) for s in seqs)
    B = len(rows)
    x = torch.full((B, n), PAD, dtype=torch.long)
    valid = torch.zeros(B, n, dtype=torch.bool)
    weights = torch.zeros(B, n - 1, dtype=torch.float32)
    for j, (s, m) in enumerate(zip(seqs, metas)):
        x[j, :len(s)] = torch.tensor(s)
        valid[j, :len(s)] = True
        a_start, a_len = m["answer_start"], m["answer_len"]
        # 目标：位置 a_start-1(SEP) 预测 a[0]，...；最后 answer 位置预测 EOS
        for k in range(a_len):
            pos = a_start - 1 + k
            weights[j, pos] = W_FIRST if k == 0 else W_CONTENT
        weights[j, a_start + a_len - 1] = W_EOS
    return x, valid, weights


def grouped_loss(logits, target, weights):
    B = weights.size(0)
    tok_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1),
                               reduction="none", ignore_index=PAD)
    w = weights.reshape(-1)
    per = (tok_loss * w).reshape(B, -1).sum(1)
    denom = w.reshape(B, -1).sum(1).clamp_min(1e-6)
    return (per / denom).mean()


def set_eval(model):
    was = model.training
    old_load = getattr(model, "use_router_load_update", None)
    old_grad = getattr(model, "router_gradient_path_enabled", None)
    model.eval()
    if old_load is not None:
        model.use_router_load_update = False
    if old_grad is not None:
        model.router_gradient_path_enabled = False
    return was, old_load, old_grad


def restore_mode(model, saved):
    was, old_load, old_grad = saved
    model.train(was)
    if old_load is not None:
        model.use_router_load_update = old_load
    if old_grad is not None:
        model.router_gradient_path_enabled = old_grad


@torch.inference_mode()
def generate(model, tok: M2Tokenizer, prompt, max_new=12, collect_states=False):
    saved = set_eval(model)
    inp = tok.encode_inference(prompt)               # prompt + SEP
    logits, state = model(torch.tensor([inp]))
    allowed = torch.tensor(allowed_ids())
    out = []
    finish = "max_tokens"
    states = []
    t0 = time.perf_counter()
    for _ in range(max_new):
        tail = logits[0, -1].index_select(0, allowed)
        tid = int(allowed[int(torch.argmax(tail))])
        if tid == EOS:
            finish = "eos"
            break
        out.append(tid)
        if collect_states:
            # 各架构代表状态：GRU/双记忆用 h，Transformer 用 mem 均值池化
            if "h" in state:
                states.append(state["h"][0].detach().clone())
            elif "mem" in state:
                states.append(state["mem"][0].float().mean(0).detach().clone())
            else:
                first = next(iter(state.values()))
                states.append(first[0].detach().float().reshape(-1).clone())
        logits, state = model(torch.tensor([[tid]]), state)
    seconds = time.perf_counter() - t0
    restore_mode(model, saved)
    result = {"text": tok.decode(out), "token_ids": out,
              "finish_reason": finish, "seconds": seconds}
    if collect_states:
        result["states"] = states
    return result


@torch.inference_mode()
def full_answer_lp(model, tok: M2Tokenizer, prompt, answer):
    """log P(answer + EOS | prompt + SEP)，不含 prompt 与 SEP 自身概率。"""
    saved = set_eval(model)
    seq, meta = tok.encode_pair(prompt, answer)
    logits, _ = model(torch.tensor([seq[:-1]]))
    logp = torch.log_softmax(logits[0], -1)
    a_start = meta["answer_start"]  # answer[0] 由 seq[a_start-1]=SEP 预测
    total = 0.0
    for j in range(meta["answer_len"] + 1):
        pos = a_start - 1 + j              # 预测位置
        total += float(logp[pos, seq[pos + 1]])
    restore_mode(model, saved)
    return total


def evaluate_panel(model, tok, panel, collect_details=False):
    per = defaultdict(lambda: {"n": 0, "ok": 0, "eos0": 0, "empty": 0, "overlong": 0})
    details = []
    for r in panel:
        g = generate(model, tok, r["prompt"])
        ok = g["text"].strip() == str(r["answer"]).strip()
        d = per[r["task"]]
        d["n"] += 1; d["ok"] += int(ok)
        d["eos0"] += int(g["finish_reason"] == "eos" and not g["text"].strip())
        d["empty"] += int(not g["text"].strip())
        d["overlong"] += int(g["finish_reason"] == "max_tokens")
        if collect_details:
            details.append({"task": r["task"], "prompt": r["prompt"],
                            "expected": r["answer"], "predicted": g["text"],
                            "ok": ok, "finish": g["finish_reason"], "tokens": g["token_ids"]})
    per_task = {t: {"n": d["n"], "ok": d["ok"],
                    "accuracy": round(d["ok"] / max(1, d["n"]), 4),
                    "immediate_eos_rate": round(d["eos0"] / max(1, d["n"]), 4),
                    "empty_rate": round(d["empty"] / max(1, d["n"]), 4),
                    "overlong_rate": round(d["overlong"] / max(1, d["n"]), 4)}
                for t, d in sorted(per.items())}
    n = sum(d["n"] for d in per.values())
    ok = sum(d["ok"] for d in per.values())
    out = {"per_task": per_task,
           "macro_accuracy": round(sum(d["accuracy"] for d in per_task.values()) / max(1, len(per_task)), 4),
           "micro_accuracy": round(ok / max(1, n), 4),
           "immediate_eos_rate": round(sum(d["eos0"] for d in per.values()) / max(1, n), 4),
           "overlong_rate": round(sum(d["overlong"] for d in per.values()) / max(1, n), 4),
           "total_n": n, "total_ok": ok}
    if collect_details:
        out["details"] = details
    return out


def evaluate_counterfactual(model, tok, limit=400):
    pairs = [json.loads(l) for l in (DATA / "m2_counterfactual.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()][:limit]
    oks = []
    for p in pairs:
        sw = [q for q in p["queries"] if q["task"] == "B_swap"][:2]
        if len(sw) < 2:
            continue
        vals = [generate(model, tok, q["prompt"])["text"].strip() == str(q["answer"]).strip()
                for q in sw]
        oks.append(all(vals))
    n = len(oks)
    return {"pairs": n, "ok": sum(oks), "paired_accuracy": round(sum(oks) / max(1, n), 4)}


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
