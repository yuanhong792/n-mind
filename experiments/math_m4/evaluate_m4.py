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
    invalid_special = [i for i, tid in enumerate(out)
                       if tid in (PAD, SEP, 1, 3) and not (tid == EOS and i == len(out) - 1)]
    return {"text": tok.decode(out, skip_special=True), "ids": out,
            "finish_reason": "eos" if finished else "length",
            "immediate_eos": len(out) == 1 and out[0] == EOS,
            "invalid_special_tokens": invalid_special}


def eval_one(model, tok, row, hard_gate=True):
    g = generate(model, tok, row["prompt"], hard_gate=hard_gate)
    expected_ids = tok.encode(row["answer"]) + [EOS]
    strict_ok = (
        g["ids"] == expected_ids
        and g["finish_reason"] == "eos"
        and not g["invalid_special_tokens"]
    )
    got = g["text"]
    return {"task": row["task"], "prompt": row["prompt"], "expected": row["answer"],
            "predicted": got, "raw_ids": g["ids"], "expected_ids": expected_ids,
            "decoded_content_ok": got == row["answer"],
            "content_ok": strict_ok, "full_ok": strict_ok, "strict_ok": strict_ok,
            "invalid_special_tokens": g["invalid_special_tokens"],
            "finish_reason": g["finish_reason"], "immediate_eos": g["immediate_eos"],
            "overlong": g["finish_reason"] == "length"}


def eval_rows(model, tok, rows, hard_gate=True):
    per, all_results = {}, []
    for r in rows:
        e = eval_one(model, tok, r, hard_gate)
        all_results.append(e)
        d = per.setdefault(e["task"], {"n": 0, "ok": 0, "full_ok": 0,
                                        "immediate_eos": 0, "overlong": 0,
                                        "decoded_ok": 0})
        d["n"] += 1
        d["ok"] += int(e["strict_ok"])
        d["full_ok"] += int(e["strict_ok"])
        d["decoded_ok"] += int(e["decoded_content_ok"])
        d["immediate_eos"] += int(e["immediate_eos"])
        d["overlong"] += int(e["overlong"])
    n = len(all_results)
    ok = sum(e["strict_ok"] for e in all_results)
    decoded_ok = sum(e["decoded_content_ok"] for e in all_results)
    full = sum(e["full_ok"] for e in all_results)
    macro = sum(d["ok"] / d["n"] for d in per.values()) / len(per) if per else 0.0
    return {"n": n, "ok": ok, "content_accuracy": round(ok / max(1, n), 4),
            "decoded_content_accuracy": round(decoded_ok / max(1, n), 4),
            "full_answer_natural_eos": round(full / max(1, n), 4),
            "macro_accuracy": round(macro, 4),
            "strict_macro": round(macro, 4),
            "immediate_eos_rate": round(sum(e["immediate_eos"] for e in all_results) / max(1, n), 4),
            "overlong_rate": round(sum(e["overlong"] for e in all_results) / max(1, n), 4),
            "per_task": per, "results": all_results}


def tf_position_accuracy(model, tok, rows, protocol="A", hard_gate=True):
    """Teacher-forced 逐位置准确率（原始 token id，答案位置），与自由生成分开报告。"""
    correct = total = 0
    kw = {"hard_gate": hard_gate} if "hard_gate" in model.forward.__code__.co_varnames else {}
    for r in rows:
        seq, _ = tok.encode_pair(r["prompt"], r["answer"])
        sep = seq.index(SEP)
        x = torch.tensor([seq[:-1]])
        valid = torch.ones(1, x.size(1), dtype=torch.bool)
        logits, _ = model(x, valid_mask=valid, **kw)
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


def load_critical_minimums():
    """从冻结预登记方案读取关键任务下限，避免代码与方案漂移。"""
    p = HERE / "M4_PREREGISTERED_PLAN.json"
    if not p.exists():
        raise FileNotFoundError(p)
    plan = json.loads(p.read_text(encoding="utf-8"))
    return plan["capability_gate"]["critical_tasks_minimums"]


def critical_metrics(model, tok, panel, cf_pairs, distance_rows, hard_gate=True):
    """关键能力指标：严格 token + 自然 EOS；反事实按配对两条都正确计。"""
    ev = eval_rows(model, tok, panel, hard_gate=hard_gate)
    cf = counterfactual_eval(model, tok, cf_pairs)
    dist = distance_eval(model, tok, distance_rows)
    out = {}
    for task, cell in ev["per_task"].items():
        out[task] = round(cell["ok"] / cell["n"], 4)
    out["counterfactual_same_query_fact_change"] = cf.get(
        "same_query_fact_change", {}).get("pair_rate", 0.0)
    out["counterfactual_unrelated_fact_invariance"] = cf.get(
        "unrelated_fact_change", {}).get("pair_rate", 0.0)
    out["query_switch"] = cf.get("query_switch", {}).get("query_switch_rate", 0.0)
    out["order_swap_invariance"] = cf.get("order_swap", {}).get("pair_rate", 0.0)
    return {
        "critical": out,
        "strict_macro": ev["strict_macro"],
        "decoded_content_accuracy": ev["decoded_content_accuracy"],
        "full_answer_natural_eos": ev["full_answer_natural_eos"],
        "counterfactual": cf,
        "distance": dist,
    }


def iso_record(curve, seed=None, critical_minimums=None):
    """M41 严格 ISO 判定。

    只接受连续两个独立验证点；成本计入第二个确认点；单 seed 独立判断。
    strict_macro 必须来自原始生成 token 序列 + 自然 EOS；
    critical 必须来自预登记关键任务下限；hard_gate 必须为 True。
    """
    mins = critical_minimums if critical_minimums is not None else load_critical_minimums()
    out = {}
    for tgt in ISO_TARGETS:
        hits, reasons = [], []
        for pt in curve:
            problems = []
            if pt.get("hard_gate") is not True:
                problems.append("hard_gate_not_true")
            strict = pt.get("strict_macro")
            if strict is None:
                problems.append("strict_macro_missing")
            elif float(strict) < tgt:
                problems.append("strict_macro_below_target")
            crit = pt.get("critical")
            if not isinstance(crit, dict):
                problems.append("critical_metrics_missing")
            else:
                for task, floor in mins.items():
                    if float(crit.get(task, 0.0)) < float(floor):
                        problems.append(f"critical_below_floor:{task}")
            if not problems:
                hits.append(True)
                reasons.append({"step": pt.get("step"), "strict_macro": strict,
                                "critical_snapshot": {
                                    k: crit.get(k) for k in mins}})
            else:
                hits.append(False)
                reasons.append({"step": pt.get("step"), "problems": problems})

        pair = next((i for i in range(len(hits) - 1) if hits[i] and hits[i + 1]), None)
        key = f"{int(tgt * 100)}%"
        if pair is None:
            out[key] = {"status": "TARGET_NOT_REACHED", "seed": seed,
                        "rule": "需要连续两个独立验证点同时达标",
                        "last_point": reasons[-1] if reasons else None}
            continue
        first, second = curve[pair], curve[pair + 1]
        cost = second.get("cumulative_macs")
        cost_source = "cumulative_macs@second_confirmation"
        if cost is None:
            cost = second.get("cost_macs")
            cost_source = "cost_macs@second_confirmation"
        out[key] = {
            "status": "REACHED_STABLE" if cost is not None else "REACHED_STABLE_COST_UNAVAILABLE",
            "seed": seed,
            "first_confirmation_step": first.get("step"),
            "confirmed_step": second.get("step"),
            "strict_macro": second.get("strict_macro"),
            "cost": cost,
            "cost_source": cost_source,
            "evidence": reasons[pair:pair + 2],
            "note": "末尾单点达标不算稳定；成本只取第二个确认点。",
        }
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
