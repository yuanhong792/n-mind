"""V171 P2/P3/P5: unified, math-correct evaluation protocol.

Two explicitly separated scoring modes:
  A) DISCRIMINATIVE_TOKEN_PROXY - first discriminative token logprob only (fast monitor).
  B) FULL_ANSWER_CONDITIONAL    - log P(answer_tokens + EOS | prompt), autoregressive,
                                  every answer token + EOS included.

Teacher-forced full-sequence forward and incremental state-passing forward are
mathematically equivalent for a causal LM; we compute B with the single-forward
teacher-forced form (shared prompt, correct joint probability) and VERIFY the
agreement against the incremental form in the regression tests.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
from collections import Counter, defaultdict

import torch

ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT / "src"))
from nmind.language.tokenizer_v2 import ByteBPETokenizer, EOS
from nmind.language.bpe_decoder import (effective_token_ids, generate_bpe,
                                        utf8_prefix_status)

V17 = ROOT / "experiments/language_v1/v17"
V171 = ROOT / "experiments/language_v1/v171"

LEGACY = [("用户：你好\n助手：","你好！"),("用户：你叫什么？\n助手：","我叫N-MIND。"),("用户：谢谢\n助手：","不客气！"),
          ("用户：小明有红色苹果\n助手：","小明的苹果是红色的。"),("用户：小红有绿色苹果\n助手：","小红的苹果是绿色的。"),
          ("用户：小刚有蓝色苹果\n助手：","小刚的苹果是蓝色的。"),("用户：小丽有黄色苹果\n助手：","小丽的苹果是黄色的。"),
          ("用户：我叫小明\n助手：","你好，小明！"),("用户：我叫什么？\n助手：","你叫小明。")]
COLORS = ["红色","绿色","蓝色","黄色","紫色","橙色","黑色","白色"]
PERSONS = ["王明","李红","张刚","陈丽","赵强","周美","孙华","吴军","郑芳","刘杰",
           "黄敏","徐磊","马琳","朱婷","胡宇","郭浩","何欣","高睿","林彤","罗凯"]
OBJECTS = ["苹果","书包","杯子","香蕉","橘子","皮球","铅笔","本子"]
LOCATIONS = ["桌上","柜子","椅子","床边"]

# 统一候选集：所有模型完全相同、有界、可复现
CANDIDATES_BY_TASK = {
    "A_dialogue": ["你好！", "我叫N-MIND。", "不客气！"],
    "B_bind": COLORS,
    "C_counter": COLORS,
    "D_position": PERSONS + OBJECTS + LOCATIONS,
    "E_context": PERSONS,
}
MAX_NEW_TOKENS = 16


def build_tokenizer() -> ByteBPETokenizer:
    return ByteBPETokenizer.train([p + a for p, a in LEGACY], vocab_size=4096)


def build_vocab(tok):
    valid = torch.tensor(sorted(effective_token_ids(tok)))
    lookup = torch.full((tok.vocab_size,), -1, dtype=torch.long)
    lookup[valid] = torch.arange(valid.numel())
    return valid, lookup


def load_rows(split: str, base=V17):
    return [json.loads(l) for l in open(base / f"v17_{split}.jsonl", encoding="utf-8")]


# ---------- B: full answer conditional logprob ----------
def full_answer_logprob(model, tok, prompt: str, answer: str, valid, lookup,
                        *, return_per_token: bool = False):
    """log P(tok(answer)+EOS | prompt), 自回归逐 token, 所有答案 token 与 EOS 均计入。

    单次 teacher-forced 前向：因果 LM 中位置 t 的 logits 预测 t+1, 故在
    [prompt+answer+EOS] 上做一次前向并读取答案各位置的 logprob 即为联合条件概率，
    与逐 token 增量 state passing 等价（回归测试验证）。
    """
    pids = tok.encode(prompt)
    aids = tok.encode(answer) + [EOS]
    seq = pids + aids
    L = len(seq)
    if L < 2:
        return (float("-inf"), []) if return_per_token else float("-inf")
    x = torch.tensor([seq[:-1]], dtype=torch.long)
    with torch.inference_mode():
        logits, _ = model(x)
        z = logits.index_select(-1, valid)
        t_start = len(pids) - 1          # 预测第一个答案 token 的 logit 位置
        t_end = L - 2                    # 预测 EOS 的 logit 位置（含）
        if t_start < 0 or t_end < t_start:
            return (float("-inf"), []) if return_per_token else float("-inf")
        pred = z[0, t_start:t_end + 1]
        lp = torch.log_softmax(pred, -1)
        targets = torch.tensor(seq[t_start + 1:t_end + 2], dtype=torch.long)
        idx = lookup[targets]
        if (idx < 0).any():
            return (float("-inf"), []) if return_per_token else float("-inf")
        per = lp.gather(1, idx.unsqueeze(1)).squeeze(1)
        vals = [float(v) for v in per]
    return (sum(vals), vals) if return_per_token else sum(vals)


def full_answer_logprob_batch(model, tok, prompt: str, answers: list[str], valid, lookup,
                              *, return_per_token: bool = False):
    """批量版：一份 prompt 对多个候选, 一次前向算完全部完整答案条件概率。

    因果性：位置 t 的 logits 只依赖 tokens[0..t], 故右侧 padding 不影响答案区 logits。
    返回 {answer: total_logprob}。数学上与逐个 full_answer_logprob 一致（测试验证）。
    """
    pids = tok.encode(prompt)
    seqs = [pids + tok.encode(a) + [EOS] for a in answers]
    L = max(len(s) for s in seqs)
    B = len(seqs)
    x = torch.zeros((B, L - 1), dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, :len(s) - 1] = torch.tensor(s[:-1])
    with torch.inference_mode():
        logits, _ = model(x)
        z = logits.index_select(-1, valid)
        out = {}; per_map = {}
        for i, s in enumerate(seqs):
            t_start = len(pids) - 1
            t_end = len(s) - 2
            if t_end < t_start:
                out[answers[i]] = float("-inf"); per_map[answers[i]] = []; continue
            pred = z[i, t_start:t_end + 1]
            lp = torch.log_softmax(pred, -1)
            targets = torch.tensor(s[t_start + 1:t_end + 2], dtype=torch.long)
            idx = lookup[targets]
            if (idx < 0).any():
                out[answers[i]] = float("-inf"); per_map[answers[i]] = []; continue
            per = lp.gather(1, idx.unsqueeze(1)).squeeze(1)
            vals = [float(v) for v in per]
            out[answers[i]] = sum(vals); per_map[answers[i]] = vals
        if return_per_token:
            return out, per_map
        return out


def full_answer_incremental_logprob(model, tok, prompt: str, answer: str, valid, lookup):
    """独立实现：逐 token 增量 state passing, 仅用于等价性验证, 不用于主评分。"""
    aids = tok.encode(answer) + [EOS]
    with torch.inference_mode():
        logits, state = model(torch.tensor([tok.encode(prompt)]))
        total = 0.0
        for tid in aids:
            lp = torch.log_softmax(logits[0, -1].index_select(0, valid), -1)
            pos = int((valid == tid).nonzero(as_tuple=True)[0][0])
            total += float(lp[pos])
            logits, state = model(torch.tensor([[tid]]), state)
    return total


# ---------- A: fast discriminative-token proxy ----------
def discriminative_positions(tok):
    enc = {c: tok.encode(c) for c in COLORS}
    fd = {}
    for c, ids in enc.items():
        max_lcp = 0
        for other, oids in enc.items():
            if other == c:
                continue
            lcp = 0
            for a, b in zip(ids, oids):
                if a == b:
                    lcp += 1
                else:
                    break
            max_lcp = max(max_lcp, lcp)
        fd[c] = min(max_lcp, len(ids) - 1)
    return enc, fd


def proxy_first_token_lp(model, tok, prompt: str, cands, valid, fd):
    """一次 prompt forward, 每个候选取判别 token 的 logprob。标记 PROXY, 非完整概率。"""
    with torch.inference_mode():
        logits, _ = model(torch.tensor([tok.encode(prompt)]))
        lp = torch.log_softmax(logits[0, -1].index_select(0, valid), -1)
        out = {}
        for c in cands:
            aids = tok.encode(c)
            tid = aids[min(fd.get(c, 0), len(aids) - 1)]
            hits = (valid == tid).nonzero(as_tuple=True)[0]
            out[c] = float(lp[int(hits[0])]) if hits.numel() else float("-inf")
        return out


def proxy_first_token_ce(model, tok, prompt: str, answer: str, valid, fd):
    aids = tok.encode(answer)
    tid = aids[min(fd.get(answer, 0), len(aids) - 1)]
    lp = proxy_first_token_lp(model, tok, prompt, [answer], valid, fd)
    return -lp[answer]


def tf_ce_full(model, tok, prompt: str, answer: str, valid, lookup):
    """完整答案 Teacher-forced CE = -mean(logprob of answer tokens + EOS)。"""
    total, per = full_answer_logprob(model, tok, prompt, answer, valid, lookup,
                                     return_per_token=True)
    if not per:
        return None
    return -total / len(per)


# ---------- EOS / free generation diagnostics (P5) ----------
def eos_diagnostics(g):
    raw = bytes.fromhex(g.raw_byte_hex) if g.raw_byte_hex else b""
    status = utf8_prefix_status(raw)
    ntok = len(g.token_ids)
    text_nonempty = bool(g.text.strip())
    return {
        "finish_reason": g.finish_reason,
        "generated_token_count": ntok,
        "utf8_status": status,
        "text": g.text,
        "immediate_eos": bool(ntok == 0 and g.finish_reason == "eos"),
        "non_empty_answer": bool(ntok > 0 and text_nonempty),
        "valid_utf8_non_empty": bool(status == "complete" and ntok > 0 and text_nonempty),
        "correct_eos": bool(g.finish_reason == "eos" and ntok > 0
                            and status == "complete" and text_nonempty),
    }


def evaluate_model(model, tok, rows, *, valid, lookup, fd,
                   max_new_tokens: int = MAX_NEW_TOKENS,
                   candidate_set=CANDIDATES_BY_TASK,
                   do_free: bool = True,
                   do_full: bool = True,
                   label: str = ""):
    """统一评测：A(PROXY) + B(FULL) + 自由生成 + EOS 诊断。所有指标带分子/分母。"""
    model.eval()
    model.use_router_load_update = False
    per_task = defaultdict(lambda: {
        "n": 0, "proxy_correct": 0, "full_correct": 0,
        "free_correct": 0, "immediate_eos": 0, "non_empty": 0,
        "valid_utf8_non_empty": 0, "correct_eos": 0, "utf8_complete": 0,
        "proxy_ce_sum": 0.0, "full_tf_ce_sum": 0.0, "full_lp_sum": 0.0,
    })
    details = []
    finish_counter = Counter()
    for r in rows:
        task = r["task"]
        gold = r["answer"]
        cands = list(candidate_set.get(task, [gold]))
        if gold not in cands:
            cands.append(gold)
        d = per_task[task]; d["n"] += 1
        # A: proxy
        plp = proxy_first_token_lp(model, tok, r["prompt"], cands, valid, fd)
        ppred = max(plp, key=plp.get)
        d["proxy_correct"] += int(ppred == gold)
        pce = -plp[gold]
        d["proxy_ce_sum"] += pce
        # B: full answer conditional (batched: one forward for all candidates)
        fpred, gold_lp, ftf_ce = None, None, None
        if do_full:
            scores, per_map = full_answer_logprob_batch(model, tok, r["prompt"], cands, valid, lookup,
                                                        return_per_token=True)
            fpred = max(scores, key=scores.get)
            gold_lp = scores[gold]
            d["full_correct"] += int(fpred == gold)
            d["full_lp_sum"] += gold_lp
            gp = per_map.get(gold, [])
            ftf_ce = (-gold_lp / len(gp)) if gp else None
            d["full_tf_ce_sum"] += (ftf_ce if ftf_ce is not None else 0.0)
        # free generation + EOS
        ed = None
        if do_free:
            g = generate_bpe(model, tok, r["prompt"], max_new_tokens=max_new_tokens)
            ed = eos_diagnostics(g)
            finish_counter[ed["finish_reason"]] += 1
            d["free_correct"] += int(g.text.strip() == gold)
            d["immediate_eos"] += int(ed["immediate_eos"])
            d["non_empty"] += int(ed["non_empty_answer"])
            d["valid_utf8_non_empty"] += int(ed["valid_utf8_non_empty"])
            d["correct_eos"] += int(ed["correct_eos"])
            d["utf8_complete"] += int(ed["utf8_status"] == "complete")
        details.append({
            "task": task, "expected": gold, "proxy_pred": ppred,
            "full_pred": fpred, "proxy_ce": round(pce, 4),
            "full_gold_logprob": round(gold_lp, 4) if gold_lp is not None else None,
            "full_tf_ce": round(ftf_ce, 4) if ftf_ce is not None else None,
            "free": (ed or {}).get("text"),
            "finish": (ed or {}).get("finish_reason"),
            "utf8": (ed or {}).get("utf8_status"),
            "immediate_eos": (ed or {}).get("immediate_eos"),
        })
    summary = {}
    for t, d in sorted(per_task.items()):
        n = d["n"]
        summary[t] = {
            "n": n,
            "DISCRIMINATIVE_TOKEN_PROXY_acc": round(d["proxy_correct"] / n, 4),
            "FULL_ANSWER_CONDITIONAL_acc": round(d["full_correct"] / n, 4) if do_full else None,
            "free_acc": round(d["free_correct"] / n, 4) if do_free else None,
            "proxy_first_token_ce": round(d["proxy_ce_sum"] / n, 4),
            "full_answer_tf_ce": round(d["full_tf_ce_sum"] / n, 4) if do_full else None,
            "full_answer_mean_logprob": round(d["full_lp_sum"] / n, 4) if do_full else None,
            "immediate_eos_rate": round(d["immediate_eos"] / n, 4) if do_free else None,
            "non_empty_answer_rate": round(d["non_empty"] / n, 4) if do_free else None,
            "valid_utf8_non_empty_rate": round(d["valid_utf8_non_empty"] / n, 4) if do_free else None,
            "correct_eos_rate": round(d["correct_eos"] / n, 4) if do_free else None,
            "utf8_complete_rate": round(d["utf8_complete"] / n, 4) if do_free else None,
        }
    # 全局 EOS 汇总（所有样本同一分母）
    N = len(rows)
    totals = defaultdict(int)
    for d in per_task.values():
        for k in ("immediate_eos", "non_empty", "valid_utf8_non_empty", "correct_eos", "utf8_complete", "free_correct"):
            totals[k] += d[k]
    overall = {
        "n": N,
        "immediate_eos_rate": round(totals["immediate_eos"] / N, 4) if do_free and N else None,
        "non_empty_answer_rate": round(totals["non_empty"] / N, 4) if do_free and N else None,
        "valid_utf8_non_empty_rate": round(totals["valid_utf8_non_empty"] / N, 4) if do_free and N else None,
        "correct_eos_rate": round(totals["correct_eos"] / N, 4) if do_free and N else None,
        "utf8_complete_rate": round(totals["utf8_complete"] / N, 4) if do_free and N else None,
        "free_acc": round(totals["free_correct"] / N, 4) if do_free and N else None,
        "finish_reason_counts": dict(finish_counter),
    }
    return {"label": label, "n_rows": N, "summary": summary,
            "overall": overall, "details": details}


def counterfactual_paired(model, tok, rows, *, valid, lookup, fd,
                          candidate_set=CANDIDATES_BY_TASK):
    """反事实成对：同一 key 两题(原/改)必须同时用 FULL 评分正确才算一对通过。"""
    cmap = defaultdict(list)
    for r in rows:
        if r["task"] == "C_counter":
            cmap[r["key"]].append(r)
    pairs = [pr for pr in cmap.values() if len(pr) == 2 and pr[0]["answer"] != pr[1]["answer"]]
    ok = 0
    pair_details = []
    for pr in pairs:
        res = []
        for r in pr:
            cands = list(candidate_set.get(r["task"], [r["answer"]]))
            if r["answer"] not in cands:
                cands.append(r["answer"])
            scores = full_answer_logprob_batch(model, tok, r["prompt"], cands, valid, lookup)
            res.append({"expected": r["answer"], "pred": max(scores, key=scores.get),
                        "correct": max(scores, key=scores.get) == r["answer"]})
        both = all(x["correct"] for x in res)
        ok += int(both)
        pair_details.append({"both_correct": both, "questions": res})
    return {"pairs_total": len(pairs), "pairs_passed": ok,
            "paired_accuracy": round(ok / len(pairs), 4) if pairs else None,
            "details": pair_details}


def shortcut_baselines(rows):
    freq = Counter(r["answer"] for r in rows)
    fixed = max(freq.values()) / len(rows) if rows else 0.0
    b = [r for r in rows if r["task"] == "B_bind"]
    nearest = sum(1 for r in b if r["meta"].get("nearest_attr") == r["answer"]) / len(b) if b else 0.0
    position = sum(1 for r in b if r["meta"].get("target_is_first")) / len(b) if b else 0.0
    return {"fixed_answer": round(fixed, 4), "fixed_answer_n": len(rows),
            "nearest_attribute": round(nearest, 4), "nearest_attribute_n": len(b),
            "position_first": round(position, 4), "position_first_n": len(b)}
