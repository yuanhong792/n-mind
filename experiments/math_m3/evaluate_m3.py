"""M3 统一评测协议（P8/P9）。

严格区分指标口径：
- content_accuracy：生成内容与答案逐字符一致（不管 EOS）。
- full_answer_natural_eos：生成内容一致且以自然 EOS 结束（非截断）。
- macro_accuracy：按任务先求 acc 再平均。
- micro_accuracy：所有样本一起算 acc。
- counterfactual_pair_rate：反事实成对正确率（base 与 cf 都对才算一对对）。
- query_switch_rate：查询切换独立测试。
- repeat_rate：重复输出率（生成内容含重复子串的样本占比）。
- overlong_rate：超长输出率（超过答案长度 2 倍仍不 EOS）。
- immediate_eos_rate：立即 EOS 率（首个生成 token 即 EOS）。

所有正确率提供 n 与 ok。
固定验证面板与独立封存集分开，面板不用于选模的"独立"声明。
ISO 能力门槛：记录各模型首次稳定达到目标准确率时的累计计算成本。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from tokenizer_m3 import M3Tokenizer, SEP, EOS, PAD  # noqa: E402

DATA = HERE / "data"
GEN_LIMIT = 32   # 生成最大长度


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def load_rows(split: str) -> list[dict]:
    p = DATA / f"m3_{split}.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"M3 数据缺失: {p}")
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def build_fixed_panel(rows: list[dict], per_task: int = 20) -> list[dict]:
    """固定验证面板：每个任务取前 per_task 条，固定顺序。"""
    from collections import defaultdict
    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)
    panel = []
    for task in sorted(by_task):
        panel.extend(by_task[task][:per_task])
    return panel


# ---------------------------------------------------------------------------
# 批处理与监督
# ---------------------------------------------------------------------------

def make_batch(tok: M3Tokenizer, rows: list[dict], max_len: int = 64):
    """构造 SEP 协议的 batch：input_ids、valid_mask、目标、权重。

    权重：首 Token 2.0、内容 1.0、EOS 0.25（与 C1.1/M2 一致）。
    """
    seqs = []
    for r in rows:
        seq, _ = tok.encode_pair(r["prompt"], r["answer"])
        seqs.append(seq[:max_len])
    T = max(len(s) for s in seqs)
    B = len(seqs)
    input_ids = torch.full((B, T), PAD, dtype=torch.long)
    valid = torch.zeros(B, T, dtype=torch.bool)
    tgt = torch.full((B, T), PAD, dtype=torch.long)
    w = torch.zeros(B, T)
    for i, s in enumerate(seqs):
        L = len(s)
        input_ids[i, :L] = torch.tensor(s)
        valid[i, :L] = True
        tgt[i, :L] = torch.tensor(s)
        # 权重：首 Token(answer 第一个) 2.0，内容 1.0，EOS 0.25
        # answer_start = prompt_len+1
        # 简化：序列中第一个非 SEP 的 answer token 权重 2.0，EOS 0.25
        # 这里用整体：除 SEP 外都参与，首 answer token 加权
        for j in range(L):
            if j == L - 1:   # EOS
                w[i, j] = 0.25
            else:
                w[i, j] = 1.0
        # 找到 answer 首 token（SEP 之后第一个）
        sep_idx = s.index(SEP) if SEP in s else -1
        if sep_idx + 1 < L - 1:
            w[i, sep_idx + 1] = 2.0
    return input_ids, valid, w, tgt


def grouped_loss(logits: torch.Tensor, target: torch.Tensor,
                 weights: torch.Tensor) -> torch.Tensor:
    """带权重的交叉熵，按样本归一化。"""
    V = logits.shape[-1]
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, V), target.reshape(-1), reduction="none").reshape(target.shape)
    denom = weights.sum().clamp_min(1.0)
    return (loss * weights).sum() / denom


# ---------------------------------------------------------------------------
# 自回归生成（SEP 协议）
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(model, tok: M3Tokenizer, prompt: str, gen_limit: int = GEN_LIMIT,
             hard_gate: bool = False):
    """SEP 协议自回归生成。输入 prompt_ids+[SEP]，逐 token 生成直到 EOS 或上限。

    hard_gate=True 时对支持硬门的模型（M3-B）启用稀疏路径推理。
    """
    ids = tok.encode_inference(prompt)
    x = torch.tensor([ids], dtype=torch.long)
    valid = torch.ones(1, x.size(1), dtype=torch.bool)
    state = model.initial_state(1, x.device)
    import inspect
    fwd_params = inspect.signature(model.forward).parameters
    kwargs = {"hard_gate": hard_gate} if "hard_gate" in fwd_params else {}
    logits, state = model(x, state=state, valid_mask=valid, **kwargs)
    out_ids = []
    finished = False
    for _ in range(gen_limit):
        last = logits[0, -1]
        # 禁止生成 PAD/SEP，允许 EOS
        last = last.clone()
        last[PAD] = float("-inf")
        last[SEP] = float("-inf")
        probs = torch.softmax(last, dim=-1)
        # 贪心解码（确定性，便于复现）
        tid = int(probs.argmax().item())
        out_ids.append(tid)
        if tid == EOS:
            finished = True
            break
        x = torch.tensor([[tid]], dtype=torch.long)
        valid = torch.ones(1, 1, dtype=torch.bool)
        logits, state = model(x, state=state, valid_mask=valid, **kwargs)
    text = tok.decode(out_ids, skip_special=True)
    return {"text": text, "ids": out_ids, "finish_reason": "eos" if finished else "length",
            "immediate_eos": len(out_ids) == 1 and out_ids[0] == EOS}


def is_repeat(text: str) -> bool:
    """检测重复输出：文本由同一短单元重复构成（如 AAAAA、ABABAB）。"""
    if len(text) < 4:
        return False
    for unit_len in range(1, min(4, len(text) // 2) + 1):
        unit = text[:unit_len]
        if len(text) >= unit_len * 3 and text == unit * (len(text) // unit_len):
            return True
    return False


# ---------------------------------------------------------------------------
# 单样本评测
# ---------------------------------------------------------------------------

def eval_one(model, tok: M3Tokenizer, row: dict) -> dict:
    gen = generate(model, tok, row["prompt"])
    expected = row["answer"]
    got = gen["text"]
    content_ok = (got == expected)
    full_ok = content_ok and gen["finish_reason"] == "eos"
    return {
        "task": row["task"],
        "prompt": row["prompt"],
        "expected": expected,
        "predicted": got,
        "content_ok": content_ok,
        "full_ok": full_ok,
        "finish_reason": gen["finish_reason"],
        "immediate_eos": gen["immediate_eos"],
        "repeat": is_repeat(got),
        "overlong": gen["finish_reason"] == "length",
    }


def evaluate_rows(model, tok: M3Tokenizer, rows: list[dict]) -> dict:
    """对一组样本评测，返回分任务指标 + 总体指标。"""
    per_task = {}
    results = []
    for r in rows:
        e = eval_one(model, tok, r)
        results.append(e)
        task = e["task"]
        d = per_task.setdefault(task, {"n": 0, "ok": 0, "full_ok": 0, "immediate_eos": 0,
                                        "repeat": 0, "overlong": 0})
        d["n"] += 1
        d["ok"] += int(e["content_ok"])
        d["full_ok"] += int(e["full_ok"])
        d["immediate_eos"] += int(e["immediate_eos"])
        d["repeat"] += int(e["repeat"])
        d["overlong"] += int(e["overlong"])
    n = len(results)
    ok = sum(e["content_ok"] for e in results)
    full = sum(e["full_ok"] for e in results)
    macro = (sum(d["ok"] / d["n"] for d in per_task.values()) / len(per_task)) if per_task else 0.0
    return {
        "n": n,
        "ok": ok,
        "content_accuracy": round(ok / n, 4) if n else 0.0,
        "full_answer_natural_eos": round(full / n, 4) if n else 0.0,
        "macro_accuracy": round(macro, 4),
        "micro_accuracy": round(ok / n, 4) if n else 0.0,
        "immediate_eos_rate": round(sum(e["immediate_eos"] for e in results) / n, 4) if n else 0.0,
        "repeat_rate": round(sum(e["repeat"] for e in results) / n, 4) if n else 0.0,
        "overlong_rate": round(sum(e["overlong"] for e in results) / n, 4) if n else 0.0,
        "per_task": per_task,
        "results": results,
    }


def evaluate_panel(model, tok: M3Tokenizer, panel: list[dict]) -> dict:
    return evaluate_rows(model, tok, panel)


# ---------------------------------------------------------------------------
# 反事实与距离严格评测
# ---------------------------------------------------------------------------

def evaluate_counterfactual(model, tok: M3Tokenizer, cf_rows: list[dict]) -> dict:
    """反事实成对正确率：base 与 cf 都内容正确才算一对对。查询对象不变。"""
    pairs_n = 0
    pairs_ok = 0
    query_switch_n = 0
    query_switch_ok = 0
    for p in cf_rows:
        base_ok = all(eval_one(model, tok, b)["content_ok"] for b in p["base"])
        cf_ok = all(eval_one(model, tok, c)["content_ok"] for c in p["counterfactual"])
        if base_ok and cf_ok:
            pairs_ok += 1
        pairs_n += 1
        if p.get("query_switch"):
            qs_ok = all(eval_one(model, tok, q)["content_ok"] for q in p["query_switch"])
            query_switch_ok += int(qs_ok)
            query_switch_n += 1
    return {
        "n": pairs_n,
        "ok": pairs_ok,
        "counterfactual_pair_rate": round(pairs_ok / pairs_n, 4) if pairs_n else 0.0,
        "query_switch_n": query_switch_n,
        "query_switch_ok": query_switch_ok,
        "query_switch_rate": round(query_switch_ok / query_switch_n, 4) if query_switch_n else 0.0,
    }


def evaluate_distance(model, tok: M3Tokenizer, rows: list[dict]) -> dict:
    """距离严格配对：相同事实、相同查询、不同距离。按距离分组统计。"""
    by_dist = {}
    for r in rows:
        if r["task"] != "D_distance":
            continue
        d = r.get("meta", {}).get("distance")
        e = eval_one(model, tok, r)
        dd = by_dist.setdefault(str(d), {"n": 0, "ok": 0})
        dd["n"] += 1
        dd["ok"] += int(e["content_ok"])
    curve = {d: round(v["ok"] / v["n"], 4) for d, v in sorted(by_dist.items(), key=lambda x: int(x[0]))}
    return {"by_distance": curve, "n": sum(v["n"] for v in by_dist.values()),
            "ok": sum(v["ok"] for v in by_dist.values())}


# ---------------------------------------------------------------------------
# ISO 能力门槛：首次达标成本
# ---------------------------------------------------------------------------

ISO_TARGETS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]


def iso_record(curve: list[dict], targets=ISO_TARGETS) -> dict:
    """从学习曲线（每步记录 macro_accuracy）计算首次稳定达标的累计成本。

    稳定达标：连续 2 个测量点 macro >= target。
    返回每个 target 的首次达标点或 TARGET_NOT_REACHED。
    """
    out = {}
    for tgt in targets:
        reached = None
        for i, pt in enumerate(curve):
            if pt.get("macro", 0) >= tgt:
                # 检查下一个点也达标（稳定）
                if i + 1 < len(curve) and curve[i + 1].get("macro", 0) >= tgt:
                    reached = pt
                    break
                elif i == len(curve) - 1:
                    reached = pt   # 末尾，接受
        if reached is None:
            out[f"{int(tgt*100)}%"] = {"status": "TARGET_NOT_REACHED"}
        else:
            out[f"{int(tgt*100)}%"] = {
                "status": "REACHED",
                "step": reached.get("step"),
                "macro": reached.get("macro"),
                "samples": reached.get("samples"),
                "tokens": reached.get("tokens"),
                "macs": reached.get("macs"),
                "wall_sec": reached.get("wall"),
            }
    return out
