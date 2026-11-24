"""M2.1 P0.2：真正受监督位置的冲突重统计。

M2 缺陷：原扫描器把「prompt 内部未受监督的延续字符」也计入冲突判断。
本轮只扫描实际受监督预测位置：
    完整历史 token 序列 → 实际目标 token → 实际 loss 权重 > 0
只把「相同受监督历史（含相同权重定义下的目标位置）对应不同目标 token」计为真实冲突。

输出：
- 冲突前缀数量、受影响监督位置数量、受影响样本比例、各任务冲突分布、
  冲突 token 实际频率、SEP 前后变化、完整 prompt 多答案真实歧义。
- 冲突样本条件熵与理论最低交叉熵（H = -Σ p log p，按经验频率估计）。
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
M3 = HERE.parent
M2ROOT = M3.parent / "math_m2"
sys.path.insert(0, str(M2ROOT))
sys.path.insert(0, str(M3))

from tokenizer_m2 import M2Tokenizer, EOS, SEP  # noqa: E402
from m2_protocol_repair import encode_sequence, PROTOCOLS  # noqa: E402

W_FIRST, W_CONTENT, W_EOS = 2.0, 1.0, 0.25
HISTORY_FILES = [
    M2ROOT / "data" / "m2_train.jsonl",
    M2ROOT / "data" / "m2_val.jsonl",
    M2ROOT / "data" / "m2_sealed.jsonl",
    M2ROOT.parent / "math_m1" / "data" / "m1_train.jsonl",
    M2ROOT.parent / "compute_c1" / "data" / "c1_train.jsonl",
    M2ROOT.parent / "compute_c11" / "data" / "c11_train.jsonl",
]


def load_samples():
    samples = []
    for p in HISTORY_FILES:
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(r.get("prompt"), str) and r.get("answer") is not None:
                samples.append({"prompt": r["prompt"], "answer": str(r["answer"]),
                                "task": r.get("task", "?"), "source": p.name})
    return samples


def supervised_positions(tok, sample, protocol):
    """返回 [(history_tuple, target_id, weight, task)]，只含 loss 权重>0 的预测位置。"""
    seq, meta = encode_sequence(tok, sample["prompt"], sample["answer"], protocol)
    a_start, a_len = meta["answer_start"], meta["answer_len"]
    out = []
    # answer[0] 由位置 a_start-1 的历史预测；answer[k] 由 a_start-1+k 的历史预测
    for k in range(a_len):
        pos = a_start - 1 + k
        hist = tuple(seq[:pos + 1])
        w = W_FIRST if k == 0 else W_CONTENT
        out.append((hist, seq[pos + 1], w, sample["task"]))
    # EOS 由最后一个 answer 位置预测
    eos_pos = a_start + a_len - 1
    out.append((tuple(seq[:eos_pos + 1]), EOS, W_EOS, sample["task"]))
    return out


def analyze(protocol, samples, tok):
    # hist -> Counter(target_id)，同时记录 task 分布与样本来源
    hist_targets = defaultdict(Counter)
    hist_tasks = defaultdict(Counter)
    n_positions = 0
    sample_hits = set()
    for i, s in enumerate(samples):
        for hist, target, w, task in supervised_positions(tok, s, protocol):
            n_positions += 1
            hist_targets[hist][target] += 1
            hist_tasks[hist][task] += 1
    conflicts = {h: c for h, c in hist_targets.items() if len(c) > 1}
    # 受影响监督位置 = 冲突历史出现次数之和
    affected_positions = sum(sum(c.values()) for c in conflicts.values())
    # 冲突 token 频率（仅冲突历史中的目标）
    conflict_token_freq = Counter()
    for c in conflicts.values():
        conflict_token_freq.update(c)
    # 条件熵：对每个冲突历史，用经验分布 p(target|hist) 计算 H
    entropies = []
    for h, c in conflicts.items():
        total = sum(c.values())
        p = [v / total for v in c.values()]
        entropies.append(-sum(pi * math.log2(pi) for pi in p if pi > 0))
    mean_ent = sum(entropies) / max(1, len(entropies))
    # 理论最低交叉熵（用经验分布自编码）= 条件熵；均匀基线 = log2(平均目标数)
    avg_targets = sum(len(c) for c in conflicts.values()) / max(1, len(conflicts))
    uniform_ce = math.log2(avg_targets) if avg_targets > 1 else 0.0
    # 受影响样本：重新扫一遍，命中冲突 hist 的样本
    conflict_hist_set = set(conflicts)
    for i, s in enumerate(samples):
        if any(h in conflict_hist_set for h, _, _, _ in supervised_positions(tok, s, protocol)):
            sample_hits.add(i)
    # 任务分布：按冲突历史Dominant task归属
    task_conflicts = Counter()
    for h, c in conflicts.items():
        dom = hist_tasks[h].most_common(1)[0][0] if hist_tasks[h] else "?"
        task_conflicts[dom] += 1
    # 真歧义：完整 prompt（不含 SEP/EOS）对应多个不同答案
    prompt_answers = defaultdict(set)
    for s in samples:
        prompt_answers[s["prompt"]].add(s["answer"])
    true_ambig = {p: sorted(a) for p, a in prompt_answers.items() if len(a) > 1}
    return {
        "protocol": protocol,
        "n_samples": len(samples),
        "n_supervised_positions": n_positions,
        "conflict_prefix_count": len(conflicts),
        "conflict_prefix_rate": round(len(conflicts) / max(1, n_positions), 6),
        "affected_supervised_positions": affected_positions,
        "affected_position_rate": round(affected_positions / max(1, n_positions), 6),
        "affected_samples": len(sample_hits),
        "affected_sample_rate": round(len(sample_hits) / max(1, len(samples)), 6),
        "conflict_token_frequency": dict(conflict_token_freq.most_common(20)),
        "task_conflict_distribution": dict(task_conflicts.most_common()),
        "conditional_entropy_bits_mean": round(mean_ent, 4),
        "theoretical_min_cross_entropy_bits": round(mean_ent, 4),
        "uniform_baseline_ce_bits": round(uniform_ce, 4),
        "true_prompt_ambiguity_count": len(true_ambig),
        "true_prompt_ambiguity_examples": list(true_ambig.items())[:10],
        "conflict_examples": [
            {"history": tok.decode(list(h)), "targets": dict(c), "tasks": dict(hist_tasks[h])}
            for h, c in list(conflicts.items())[:15]
        ],
    }


def main():
    tok = M2Tokenizer()
    samples = load_samples()
    out = {"status": "SUPERVISED_CONFLICT_AUDIT_DONE",
           "definition": "只统计 loss 权重>0 的预测位置；相同受监督历史对应不同目标 token 才算冲突",
           "n_samples_loaded": len(samples),
           "by_protocol": {p: analyze(p, samples, tok) for p in PROTOCOLS}}
    lg = out["by_protocol"]["legacy"]
    sp = out["by_protocol"]["sep"]
    out["sep_vs_legacy"] = {
        "conflict_prefix_reduction": lg["conflict_prefix_count"] - sp["conflict_prefix_count"],
        "affected_position_reduction": lg["affected_supervised_positions"] - sp["affected_supervised_positions"],
        "note": "SEP 使 answer 首 token 的受监督历史包含 SEP，结构性切断与 prompt 内部延续的共享",
    }
    (M3 / "M3_SUPERVISED_CONFLICT_AUDIT.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": out["status"], "n_samples": len(samples),
        "legacy": {"conflicts": lg["conflict_prefix_count"],
                   "affected_pos": lg["affected_supervised_positions"],
                   "affected_rate": lg["affected_position_rate"], "H_bits": lg["conditional_entropy_bits_mean"]},
        "sep": {"conflicts": sp["conflict_prefix_count"],
                "affected_pos": sp["affected_supervised_positions"],
                "affected_rate": sp["affected_position_rate"], "H_bits": sp["conditional_entropy_bits_mean"]},
        "reduction": out["sep_vs_legacy"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
