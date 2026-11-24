"""M2 P0：历史自回归监督前缀冲突扫描。

问题定义：
  历史（M1/C1）协议把训练序列写成 prompt_ids + answer_ids + [EOS]，答案生成的
  上下文就是 prompt_ids 本身。若某样本的完整 prompt P 同时也是另一条样本 prompt
  的真前缀，则在同一上下文 P 处，一条样本要求输出「答案首字符」，另一条要求输出
  「prompt 延续字符」，两者监督目标不同 → 前缀冲突 / 边界不可辨识。

SEP 协议：
  序列 = prompt_ids + [SEP] + answer_ids + [EOS]，答案生成上下文为 prompt_ids+[SEP]。
  SEP=101 不在 prompt 字符表(4..100)中，故不可能作为任何 prompt 的前缀出现，
  从结构上消除上述冲突。

本脚本用真实历史数据统计 legacy 冲突、核验 SEP 修复，并如实记录 SEP 无法解决的
「同 prompt 不同 answer」真实歧义。
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from tokenizer_m2 import M2Tokenizer, SEP, vocab_sha

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# 历史数据来源（相对 math_m2）
HISTORY_FILES = [
    ROOT / "math_m1" / "data" / "m1_train.jsonl",
    ROOT / "math_m1" / "data" / "m1_val.jsonl",
    ROOT / "math_m1" / "data" / "m1_sealed.jsonl",
    ROOT / "compute_c1" / "data" / "c1_train.jsonl",
    ROOT / "compute_c11" / "data" / "c11_train.jsonl",
]


def load_history():
    samples = []
    sources = {}
    for p in HISTORY_FILES:
        if not p.exists():
            sources[str(p.relative_to(ROOT))] = {"exists": False, "n": 0}
            continue
        rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        sources[str(p.relative_to(ROOT))] = {"exists": True, "n": len(rows)}
        for r in rows:
            prompt = r.get("prompt")
            answer = r.get("answer")
            if not isinstance(prompt, str) or answer is None:
                continue
            samples.append({"prompt": prompt, "answer": str(answer), "task": r.get("task", "?")})
    return samples, sources


def audit(samples, tok: M2Tokenizer):
    # prompt_ids -> 该 prompt 的所有答案首 token（真实歧义检测）
    prompt_answer_first = defaultdict(set)
    # 所有 prompt 的「真前缀」（长度 1..len-1）集合
    prompt_strict_prefixes = set()
    prompt_ids_list = []
    for s in samples:
        p = tuple(tok.encode(s["prompt"]))
        a = tok.encode(s["answer"])
        prompt_ids_list.append(p)
        if a:
            prompt_answer_first[p].add(a[0])
        for k in range(1, len(p)):
            prompt_strict_prefixes.add(p[:k])

    unique_prompts = set(prompt_ids_list)
    # legacy 冲突：完整 prompt 同时是别的 prompt 的真前缀，且答案首 token 与延续 token 分歧
    # 先收集每个前缀在 prompt 内部后续出现的 token 集合
    prefix_next_in_prompt = defaultdict(set)
    for p in unique_prompts:
        for k in range(1, len(p)):
            prefix_next_in_prompt[p[:k]].add(p[k])

    legacy_conflict_prompts = []
    legacy_boundary_unexplainable = []
    for p, firsts in prompt_answer_first.items():
        cont = prefix_next_in_prompt.get(p, set())
        if not cont:
            continue  # 不是任何 prompt 的真前缀 → legacy 下该边界无歧义
        # 冲突：答案首 token 与 prompt 内延续 token 存在不同取值
        disagree = firsts - cont
        if disagree:
            legacy_conflict_prompts.append({
                "prompt": tok.decode(list(p)),
                "answer_first_ids": sorted(firsts),
                "prompt_continuation_ids": sorted(cont),
            })
    # 真实歧义：同 prompt 多个不同答案首 token（SEP 也无法解决）
    for p, firsts in prompt_answer_first.items():
        if len(firsts) > 1:
            legacy_boundary_unexplainable.append({
                "prompt": tok.decode(list(p)),
                "answer_first_ids": sorted(firsts),
            })

    # SEP 核验：答案上下文 = prompt+[SEP]，SEP 不在字符表 → 不可能成为任何 prompt 前缀
    sep_ctx = {p + (SEP,) for p in unique_prompts}
    sep_overlap_with_prompt_prefix = len(sep_ctx & prompt_strict_prefixes)
    sep_overlap_with_unique_prompt = len(sep_ctx & unique_prompts)

    total = len(samples)
    n_conflict = len(legacy_conflict_prompts)
    n_true_ambig = len(legacy_boundary_unexplainable)
    report = {
        "status": "P0_PREFIX_CONFLICT_AUDIT_DONE",
        "tokenizer_sha256": vocab_sha(),
        "sep_token_id": SEP,
        "sources": {},  # 由 main 注入
        "total_samples": total,
        "unique_prompts": len(unique_prompts),
        "legacy_protocol": {
            "answer_context": "prompt_ids",
            "conflict_definition": "完整 prompt 同时是其他 prompt 真前缀，且答案首 token 与 prompt 延续 token 分歧",
            "conflict_prompt_count": n_conflict,
            "conflict_rate": round(n_conflict / max(1, len(prompt_answer_first)), 6),
            "examples": legacy_conflict_prompts[:20],
        },
        "sep_protocol": {
            "answer_context": "prompt_ids + [SEP]",
            "sep_in_char_vocab": False,
            "sep_overlap_with_prompt_strict_prefix": sep_overlap_with_prompt_prefix,
            "sep_overlap_with_full_prompt": sep_overlap_with_unique_prompt,
            "structural_conflict_eliminated": (
                sep_overlap_with_prompt_prefix == 0 and sep_overlap_with_unique_prompt == 0),
        },
        "true_ambiguity_sep_cannot_fix": {
            "definition": "完全相同的 prompt 对应多个不同答案首 token",
            "count": n_true_ambig,
            "examples": legacy_boundary_unexplainable[:20],
        },
        "verdict": (
            "SEP_STRUCTURALLY_FIXES_PREFIX_CONFLICT"
            if sep_overlap_with_prompt_prefix == 0 and sep_overlap_with_unique_prompt == 0
            else "REVIEW"),
    }
    return report


def main():
    tok = M2Tokenizer()
    samples, sources = load_history()
    report = audit(samples, tok)
    report["sources"] = sources
    out = HERE / "M2_PREFIX_CONFLICT_AUDIT.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "total_samples": report["total_samples"],
        "unique_prompts": report["unique_prompts"],
        "legacy_conflicts": report["legacy_protocol"]["conflict_prompt_count"],
        "legacy_conflict_rate": report["legacy_protocol"]["conflict_rate"],
        "sep_structural_conflict": report["sep_protocol"]["structural_conflict_eliminated"],
        "sep_overlap": report["sep_protocol"]["sep_overlap_with_prompt_strict_prefix"],
        "true_ambiguity": report["true_ambiguity_sep_cannot_fix"]["count"],
        "verdict": report["verdict"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
