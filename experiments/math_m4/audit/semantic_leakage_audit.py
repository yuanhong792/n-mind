"""M3.1 P0：数据分布与规范化语义泄漏审计（零梯度）。

M3 仅用 fact_group 字符串判断语义隔离，任务书指出这不充分：
不同任务名、查询对象、变量交换、输入排列可能产生同一语义。
本脚本独立构造 canonical semantic key（排序变量赋值 + 查询对象），
检查 train/val/sealed/generalization 之间的：
  Prompt 重合 / Token 序列重合 / 规范化语义重合 / 反事实事实族重合 /
  查询位置分布 / 核心任务训练覆盖 / 重复样本率。

运行：python semantic_leakage_audit.py
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
M3 = HERE.parent / "math_m3"
ASSIGN = re.compile(r"^([a-z])=(\d+)$")
GET = re.compile(r"^GET\s+([a-z])$")

CORE_TASKS = ["A1_copy1", "A2_copy2", "A3_copyN", "A5_repeat", "A4_reverse", "A6_index",
              "B2_bind", "B4_bind", "B8_bind", "B_counterfactual", "B_query_switch", "B_rename",
              "C_reassign", "C_history_current", "C_indirect", "D_distance", "D_interference"]


def canonical(prompt: str) -> str:
    """规范化语义键：变量按名排序取最后赋值，附查询对象；非赋值任务用原文。"""
    assigns: dict[str, str] = {}
    query = None
    matched = 0
    for line in prompt.strip().splitlines():
        s = line.strip()
        m = ASSIGN.match(s)
        if m:
            assigns[m.group(1)] = m.group(2)
            matched += 1
            continue
        g = GET.match(s)
        if g:
            query = g.group(1)
            matched += 1
    if matched == 0:
        return "RAW|" + prompt
    facts = "|".join(f"{k}={v}" for k, v in sorted(assigns.items()))
    return f"{facts}|GET:{query or '-'}"


def load(split: str):
    p = M3 / "data" / f"m3_{split}.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    splits = {s: load(s) for s in ("train", "val", "sealed", "generalization")}
    summary = {"status": "COMPLETE", "n_rows": {s: len(v) for s, v in splits.items()},
               "splits": {}, "pairwise": {}, "task_coverage": {},
               "query_position_distribution": {}, "duplicate_rate": {}}
    keys = {}
    for s, rows in splits.items():
        prompts = [r["prompt"] for r in rows]
        toks = [tuple(ord(c) for c in p) for p in prompts]
        cans = [canonical(p) for p in prompts]
        keys[s] = {"prompt": set(prompts), "token": set(toks), "canonical": set(cans)}
        summary["splits"][s] = {
            "unique_prompt": len(set(prompts)), "unique_token_seq": len(set(toks)),
            "unique_canonical": len(set(cans)),
            "duplicate_prompt_rate": round(1 - len(set(prompts)) / max(1, len(prompts)), 4),
            "duplicate_canonical_rate": round(1 - len(set(cans)) / max(1, len(cans)), 4),
            "fact_group_unique": len({r.get("fact_group", "") for r in rows}),
        }
        counts = Counter(r["task"] for r in rows)
        summary["task_coverage"][s] = dict(counts)
        qpos = Counter()
        for r in rows:
            assigns, q = [], None
            for line in r["prompt"].strip().splitlines():
                m = ASSIGN.match(line.strip())
                if m:
                    assigns.append(m.group(1))
                g = GET.match(line.strip())
                if g:
                    q = g.group(1)
            if q and q in assigns:
                i = assigns.index(q)
                qpos["first" if i == 0 else ("last" if i == len(assigns) - 1 else "middle")] += 1
        summary["query_position_distribution"][s] = dict(qpos)

    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            summary["pairwise"][f"{a}_vs_{b}"] = {
                k: len(keys[a][k] & keys[b][k]) for k in ("prompt", "token", "canonical")}

    # 反事实事实族：canonical 相同但 fact_group 不同 → 字符串隔离是假象
    cf = [json.loads(l) for l in (M3 / "data" / "m3_counterfactual.jsonl")
          .read_text(encoding="utf-8").splitlines() if l.strip()]
    fg_by_can = defaultdict(set)
    for p in cf:
        for r in p["base"] + p["counterfactual"]:
            fg_by_can[canonical(r["prompt"])].add(r.get("fact_group", ""))
    summary["counterfactual_fact_family"] = {
        "n_canonical": len(fg_by_can),
        "n_canonical_with_multiple_fact_groups": sum(1 for v in fg_by_can.values() if len(v) > 1),
        "note": "同一规范化事实族若对应多个 fact_group 字符串，说明字符串隔离不可作为语义隔离证据",
    }

    missing = [t for t in CORE_TASKS if summary["task_coverage"]["train"].get(t, 0) == 0]
    summary["core_tasks_missing_in_train"] = missing
    p = HERE / "M31_LEAKAGE_AUDIT.json"
    p.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[leak] missing_in_train={missing}")
    print(f"[leak] canonical overlap train/val={summary['pairwise']['train_vs_val']['canonical']} "
          f"train/sealed={summary['pairwise']['train_vs_sealed']['canonical']}")
    print(f"[leak] 写出 {p}")


if __name__ == "__main__":
    main()
