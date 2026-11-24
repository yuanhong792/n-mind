"""M3.1 P0：反事实数据捷径审计（零梯度）。

任务书指出 M3 反事实始终查询第一个变量，模型可直接输出首个数字得分。
本脚本读取 M3 的 400 对反事实，量化：
  - 查询位置分布（首/中/末）。
  - 首变量捷径命中率（不读 GET，直接输出第一行数字）。
  - 查询切换覆盖率。
  - 同查询事实修改类型（M3 是交换赋值，非单变量修改）。
  - 无关事实修改覆盖（M3 是否提供）。
并定义 M4 必须满足的均衡判据。

运行：python counterfactual_audit.py
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
M3 = HERE.parent / "math_m3"

ASSIGN = re.compile(r"^([a-z])=(\d+)$")
GET = re.compile(r"^GET\s+([a-z])$")


def parse_prompt(prompt: str):
    lines = prompt.strip().splitlines()
    assigns, query = [], None
    for line in lines:
        m = ASSIGN.match(line.strip())
        if m:
            assigns.append((m.group(1), m.group(2)))
            continue
        g = GET.match(line.strip())
        if g:
            query = g.group(1)
    return assigns, query


def first_value_shortcut(prompt: str) -> str | None:
    assigns, _ = parse_prompt(prompt)
    return assigns[0][1] if assigns else None


def main():
    rows = [json.loads(l) for l in (M3 / "data" / "m3_counterfactual.jsonl")
            .read_text(encoding="utf-8").splitlines() if l.strip()]
    stats = {"n_pairs": len(rows), "query_position": Counter(),
             "base_first_shortcut_ok": 0, "cf_first_shortcut_ok": 0,
             "both_first_shortcut_ok": 0, "query_switch_pairs": 0,
             "facts_changed": Counter(), "target_changed_when_query_same": 0,
             "unrelated_fact_change_pairs": 0, "order_swap_pairs": 0,
             "n_vars": Counter(), "query_var_matches_first": 0}
    for p in rows:
        base = p["base"][0]
        cf = p["counterfactual"][0]
        ba, bq = parse_prompt(base["prompt"])
        ca, cq = parse_prompt(cf["prompt"])
        if not ba or not bq:
            continue
        stats["n_vars"][len(ba)] += 1
        idx = [v for v, _ in ba].index(bq) if bq in [v for v, _ in ba] else -1
        pos = "first" if idx == 0 else ("last" if idx == len(ba) - 1 else "middle")
        stats["query_position"][pos] += 1
        if idx == 0:
            stats["query_var_matches_first"] += 1
        if bq == cq:
            bmap, cmap = dict(ba), dict(ca)
            if bmap.get(bq) != cmap.get(cq):
                stats["target_changed_when_query_same"] += 1
            changed = [v for v in set(bmap) | set(cmap) if bmap.get(v) != cmap.get(v)]
            stats["facts_changed"][len(changed)] += 1
            if len(changed) == 1 and changed[0] != bq:
                stats["unrelated_fact_change_pairs"] += 1
            if sorted(changed) == sorted(bmap.keys()) or len(changed) == len(bmap):
                stats["order_swap_pairs"] += 1
        else:
            stats["query_switch_pairs"] += 1
        # 首变量捷径：只看第一个赋值数字是否与答案一致
        bshort = first_value_shortcut(base["prompt"])
        cshort = first_value_shortcut(cf["prompt"])
        b_ok = bshort == base["answer"]
        c_ok = cshort == cf["answer"]
        stats["base_first_shortcut_ok"] += int(b_ok)
        stats["cf_first_shortcut_ok"] += int(c_ok)
        stats["both_first_shortcut_ok"] += int(b_ok and c_ok)

    n = max(1, stats["n_pairs"])
    out = {
        "status": "COMPLETE", "target": "M3 counterfactual shortcut audit",
        "stats": {k: (dict(v) if isinstance(v, Counter) else v) for k, v in stats.items()},
        "rates": {
            "query_first_ratio": round(stats["query_var_matches_first"] / n, 4),
            "base_first_value_shortcut": round(stats["base_first_shortcut_ok"] / n, 4),
            "cf_first_value_shortcut": round(stats["cf_first_shortcut_ok"] / n, 4),
            "pair_first_value_shortcut": round(stats["both_first_shortcut_ok"] / n, 4),
            "query_switch_coverage": round(stats["query_switch_pairs"] / n, 4),
            "unrelated_fact_change_coverage": round(stats["unrelated_fact_change_pairs"] / n, 4),
        },
        "m4_requirement": {
            "query_position_balance": "first/middle/last 每类占比 >= 0.25",
            "same_query_fact_change": "目标变量单值修改必须覆盖",
            "unrelated_fact_invariance": "必须覆盖",
            "query_switch": "必须覆盖且单独统计",
            "order_swap": "必须覆盖且保持语义结果",
            "pair_rule": "配对两条都正确才计通过",
        },
    }
    p = HERE / "M31_COUNTERFACTUAL_AUDIT.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[cf] query_position={dict(stats['query_position'])} "
          f"first_ratio={out['rates']['query_first_ratio']} "
          f"pair_shortcut={out['rates']['pair_first_value_shortcut']}")
    print(f"[cf] 写出 {p}")


if __name__ == "__main__":
    main()
