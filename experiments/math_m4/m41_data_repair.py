"""M41 数据最小修复：仅修复退化的 same_query_fact_change 反事实配对。

问题：原生成器用 (old + 1 + i) % 10，当 i%10=9 时新值等于旧值，
配对没有发生目标事实变化。修复只在反事实集合中改这 10 条，
不修改 train/val/sealed/distance/generalization，不扩大预算，不改变任务。
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
PATH = HERE / "data" / "m4_counterfactual.jsonl"


def query_var(prompt):
    for line in prompt.splitlines():
        if line.strip().startswith("GET "):
            return line.strip().split()[-1]
    return None


def canonical(prompt):
    assigns, query, matched = {}, None, 0
    for line in prompt.strip().splitlines():
        s = line.strip()
        m = re.fullmatch(r"([a-z])=([0-9])", s)
        if m:
            assigns[m.group(1)] = m.group(2)
            matched += 1
        elif s.startswith("GET "):
            query = s.split()[-1]
            matched += 1
    if matched == 0:
        return "RAW|" + prompt
    return "|".join(f"{k}={v}" for k, v in sorted(assigns.items())) + \
        f"|GET:{query or '-'}"


def main():
    raw = PATH.read_bytes()
    before_sha = hashlib.sha256(raw).hexdigest()
    rows = [json.loads(l) for l in raw.decode("utf-8").splitlines() if l.strip()]
    repaired = []
    seen = set()
    for p in rows:
        if p["type"] != "same_query_fact_change":
            continue
        base, counter = p["base"][0], p["counterfactual"][0]
        if base["answer"] != counter["answer"]:
            continue
        i = int(p["pair_id"].rsplit("_", 1)[1])
        q = query_var(base["prompt"])
        old = base["answer"]
        new = str((int(old) + 1 + (i % 9)) % 10)
        assert q and new != old, (p["pair_id"], q, old, new)
        lines = []
        hits = 0
        for line in base["prompt"].splitlines():
            m = re.fullmatch(r"([a-z])=([0-9])", line.strip())
            if m and m.group(1) == q:
                lines.append(f"{q}={new}")
                hits += 1
            else:
                lines.append(line)
        assert hits == 1, (p["pair_id"], hits)
        before = {"prompt": counter["prompt"], "answer": counter["answer"],
                  "canonical": counter["canonical"]}
        counter["prompt"] = "\n".join(lines)
        counter["answer"] = new
        counter["canonical"] = canonical(counter["prompt"])
        counter["meta"]["m41_repair"] = {
            "reason": "new_value_equals_old_value",
            "old_answer": old, "new_answer": new,
        }
        repaired.append({"pair_id": p["pair_id"], "query": q, "before": before,
                         "after": {"prompt": counter["prompt"], "answer": new,
                                   "canonical": counter["canonical"]}})
        seen.add(p["pair_id"])

    out_text = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n"
    PATH.write_text(out_text, encoding="utf-8")
    after_sha = hashlib.sha256(PATH.read_bytes()).hexdigest()
    audit = {
        "status": "COMPLETE",
        "file": str(PATH),
        "before_sha256": before_sha,
        "after_sha256": after_sha,
        "n_rows": len(rows),
        "n_repaired": len(repaired),
        "scope": "仅 m4_counterfactual.jsonl 中退化的 same_query_fact_change 配对",
        "untouched": ["m4_train.jsonl", "m4_val.jsonl", "m4_sealed.jsonl",
                      "m4_distance.jsonl", "m4_generalization.jsonl"],
        "repaired_pairs": repaired,
    }
    p = HERE / "M41_DATA_REPAIR_AUDIT.json"
    p.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[m41-data-repair] repaired={len(repaired)} 写出 {p}")


if __name__ == "__main__":
    main()
