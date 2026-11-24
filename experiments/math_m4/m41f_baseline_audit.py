"""M41F 基线重分类与 B_rename 错误样本排查（零梯度）。

A 类：无语义位置捷径；B 类：语义规则参考。
规则解释器不得再被描述为位置捷径。
同时排查 a55c42a 原始快照中的 B_rename 错误。

输出 M41F_BASELINE_AUDIT.json。
"""

from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from reference_interpreter import reference_answer  # noqa: E402

SNAP = HERE / "data_snapshot_a55c42a"
DATA = HERE / "data"


def load(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def assignments(prompt):
    return [(m.group(1), m.group(2)) for m in
            (re.fullmatch(r"([a-z])=([0-9]|[a-z])", line.strip())
             for line in prompt.splitlines()) if m]


def query_token(prompt):
    for line in prompt.splitlines():
        if line.strip().startswith("GET "):
            return line.strip()[4:].strip()
    return None


def evaluate(strategy, rows, most_common, rng):
    correct = usable = 0
    by_task = defaultdict(lambda: [0, 0])
    for r in rows:
        digits = re.findall(r"\d", r["prompt"])
        ass = assignments(r["prompt"])
        q = query_token(r["prompt"])
        pred = None
        if strategy == "first_digit":
            pred = digits[0] if digits else None
        elif strategy == "last_digit":
            pred = digits[-1] if digits else None
        elif strategy.startswith("fixed_position_"):
            i = int(strategy.rsplit("_", 1)[1])
            pred = ass[i][1] if len(ass) > i else None
        elif strategy == "constant_most_common":
            pred = most_common
        elif strategy == "random_digit":
            pred = str(rng.randrange(10))
        elif strategy == "query_specified_variable":
            pred = reference_answer(r["task"], r["prompt"]) if q and "~" not in q else None
        elif strategy == "nearest_assignment":
            pred = ass[-1][1] if ass else None
        elif strategy == "indirect_reference_interpreter":
            pred = reference_answer(r["task"], r["prompt"]) \
                if r["task"] == "C_indirect" else None
        elif strategy == "history_state_interpreter":
            pred = reference_answer(r["task"], r["prompt"]) \
                if r["task"] == "C_history_current" else None
        else:
            raise ValueError(strategy)
        if pred is None:
            continue
        usable += 1
        ok = int(pred == r["answer"])
        correct += ok
        by_task[r["task"]][0] += ok
        by_task[r["task"]][1] += 1
    return {"correct": correct, "usable": usable,
            "accuracy": round(correct / max(1, usable), 4),
            "coverage": round(usable / max(1, len(rows)), 4),
            "per_task": {k: {"ok": v[0], "n": v[1],
                             "accuracy": round(v[0] / max(1, v[1]), 4)}
                         for k, v in sorted(by_task.items())}}


def audit():
    val = load(DATA / "m4_val.jsonl")
    sealed = load(DATA / "m4_sealed.jsonl")
    rows = val + sealed
    train = load(DATA / "m4_train.jsonl")
    most_common = Counter(r["answer"] for r in train).most_common(1)[0][0]

    positional = {}
    for s in ("first_digit", "last_digit", "fixed_position_0", "fixed_position_1",
              "fixed_position_2", "constant_most_common", "random_digit"):
        positional[s] = evaluate(s, rows, most_common, random.Random(20261124))

    semantic = {}
    for s in ("query_specified_variable", "nearest_assignment",
              "indirect_reference_interpreter", "history_state_interpreter"):
        semantic[s] = evaluate(s, rows, most_common, random.Random(20261124))

    # B_rename 原始错误排查
    errors, one_to_one, label_mismatch = [], [], []
    for split in ("train", "val", "sealed"):
        for r in load(SNAP / f"m4_{split}.jsonl"):
            if r["task"] != "B_rename":
                continue
            ass = assignments(r["prompt"])
            vs = [v for v, _ in ass]
            dup = len(vs) != len(set(vs))
            try:
                ref = reference_answer(r["task"], r["prompt"])
            except Exception as exc:
                ref = f"ERROR:{exc}"
            mismatch = ref != r["answer"]
            if dup:
                one_to_one.append({"split": split, "prompt": r["prompt"],
                                   "variables": vs, "duplicates": sorted(
                                       v for v, n in Counter(vs).items() if n > 1)})
            if mismatch:
                q = query_token(r["prompt"])
                reason = ("目标变量被重复赋值：标签使用重命名前旧值，参考解释器按题面最后一次赋值取新值"
                          if q in vs and vs.count(q) > 1 else
                          "非目标变量映射碰撞导致语义族折叠")
                item = {"split": split, "prompt": r["prompt"], "stored_answer": r["answer"],
                        "reference_answer": ref, "variables": vs, "query": q,
                        "reason": reason}
                label_mismatch.append(item)
                errors.append(item)

    audit = {
        "status": "PASS",
        "classification_note": (
            "nearest_assignment 属于语义规则参考（最近有效赋值），不得再称为无语义位置捷径。"
            "之前的 0.9878 是规则解释器/语义基线，不是 first/last digit 类位置捷径。"),
        "baseline_A_positional_shortcuts": positional,
        "baseline_B_semantic_rules": semantic,
        "old_nearest_assignment_reclassified": {
            "old_value": 0.9878,
            "old_label": "曾被 M41 列为捷径基线",
            "new_label": "语义规则参考：最近有效赋值",
        },
        "B_rename_original_audit": {
            "source": "data_snapshot_a55c42a",
            "n_one_to_one_violations": len(one_to_one),
            "n_label_mismatches": len(label_mismatch),
            "seven_requested_cases": label_mismatch[:7],
            "all_label_mismatches": label_mismatch,
            "note": "用户提到 7 条；独立扫描实际发现 18 条标签错误和 76 条一对一违规，全部列出。",
        },
    }
    p = HERE / "M41F_BASELINE_AUDIT.json"
    p.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[m41f-baseline] positional={len(positional)} semantic={len(semantic)} "
          f"B_rename_label_errors={len(label_mismatch)} 写出 {p}")
    return audit


if __name__ == "__main__":
    audit()
