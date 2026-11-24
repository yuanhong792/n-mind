"""M41 冻结数据独立审计与捷径基线（零梯度）。

独立重算 canonical，不复用数据生成器的审计逻辑。
输出 M41_DATA_AUDIT.json；捷径只在本审计脚本中计算，不进入模型推理。
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from tokenizer_m4 import M4Tokenizer
from reference_interpreter import canonical, reference_answer

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
MAX_LEN = 96
EXPECTED = {"train": 1392, "val": 464, "sealed": 464}


def load(name):
    return [json.loads(l) for l in (DATA / f"m4_{name}.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]


def assignments(prompt):
    return [(m.group(1), m.group(2)) for m in
            (re.fullmatch(r"([a-z])=([0-9])", line.strip())
             for line in prompt.splitlines()) if m]


def query_var(prompt):
    for line in prompt.splitlines():
        if line.strip().startswith("GET "):
            return line.strip().split()[-1]
    return None


def all_rows(splits, cf, gen):
    out = []
    for split, rows in splits.items():
        for r in rows:
            out.append((split, r))
    for r in gen:
        out.append(("generalization", r))
    for p in cf:
        for key in ("base", "counterfactual", "query_switch"):
            for r in p.get(key, []):
                out.append(("counterfactual", r))
    return out


def audit():
    tok = M4Tokenizer()
    splits = {k: load(k) for k in EXPECTED}
    cf = load("counterfactual")
    dist = load("distance")
    gen = load("generalization")
    checks = {}

    counts = {k: len(v) for k, v in splits.items()}
    checks["SPLIT_COUNTS"] = {"pass": counts == EXPECTED,
                              "observed": counts, "required": EXPECTED}

    ref_fail = []
    for split, rows in splits.items():
        for r in rows:
            try:
                exp = reference_answer(r["task"], r["prompt"])
            except Exception as exc:
                exp = f"ERROR:{exc}"
            if exp != r["answer"]:
                ref_fail.append({"split": split, "task": r["task"], "prompt": r["prompt"],
                                 "stored": r["answer"], "reference": exp})
    checks["REFERENCE_INTERPRETER_LABELS"] = {
        "pass": not ref_fail,
        "observed": {"n_checked": sum(len(v) for v in splits.values()),
                     "n_mismatch": len(ref_fail), "samples": ref_fail[:10]},
        "required": {"n_mismatch": 0},
    }

    train = splits["train"]
    checks["A5_REPEAT_COVERAGE"] = {
        "pass": sum(r["task"] == "A5_repeat" for r in train) >= 96,
        "observed": {"train_A5": sum(r["task"] == "A5_repeat" for r in train)},
        "required": {"train_A5_min": 96},
    }

    # 主绑定任务的查询位置是随机采样，只记录分布；
    # 任务书要求的“查询对象位置均衡”按冻结反事实设计的 first/middle/last 配对判断。
    qpos = defaultdict(Counter)
    for r in train + splits["val"] + splits["sealed"]:
        if r["task"] in ("B2_bind", "B4_bind", "B8_bind"):
            qpos[r["task"]][int(r["meta"]["query_pos"])] += 1
    cf_pos = Counter(p["query_position"] for p in cf)
    cf_pos_by_type = defaultdict(Counter)
    for p in cf:
        cf_pos_by_type[p["type"]][p["query_position"]] += 1
    cf_balanced = all(set(v) == {"first", "middle", "last"}
                      and max(v.values()) - min(v.values()) <= 1
                      for v in cf_pos_by_type.values()) and \
        (max(cf_pos.values()) - min(cf_pos.values()) <= 4)
    checks["QUERY_POSITION_BALANCE"] = {
        "pass": cf_balanced,
        "observed": {"bindings": {k: dict(v) for k, v in qpos.items()},
                     "counterfactual_global": dict(cf_pos),
                     "counterfactual_per_type": {k: dict(v)
                                                 for k, v in cf_pos_by_type.items()}},
        "required": "每类反事实 first/middle/last 近似均衡；主绑定任务分布仅作观察",
    }

    canons = {k: {canonical(r["prompt"]) for r in v} for k, v in splits.items()}
    canons["generalization"] = {canonical(r["prompt"]) for r in gen}
    overlap = {}
    names = list(canons)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap[f"{a}_vs_{b}"] = len(canons[a] & canons[b])
    checks["CANONICAL_SPLIT_ISOLATION"] = {
        "pass": all(v == 0 for v in overlap.values()),
        "observed": overlap, "required": "所有集合间 canonical 重合为 0",
    }

    rows = all_rows(splits, cf, gen)
    by_prompt = defaultdict(set)
    for _, r in rows:
        by_prompt[r["prompt"]].add(r["answer"])
    prompt_conflicts = {p: sorted(a) for p, a in by_prompt.items() if len(a) > 1}

    # 主分割/泛化集之间检查跨任务泄漏。
    # 反事实内部 B_counterfactual 与 B_query_switch 共用同一 prompt 是设计上的任务别名，
    # 但必须确认它与主分割无 prompt 重合。
    main_entries = ([(s, r) for s, rs in splits.items() for r in rs] +
                    [("generalization", r) for r in gen])
    by_prompt_task = defaultdict(set)
    for _, r in main_entries:
        by_prompt_task[r["prompt"]].add(r["task"])
    cross_task = {p: sorted(t) for p, t in by_prompt_task.items() if len(t) > 1}
    main_prompts = {r["prompt"] for _, r in main_entries}
    cf_prompts = {r["prompt"] for p in cf for key in ("base", "counterfactual", "query_switch")
                  for r in p.get(key, [])}
    cf_main_overlap = sorted(main_prompts & cf_prompts)

    checks["NO_PROMPT_ANSWER_CONFLICT"] = {
        "pass": not prompt_conflicts,
        "observed": {"n_conflicts": len(prompt_conflicts),
                     "samples": list(prompt_conflicts.items())[:5]},
        "required": {"n_conflicts": 0},
    }
    checks["NO_CROSS_TASK_SEMANTIC_LEAK"] = {
        "pass": not cross_task and not cf_main_overlap,
        "observed": {"n_cross_task_prompts": len(cross_task),
                     "samples": list(cross_task.items())[:5],
                     "n_cf_main_prompt_overlap": len(cf_main_overlap),
                     "cf_main_samples": cf_main_overlap[:5],
                     "note": "CF 内部 B_counterfactual/B_query_switch 别名按同一 pair_id 排除"},
        "required": {"n_cross_task_prompts": 0, "n_cf_main_prompt_overlap": 0},
    }

    cf_types = Counter(p["type"] for p in cf)
    pair_ids = [p["pair_id"] for p in cf]
    cf_groups = defaultdict(list)
    cf_fail = []
    for p in cf:
        typ, pid = p["type"], p["pair_id"]
        base, counter = p["base"][0], p["counterfactual"][0]
        if base["fact_group"] != pid or counter["fact_group"] != pid:
            cf_fail.append({"pair_id": pid, "reason": "fact_group_isolation"})
        if typ == "same_query_fact_change" and base["answer"] == counter["answer"]:
            cf_fail.append({"pair_id": pid, "reason": "same_query_answer_unchanged"})
        if typ == "unrelated_fact_change" and base["answer"] != counter["answer"]:
            cf_fail.append({"pair_id": pid, "reason": "unrelated_answer_changed"})
        if typ == "order_swap" and (base["answer"] != counter["answer"] or
                                    sorted(assignments(base["prompt"])) !=
                                    sorted(assignments(counter["prompt"]))):
            cf_fail.append({"pair_id": pid, "reason": "order_swap_semantics_changed"})
        if typ == "query_switch":
            if len(p["query_switch"]) != 1:
                cf_fail.append({"pair_id": pid, "reason": "query_switch_missing"})
            elif p["query_switch"][0]["answer"] != counter["answer"]:
                cf_fail.append({"pair_id": pid, "reason": "query_switch_answer_mismatch"})
            if query_var(base["prompt"]) == query_var(counter["prompt"]):
                cf_fail.append({"pair_id": pid, "reason": "query_var_not_switched"})
        cf_groups[typ].append(p["query_position"])
    pos_per_type = {t: dict(Counter(v)) for t, v in cf_groups.items()}
    per_type_balanced = all(set(v) == {"first", "middle", "last"} and
                            max(v.values()) - min(v.values()) <= 1
                            for v in pos_per_type.values())
    checks["COUNTERFACTUAL_FAMILY_ISOLATION"] = {
        "pass": len(set(pair_ids)) == len(pair_ids) == 400
                and cf_types == Counter({t: 100 for t in
                                         ("same_query_fact_change", "unrelated_fact_change",
                                          "query_switch", "order_swap")})
                and per_type_balanced and not cf_fail,
        "observed": {"n_pairs": len(cf), "unique_pair_ids": len(set(pair_ids)),
                     "types": dict(cf_types), "query_positions_per_type": pos_per_type,
                     "failures": cf_fail[:10]},
        "required": "400 唯一 pair_id；四类各 100；查询位置均衡；语义变化符合类型定义",
    }

    dist_by = defaultdict(list)
    for r in dist:
        dist_by[r["meta"]["pair_id"]].append(r)
    dist_fail = []
    expected_d = {0, 4, 8, 16, 32, 64, 128}
    for pid, rs in dist_by.items():
        ds = {int(r["meta"]["distance"]) for r in rs}
        answers = {r["answer"] for r in rs}
        queries = {query_var(r["prompt"]) for r in rs}
        if len(rs) != 7 or ds != expected_d or len(answers) != 1 or len(queries) != 1:
            dist_fail.append({"pair_id": pid, "n": len(rs), "distances": sorted(ds),
                              "answers": sorted(answers), "queries": sorted(queries)})
    checks["DISTANCE_PAIRED_IDS"] = {
        "pass": len(dist_by) == 60 and not dist_fail,
        "observed": {"n_pairs": len(dist_by), "n_rows": len(dist),
                     "failures": dist_fail[:10]},
        "required": {"n_pairs": 60, "distances": sorted(expected_d)},
    }

    max_len_by = {}
    len_fail = []
    for name, rows in splits.items():
        lens = [len(tok.encode_pair(r["prompt"], r["answer"])[0]) for r in rows]
        max_len_by[name] = max(lens)
        if max(lens) > MAX_LEN:
            len_fail.append({"split": name, "max_len": max(lens)})
    checks["NO_SILENT_TRUNCATION"] = {
        "pass": not len_fail,
        "observed": {"max_len": max_len_by, "limit": MAX_LEN, "failures": len_fail},
        "required": f"train/val/sealed 全部 <= {MAX_LEN}",
    }

    # 捷径基线：仅用于评估，不进入模型推理。
    eval_rows = splits["val"] + splits["sealed"]
    train_answers = Counter(r["answer"] for r in train)
    most_common = train_answers.most_common(1)[0][0]
    rng = random.Random(20261124)

    def predict(r, strategy):
        digits = re.findall(r"\d", r["prompt"])
        ass = assignments(r["prompt"])
        if strategy == "first_digit":
            return digits[0] if digits else None
        if strategy == "last_digit":
            return digits[-1] if digits else None
        if strategy.startswith("fixed_position_"):
            k = int(strategy.rsplit("_", 1)[1])
            return ass[k][1] if len(ass) > k else None
        if strategy == "nearest_assignment":
            q = query_var(r["prompt"])
            for k, v in reversed(ass):
                if k == q:
                    return v
            return None
        if strategy == "constant_most_common":
            return most_common
        if strategy == "random_digit":
            return str(rng.randrange(10))
        raise ValueError(strategy)

    baselines = {}
    for strategy in ("first_digit", "last_digit", "fixed_position_0",
                     "fixed_position_1", "fixed_position_2", "nearest_assignment",
                     "constant_most_common", "random_digit"):
        rr = random.Random(20261124)
        correct = usable = 0
        by_task = defaultdict(lambda: [0, 0])
        for r in eval_rows:
            rng_local = rr
            if strategy == "random_digit":
                pred = str(rng_local.randrange(10))
            else:
                pred = predict(r, strategy)
            if pred is None:
                continue
            usable += 1
            ok = int(pred == r["answer"])
            correct += ok
            by_task[r["task"]][0] += ok
            by_task[r["task"]][1] += 1
        baselines[strategy] = {
            "correct": correct, "usable": usable,
            "accuracy": round(correct / max(1, usable), 4),
            "coverage": round(usable / max(1, len(eval_rows)), 4),
            "per_task": {k: {"ok": v[0], "n": v[1],
                             "accuracy": round(v[0] / max(1, v[1]), 4)}
                         for k, v in sorted(by_task.items())},
        }

    status = "PASS" if all(v["pass"] for v in checks.values()) else "FAILED"
    out = {"status": status, "checks": checks, "shortcut_baselines": baselines,
           "note": "捷径基线仅用于识别数据集可被非神经规则利用的程度，不参与模型推理。"}
    p = HERE / "M41F_DATA_AUDIT_FULL.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[m41-data] status={status} 写出 {p}")
    return out


if __name__ == "__main__":
    audit()
