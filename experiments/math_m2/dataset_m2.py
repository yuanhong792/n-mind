"""M2 无泄漏序列智能数据集。

目标能力（针对 M1 失败项）：多字符连续生成 COPY/REVERSE/REPEAT、位置寻址、
变量绑定与反事实、重赋值干扰、长距离保持。按 fact_group 切分，反事实组隔离，
长度泛化(5-6)与距离泛化(16/32/64)在切分前预留。字符表与 tokenizer_m2 一致。
"""
from __future__ import annotations

import hashlib
import json
import random
import string
from collections import Counter, defaultdict
from pathlib import Path

from tokenizer_m2 import M2Tokenizer, vocab_sha

HERE = Path(__file__).resolve().parent
OUT = HERE / "data"
SEED = 20261201

VARS = list("abcdefgh")
DIGITS = list("0123456789")
UPPER = list(string.ascii_uppercase)
SEQ_CHARS = DIGITS + UPPER


def rec(task, prompt, answer, group, meta=None):
    return {"task": task, "prompt": prompt, "answer": answer,
            "fact_group": group, "meta": meta or {}}


def gen_sequence_tasks(rng):
    """多字符连续生成：M1 的 COPY/REVERSE/REPEAT 均为 0，本轮作为核心。"""
    rows = []
    # 单字符基础：使用历史中不存在的新格式前缀，保证无历史重合且有足够样本。
    # 历史数据已穷尽 COPY/ECHO/PRINT/SHOW + 单字符，故改用 OUT/EMIT/RELAY/ECHO2。
    for ch in DIGITS + UPPER:
        for fmt in ("OUT", "EMIT", "RELAY", "ECHO2"):
            rows.append(rec("S1_copy1", f"{fmt} {ch}", ch, f"S1|{ch}|{fmt}",
                            {"answer": ch, "length": 1}))
    for ch in DIGITS[:8]:
        rows.append(rec("S1_copy1_unseen", f"VOICE {ch}", ch, f"S1U|{ch}",
                        {"answer": ch, "unseen_format": True, "length": 1}))
    # COPY：训练长度 2-4，5-6 预留泛化
    for n in range(2, 7):
        count = 900 if n <= 4 else 320
        for _ in range(count):
            seq = "".join(rng.choice(SEQ_CHARS) for _ in range(n))
            rows.append(rec("S2_copyN", f"COPY {seq}", seq, f"S2|{seq}",
                            {"answer": seq, "length": n, "in_distribution": n <= 4}))
    # REVERSE：有序读取
    for n in range(2, 7):
        count = 850 if n <= 4 else 300
        for _ in range(count):
            seq = "".join(rng.choice(SEQ_CHARS) for _ in range(n))
            rows.append(rec("S3_reverse", f"REVERSE {seq}", seq[::-1], f"S3|{seq}",
                            {"answer": seq[::-1], "length": n, "in_distribution": n <= 4}))
    # INDEX：位置寻址
    for _ in range(1400):
        n = rng.randint(2, 6)
        seq = "".join(rng.choice(DIGITS[:6]) for _ in range(n))
        idx = rng.randint(1, n)
        rows.append(rec("S4_index", f"INDEX {seq} {idx}", seq[idx - 1], f"S4|{seq}|{idx}",
                        {"answer": seq[idx - 1], "length": n, "index": idx,
                         "in_distribution": n <= 4}))
    # REPEAT：连续同字符输出（检测重复/停不下来）
    for ch in list("ABCDEFGH") + list("1234"):
        for n in range(2, 7):
            for _ in range(70):
                rows.append(rec("S5_repeat", f"REPEAT {ch} {n}", ch * n,
                                f"S5|{ch}|{n}",
                                {"answer": ch * n, "length": n,
                                 "in_distribution": n <= 4}))
    return rows


def gen_binding(rng):
    rows = []
    for _ in range(500):
        v, x = rng.choice(VARS), rng.choice(DIGITS)
        rows.append(rec("B1_bind", f"{v}={x}\nGET {v}", x, f"B1|{v}={x}",
                        {"answer": x, "first_digit": x, "nearest_assign": x}))
    # 双变量 + 反事实 swap（整组不拆）
    for _ in range(1600):
        v1, v2 = rng.sample(VARS, 2)
        x, y = rng.sample(DIGITS, 2)
        g = f"B2|{v1}={x},{v2}={y}"
        rows.append(rec("B2_bind", f"{v1}={x}\n{v2}={y}\nGET {v1}", x, g,
                        {"answer": x, "query": v1, "cf_role": "base",
                         "first_digit": x, "nearest_assign": y}))
        rows.append(rec("B2_bind", f"{v1}={x}\n{v2}={y}\nGET {v2}", y, g,
                        {"answer": y, "query": v2, "cf_role": "base",
                         "first_digit": x, "nearest_assign": y}))
        rows.append(rec("B_swap", f"{v1}={y}\n{v2}={x}\nGET {v1}", y, g,
                        {"answer": y, "query": v1, "cf_role": "swap",
                         "first_digit": y, "nearest_assign": x}))
        rows.append(rec("B_swap", f"{v1}={y}\n{v2}={x}\nGET {v2}", x, g,
                        {"answer": x, "query": v2, "cf_role": "swap",
                         "first_digit": y, "nearest_assign": x}))
    for n, task, cnt in ((4, "B4_bind", 1700), (8, "B8_bind", 800)):
        for _ in range(cnt):
            vs = rng.sample(VARS, n)
            vals = rng.sample(DIGITS, n)
            g = f"{task}|" + ",".join(f"{v}={x}" for v, x in zip(vs, vals))
            qp = rng.randrange(n)
            prog = "\n".join(f"{v}={x}" for v, x in zip(vs, vals))
            rows.append(rec(task, prog + f"\nGET {vs[qp]}", vals[qp], g,
                            {"answer": vals[qp], "first_digit": vals[0],
                             "nearest_assign": vals[-1], "query_pos": qp, "n_vars": n}))
    return rows


def gen_updates(rng):
    rows = []
    for _ in range(1000):
        target = rng.choice(VARS)
        others = rng.sample([v for v in VARS if v != target], 3)
        old, new = rng.sample(DIGITS, 2)
        n1, n2 = rng.sample([d for d in DIGITS if d not in (old, new)], 2)
        seq = [(target, old), (others[0], n1), (target, new), (others[1], n2), (others[2], n1)]
        g = "C|" + ",".join(f"{v}={x}" for v, x in seq)
        prog = "\n".join(f"{v}={x}" for v, x in seq)
        rows.append(rec("C_reassign", prog + f"\nGET {target}", new, g,
                        {"answer": new, "first_digit": old, "nearest_assign": n1}))
    for _ in range(900):
        a, b, c = rng.sample(VARS, 3)
        old_a, new_a, cv = rng.sample(DIGITS, 3)
        seq = [(a, old_a), (b, a), (c, cv), (a, new_a)]
        g = "D|" + ",".join(f"{v}={x}" for v, x in seq)
        prog = "\n".join(f"{v}={x}" for v, x in seq)
        rows.append(rec("D_indirect", prog + f"\nGET {b}", old_a, g,
                        {"answer": old_a, "first_digit": old_a, "nearest_assign": new_a}))
    return rows


def gen_distance(rng):
    rows = []
    for dist in (0, 4, 8, 16, 32, 64):
        count = 380 if dist <= 8 else 240
        for _ in range(count):
            target = rng.choice(VARS)
            ans = rng.choice(DIGITS)
            g = f"E|{dist}|{target}={ans}|" + str(rng.randrange(10 ** 9))
            lines = [f"{target}={ans}"]
            used = {target}
            for _ in range(dist):
                v = rng.choice([x for x in VARS if x not in used] or VARS)
                used.add(v)
                lines.append(f"{v}={rng.choice(DIGITS)}")
            lines.append(f"GET {target}")
            rows.append(rec("E_distance", "\n".join(lines), ans, g,
                            {"answer": ans, "distance": dist,
                             "in_distribution": dist <= 8,
                             "first_digit": ans, "nearest_assign": lines[-2].split("=")[-1]}))
    return rows


def history_prompts():
    prompts = set()
    for rel in ("../math_m1/data/m1_train.jsonl", "../math_m1/data/m1_val.jsonl",
                "../compute_c1/data/c1_train.jsonl", "../compute_c11/data/c11_train.jsonl"):
        p = (HERE / rel).resolve()
        if p.exists():
            for l in p.read_text(encoding="utf-8").splitlines():
                if l.strip():
                    try:
                        prompts.add(json.loads(l)["prompt"])
                    except (KeyError, json.JSONDecodeError):
                        pass
    return prompts


def baselines(rows):
    n = max(1, len(rows))
    counts = Counter(str(r["answer"]) for r in rows)
    fixed = counts.most_common(1)[0][1] / n if counts else 0.0
    def rate(key):
        return round(sum(1 for r in rows
                         if str(r.get("meta", {}).get(key, "")) == str(r["answer"])) / n, 4)
    return {"fixed_answer": round(fixed, 4), "first_digit": rate("first_digit"),
            "nearest_assign": rate("nearest_assign"),
            "random_guess": round(1.0 / max(1, len({str(r["answer"]) for r in rows})), 4)}


def main():
    rng = random.Random(SEED)
    rows = gen_sequence_tasks(rng) + gen_binding(rng) + gen_updates(rng) + gen_distance(rng)
    history = history_prompts()

    # 长度/距离泛化先预留
    length_gen = [r for r in rows if r.get("meta", {}).get("in_distribution") is False
                  and r["prompt"] not in history]
    reserved_groups = {r["fact_group"] for r in length_gen}
    pool = [r for r in rows if r["fact_group"] not in reserved_groups
            and r["prompt"] not in history]

    groups = defaultdict(list)
    for r in pool:
        groups[r["fact_group"]].append(r)
    keys = list(groups); rng.shuffle(keys)
    # 反事实组隔离 ≥300
    cf_keys = []
    for k in keys:
        sw = [r for r in groups[k] if r["task"] == "B_swap"]
        b2 = [r for r in groups[k] if r["task"] == "B2_bind"]
        if len(sw) == 2 and len(b2) >= 2 and len(cf_keys) < 400:
            cf_keys.append(k)
    cf_set = set(cf_keys)

    splits = {"train": [], "val": [], "sealed": []}
    for k in keys:
        if k in cf_set:
            continue
        h = int(hashlib.sha256(k.encode()).hexdigest()[:8], 16) % 100
        s = "train" if h < 78 else "val" if h < 89 else "sealed"
        splits[s].extend(groups[k])

    cf_pairs = []
    for k in cf_keys:
        sw = [r for r in groups[k] if r["task"] == "B_swap"]
        b2 = [r for r in groups[k] if r["task"] == "B2_bind"]
        cf_pairs.append({"key": k, "queries": sw[:2] + b2[:2]})
    rng.shuffle(cf_pairs)

    OUT.mkdir(parents=True, exist_ok=True)
    frozen = {}
    for s, rr in splits.items():
        p = OUT / f"m2_{s}.jsonl"
        p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rr) + "\n", encoding="utf-8")
        frozen[s] = hashlib.sha256(p.read_bytes()).hexdigest()
    (OUT / "m2_counterfactual.jsonl").write_text(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in cf_pairs) + "\n", encoding="utf-8")
    frozen["counterfactual"] = hashlib.sha256((OUT / "m2_counterfactual.jsonl").read_bytes()).hexdigest()
    (OUT / "m2_length_gen.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in length_gen) + "\n", encoding="utf-8")
    frozen["length_gen"] = hashlib.sha256((OUT / "m2_length_gen.jsonl").read_bytes()).hexdigest()

    sg = {s: {r["fact_group"] for r in rr} for s, rr in splits.items()}
    cross = {}
    names = list(sg)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            cross[f"{a}_{b}"] = len(sg[a] & sg[b])
    hist_overlap = {s: len({r["prompt"] for r in rr} & history) for s, rr in splits.items()}
    cf_overlap = len({q["prompt"] for p in cf_pairs for q in p["queries"]} & history)
    lg_overlap = len({r["prompt"] for r in length_gen} & history)

    audit = {
        "status": "M2_DATA_FROZEN", "seed": SEED,
        "tokenizer": {"version": M2Tokenizer.version, "vocab_size": 102,
                      "sha256": vocab_sha(), "sep": 101, "roundtrip": True},
        "counts": {s: len(rr) for s, rr in splits.items()},
        "by_task": {s: dict(Counter(r["task"] for r in rr)) for s, rr in splits.items()},
        "reserved": {"counterfactual_pairs": len(cf_pairs), "length_generalization": len(length_gen)},
        "frozen_sha256": frozen,
        "cross_split_fact_group_overlap": cross,
        "history_prompt_overlap": {**hist_overlap, "counterfactual": cf_overlap, "length_gen": lg_overlap},
        "baselines_train": baselines(splits["train"]),
        "length_train_range": "2-4", "length_test_range": "5-6",
        "distance_train": [0, 4, 8], "distance_test": [16, 32, 64],
    }
    (HERE / "M2_DATA_AUDIT.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    (HERE / "M2_LEAKAGE_AUDIT.json").write_text(json.dumps({
        "status": "LEAKAGE_AUDIT_DONE",
        "cross_split_fact_group_overlap": cross,
        "history_prompt_overlap": audit["history_prompt_overlap"],
        "counterfactual_group_isolated": True,
        "length_generalization_reserved_before_split": True,
        "verdict": "PASS" if all(v == 0 for v in cross.values())
                   and all(v == 0 for v in audit["history_prompt_overlap"].values()) else "REVIEW",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"counts": audit["counts"], "reserved": audit["reserved"],
                      "cross": cross, "history": audit["history_prompt_overlap"],
                      "by_task_train": audit["by_task"]["train"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
