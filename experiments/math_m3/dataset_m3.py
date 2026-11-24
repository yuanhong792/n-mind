"""M3 计算能力基准数据（P8 + P0.4 修复）。

任务组：
  A 连续符号生成：单/双/三四字符复制、重复、反转、位置提取
  B 关联计算：双/四变量绑定、查询对象切换(独立)、属性交换(真反事实)、变量重命名
  C 状态更新：重赋值、历史值/当前值显式区分、多次更新、间接引用
  D 信息保持：严格配对距离(相同事实/相同查询/不同距离)、无关干扰、相似变量干扰、容量
  E 低成本学习：通过 curriculum 的样本子集实现（不在数据层重复）

P0.4 修复：
- 真反事实：查询对象与查询 token 完全相同，只交换事实赋值；标签随事实更新。
- 查询切换作为独立任务 B_query_switch（改变查询对象，事实不变）。
- 距离严格配对：同一 (target, ans) 在不同距离下各生成一条；干扰行不重新赋值 target。
- 历史/当前值：用 Cur= 与 Old= 显式前缀区分；读取 OLD x 返回历史快照，GET x 返回当前值。
无泄漏：按 fact_group 切分；反事实组隔离；泛化预留；历史 prompt 排除。
"""
from __future__ import annotations

import hashlib
import json
import random
import string
from collections import Counter, defaultdict
from pathlib import Path

from tokenizer_m3 import M3Tokenizer, vocab_sha

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


def gen_A(rng):
    rows = []
    # 单字符（新格式前缀，避免历史重合）
    for ch in DIGITS + UPPER:
        for fmt in ("OUT", "EMIT", "RELAY", "SHOW2"):
            rows.append(rec("A1_copy1", f"{fmt} {ch}", ch, f"A1|{ch}|{fmt}",
                            {"answer": ch, "length": 1, "in_dist": True}))
    # 双字符
    for _ in range(600):
        s = "".join(rng.choice(SEQ_CHARS) for _ in range(2))
        rows.append(rec("A2_copy2", f"COPY {s}", s, f"A2|{s}",
                        {"answer": s, "length": 2, "in_dist": True}))
    # 三到四字符（训练），五六字符（泛化预留）
    for n in range(3, 7):
        count = 900 if n <= 4 else 320
        for _ in range(count):
            s = "".join(rng.choice(SEQ_CHARS) for _ in range(n))
            rows.append(rec("A3_copyN", f"COPY {s}", s, f"A3|{s}",
                            {"answer": s, "length": n, "in_dist": n <= 4}))
    # 反转
    for n in range(2, 7):
        count = 850 if n <= 4 else 300
        for _ in range(count):
            s = "".join(rng.choice(SEQ_CHARS) for _ in range(n))
            rows.append(rec("A4_reverse", f"REVERSE {s}", s[::-1], f"A4|{s}",
                            {"answer": s[::-1], "length": n, "in_dist": n <= 4}))
    # 重复
    for ch in list("ABCDEFGH") + list("1234"):
        for n in range(2, 7):
            for _ in range(60):
                rows.append(rec("A5_repeat", f"REPEAT {ch} {n}", ch * n,
                                f"A5|{ch}|{n}", {"answer": ch * n, "length": n,
                                                 "in_dist": n <= 4}))
    # 位置提取
    for _ in range(1200):
        n = rng.randint(2, 6)
        s = "".join(rng.choice(DIGITS[:6]) for _ in range(n))
        idx = rng.randint(1, n)
        rows.append(rec("A6_index", f"INDEX {s} {idx}", s[idx - 1],
                        f"A6|{s}|{idx}", {"answer": s[idx - 1], "length": n,
                                         "index": idx, "in_dist": n <= 4}))
    return rows


def gen_B(rng):
    rows = []
    # 双变量 + 真反事实 + 独立查询切换，整组不拆
    for _ in range(1500):
        v1, v2 = rng.sample(VARS, 2)
        x, y = rng.sample(DIGITS, 2)
        g = f"B2|{v1}={x},{v2}={y}"
        base = f"{v1}={x}\n{v2}={y}"
        # 正常绑定
        rows.append(rec("B2_bind", base + f"\nGET {v1}", x, g,
                        {"answer": x, "query": v1, "role": "base",
                         "first_digit": x, "nearest": y}))
        rows.append(rec("B2_bind", base + f"\nGET {v2}", y, g,
                        {"answer": y, "query": v2, "role": "base",
                         "first_digit": x, "nearest": y}))
        # 查询切换：事实不变，只换查询对象（独立测试）
        rows.append(rec("B_query_switch", base + f"\nGET {v2}", y, g,
                        {"answer": y, "query": v2, "role": "query_switch",
                         "first_digit": x, "nearest": y}))
        # 真反事实：查询对象完全相同(GET v1)，只交换事实；标签随事实更新
        cf = f"{v1}={y}\n{v2}={x}"
        rows.append(rec("B_counterfactual", cf + f"\nGET {v1}", y, g,
                        {"answer": y, "query": v1, "role": "counterfactual",
                         "first_digit": y, "nearest": x}))
    for n, task, cnt in ((4, "B4_bind", 1500), (8, "B8_bind", 700)):
        for _ in range(cnt):
            vs = rng.sample(VARS, n); vals = rng.sample(DIGITS, n)
            g = f"{task}|" + ",".join(f"{v}={x}" for v, x in zip(vs, vals))
            qp = rng.randrange(n)
            prog = "\n".join(f"{v}={x}" for v, x in zip(vs, vals))
            rows.append(rec(task, prog + f"\nGET {vs[qp]}", vals[qp], g,
                            {"answer": vals[qp], "first_digit": vals[0],
                             "nearest": vals[-1], "query_pos": qp}))
    # 重命名
    for _ in range(500):
        v1, v2 = rng.sample(["x", "y", "z", "w"], 2)
        x, y = rng.sample(DIGITS, 2)
        g = f"BRN|{x},{y}"
        rows.append(rec("B_rename", f"{v1}={x}\n{v2}={y}\nGET {v1}", x, g,
                        {"answer": x, "first_digit": x, "nearest": y}))
    return rows


def gen_C(rng):
    rows = []
    # 重赋值
    for _ in range(1000):
        target = rng.choice(VARS)
        others = rng.sample([v for v in VARS if v != target], 3)
        old, new = rng.sample(DIGITS, 2)
        n1, n2 = rng.sample([d for d in DIGITS if d not in (old, new)], 2)
        seq = [(target, old), (others[0], n1), (target, new), (others[1], n2), (others[2], n1)]
        g = "C|" + ",".join(f"{v}={x}" for v, x in seq)
        rows.append(rec("C_reassign", "\n".join(f"{v}={x}" for v, x in seq) + f"\nGET {target}",
                        new, g, {"answer": new, "first_digit": old, "nearest": n1}))
    # 历史值/当前值显式区分（P0.4）
    for _ in range(900):
        v = rng.choice(VARS)
        old, new = rng.sample(DIGITS, 2)
        prog = f"Cur {v}={new}\nOld {v}={old}"
        g = f"CH|{v}={old}->{new}"
        rows.append(rec("C_history_current", prog + f"\nGET {v}", new, g,
                        {"answer": new, "role": "current", "first_digit": new}))
        rows.append(rec("C_history_current", prog + f"\nOLD {v}", old, g,
                        {"answer": old, "role": "history", "first_digit": new}))
    # 间接引用
    for _ in range(800):
        a, b, c = rng.sample(VARS, 3)
        old_a, new_a, cv = rng.sample(DIGITS, 3)
        seq = [(a, old_a), (b, a), (c, cv), (a, new_a)]
        g = "D|" + ",".join(f"{v}={x}" for v, x in seq)
        prog = "\n".join(f"{v}={x}" for v, x in seq)
        rows.append(rec("C_indirect", prog + f"\nGET {b}", old_a, g,
                        {"answer": old_a, "first_digit": old_a, "nearest": new_a}))
    return rows


def gen_D(rng):
    rows = []
    # 严格配对距离：同一 (target, ans) 在 dist 0/4/8/16/32/64 各一条，干扰不重赋值 target
    for _ in range(260):
        target = rng.choice(VARS)
        ans = rng.choice(DIGITS)
        others = [v for v in VARS if v != target]
        for dist in (0, 4, 8, 16, 32, 64):
            g = f"DP|{target}={ans}|d{dist}|" + str(rng.randrange(10 ** 9))
            lines = [f"{target}={ans}"]
            used = {target}
            for j in range(dist):
                v = others[j % len(others)]
                # 干扰值不能等于 target 的 ans 语义（用固定噪声数字集）
                noise = rng.choice([d for d in DIGITS if d != ans])
                lines.append(f"{v}={noise}")
            lines.append(f"GET {target}")
            rows.append(rec("D_distance", "\n".join(lines), ans, g,
                            {"answer": ans, "distance": dist,
                             "in_dist": dist <= 8, "first_digit": ans,
                             "nearest": lines[-2].split("=")[-1]}))
    # 无关干扰 / 相似变量干扰（不重赋值 target）
    for kind in ("irrelevant", "similar"):
        for _ in range(500):
            target = rng.choice(VARS)
            ans = rng.choice(DIGITS)
            noise_vars = (rng.sample([v for v in VARS if v != target], 6) if kind == "irrelevant"
                          else [target.upper()] + [v.upper() for v in VARS[:5]])
            lines = [f"{target}={ans}"] + [f"{v}={rng.choice(DIGITS)}" for v in noise_vars]
            g = f"DI|{kind}|{target}={ans}|" + str(rng.randrange(10 ** 9))
            rows.append(rec("D_interference", "\n".join(lines) + f"\nGET {target}", ans, g,
                            {"answer": ans, "kind": kind, "first_digit": ans}))
    return rows


def history_prompts():
    prompts = set()
    for rel in ("../math_m2/data/m2_train.jsonl", "../math_m1/data/m1_train.jsonl",
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
        return round(sum(1 for r in rows if str(r.get("meta", {}).get(key, "")) == str(r["answer"])) / n, 4)
    return {"fixed_answer": round(fixed, 4), "first_digit": rate("first_digit"),
            "nearest": rate("nearest"),
            "random_guess": round(1.0 / max(1, len({str(r["answer"]) for r in rows})), 4)}


def main():
    rng = random.Random(SEED)
    rows = gen_A(rng) + gen_B(rng) + gen_C(rng) + gen_D(rng)
    history = history_prompts()
    # 泛化预留（length 5-6, distance 16/32/64）
    gen = [r for r in rows if r.get("meta", {}).get("in_dist") is False and r["prompt"] not in history]
    gen_groups = {r["fact_group"] for r in gen}
    pool = [r for r in rows if r["fact_group"] not in gen_groups and r["prompt"] not in history]

    groups = defaultdict(list)
    for r in pool:
        groups[r["fact_group"]].append(r)
    keys = list(groups); rng.shuffle(keys)
    # 反事实组严格隔离（B2 组含 counterfactual/query_switch，整组不拆）
    cf_keys = []
    for k in keys:
        cf = [r for r in groups[k] if r["task"] == "B_counterfactual"]
        if len(cf) >= 1 and len(cf_keys) < 400:
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
        base_all = [r for r in groups[k] if r["task"] == "B2_bind" and r.get("meta", {}).get("role") == "base"]
        cfs = [r for r in groups[k] if r["task"] == "B_counterfactual"]
        qs = [r for r in groups[k] if r["task"] == "B_query_switch"]
        # 严格配对：每个 counterfactual 只与「查询对象相同」的 base 配对
        base_matched = []
        for c in cfs[:2]:
            q = c["meta"].get("query")
            cand = [b for b in base_all if b["meta"].get("query") == q]
            if cand:
                base_matched.append(cand[0])
        qs_matched = []
        for c in cfs[:2]:
            q = c["meta"].get("query")
            cand = [z for z in qs if z["meta"].get("query") == q]
            if cand:
                qs_matched.append(cand[0])
        if base_matched and cfs:
            cf_pairs.append({"key": k, "base": base_matched,
                             "counterfactual": cfs[:len(base_matched)],
                             "query_switch": qs_matched})
    rng.shuffle(cf_pairs)

    OUT.mkdir(parents=True, exist_ok=True)
    frozen = {}
    for s, rr in splits.items():
        p = OUT / f"m3_{s}.jsonl"
        p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rr) + "\n", encoding="utf-8")
        frozen[s] = hashlib.sha256(p.read_bytes()).hexdigest()
    (OUT / "m3_counterfactual.jsonl").write_text(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in cf_pairs) + "\n", encoding="utf-8")
    frozen["counterfactual"] = hashlib.sha256((OUT / "m3_counterfactual.jsonl").read_bytes()).hexdigest()
    (OUT / "m3_generalization.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in gen) + "\n", encoding="utf-8")
    frozen["generalization"] = hashlib.sha256((OUT / "m3_generalization.jsonl").read_bytes()).hexdigest()

    sg = {s: {r["fact_group"] for r in rr} for s, rr in splits.items()}
    cross = {}
    names = list(sg)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            cross[f"{a}_{b}"] = len(sg[a] & sg[b])
    hist_overlap = {s: len({r["prompt"] for r in rr} & history) for s, rr in splits.items()}
    # 严格配对距离校验
    dp = defaultdict(set)
    for r in rows:
        if r["task"] == "D_distance":
            dp[r["fact_group"].split("|d")[0]].add(r["meta"]["distance"])
    pairing_ok = all(dists == {0, 4, 8, 16, 32, 64} for dists in dp.values())
    cf_query_match = all(
        all(b["meta"]["query"] == c["meta"]["query"] for b, c in zip(p["base"], p["counterfactual"]))
        for p in cf_pairs if p["base"] and p["counterfactual"])

    audit = {
        "status": "M3_DATA_FROZEN", "seed": SEED,
        "tokenizer": {"version": M3Tokenizer.version, "vocab_size": 102,
                      "sha256": vocab_sha(), "sep": 101},
        "counts": {s: len(rr) for s, rr in splits.items()},
        "by_task": {s: dict(Counter(r["task"] for r in rr)) for s, rr in splits.items()},
        "reserved": {"counterfactual_pairs": len(cf_pairs), "generalization": len(gen)},
        "frozen_sha256": frozen,
        "cross_split_fact_group_overlap": cross,
        "history_prompt_overlap": hist_overlap,
        "baselines_train": baselines(splits["train"]),
        "p0_4_fixes": {
            "counterfactual_query_object_unchanged": cf_query_match,
            "query_switch_is_separate_task": True,
            "distance_strict_pairing": pairing_ok,
            "distance_interference_does_not_reassign_target": True,
            "history_current_explicitly_marked": True,
        },
        "length_train_range": "1-4", "length_test_range": "5-6",
        "distance_train": [0, 4, 8], "distance_test": [16, 32, 64],
    }
    (HERE / "M3_DATA_AUDIT.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    (HERE / "M3_LEAKAGE_AUDIT.json").write_text(json.dumps({
        "status": "LEAKAGE_AUDIT_DONE", "cross_split": cross, "history_overlap": hist_overlap,
        "counterfactual_group_isolated": True, "generalization_reserved_before_split": True,
        "p0_4_fixes": audit["p0_4_fixes"],
        "verdict": "PASS" if all(v == 0 for v in cross.values())
                   and all(v == 0 for v in hist_overlap.values())
                   and pairing_ok and cf_query_match else "REVIEW",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"counts": audit["counts"], "reserved": audit["reserved"],
                      "cross": cross, "history": hist_overlap,
                      "p0_4_fixes": audit["p0_4_fixes"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
