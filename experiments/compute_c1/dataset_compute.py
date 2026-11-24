"""C1.0 确定性课程数据生成器（C0 符号/序列、C2 变量绑定、C3 四则运算）。

- 固定 seed，train/val/sealed 完全分离，反事实成对不拆散。
- 变量名/答案/查询位置均衡；提供捷径基线所需 meta。
- 只在生成标签时用 Python；模型训练/推理绝不读解释器答案。
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent / "data"
SEED = 20261124
VARS = list("abcdefghijklmnop")
DIGITS = list("0123456789")
LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def rec(task, prompt, answer, key, meta=None):
    return {"task": task, "prompt": prompt, "answer": answer, "key": key, "meta": meta or {}}


def gen_c0(rng):
    rows = []
    # 符号复制（单字符）
    for ch in DIGITS + LETTERS + list("+-*/=<>()"):
        rows.append(rec("C0_copy1", f"COPY {ch}", ch, f"copy1|{ch}"))
    # 序列复制/反转：长度 2..6，训练长度 <=4，长序列作为泛化测试单列
    for _ in range(3000):
        n = rng.randint(2, 6)
        seq = "".join(rng.choice(DIGITS + LETTERS) for _ in range(n))
        fam = rng.choice(["copy", "reverse"])
        if fam == "copy":
            rows.append(rec("C0_copyN", f"COPY {seq}", seq, f"copyN|{seq}",
                            {"length": n, "in_distribution": n <= 4}))
        else:
            rows.append(rec("C0_reverse", f"REVERSE {seq}", seq[::-1], f"rev|{seq}",
                            {"length": n, "in_distribution": n <= 4}))
    # NEXT：明确定义 +1 规律（数字循环 9->0，字母 Z->A）
    for _ in range(1200):
        kind = rng.choice(["digit", "letter"])
        if kind == "digit":
            s = rng.randint(0, 5); seq = [str((s + i) % 10) for i in range(4)]
            ans = str((s + 4) % 10)
        else:
            s = rng.randint(0, 20); seq = [chr(ord('A') + (s + i) % 26) for i in range(4)]
            ans = chr(ord('A') + (s + 4) % 26)
        rows.append(rec("C0_next", "NEXT " + " ".join(seq), ans,
                        f"next|{kind}|{seq[0]}", {"kind": kind}))
    # INDEX：1-based
    for _ in range(1600):
        n = rng.randint(2, 6)
        seq = "".join(rng.choice(DIGITS + LETTERS) for _ in range(n))
        idx = rng.randint(1, n)
        rows.append(rec("C0_index", f"INDEX {seq} {idx}", seq[idx - 1],
                        f"idx|{seq}|{idx}", {"length": n, "index": idx,
                                             "position": idx / max(1, n)}))
    return rows


def _bind_prompt(assignments, query):
    lines = [f"{k}={v}" for k, v in assignments] + [f"GET {query}"]
    return "\n".join(lines)


def gen_c2(rng):
    rows = []
    # 单变量
    for v in VARS:
        for val in DIGITS:
            rows.append(rec("C2_bind1", _bind_prompt([(v, val)], v), val,
                            f"b1|{v}|{val}", {"n_vars": 1, "query_pos": 0,
                                               "nearest": val, "first": val}))
    # 多变量 + 反事实成对（同语义事实同组，不拆散）
    for _ in range(5000):
        n = rng.choice([2, 2, 4, 4, 8])
        chosen = rng.sample(VARS, n)
        vals = [rng.choice(DIGITS) for _ in range(n)]
        # 避免整组值相同导致捷径可用
        if len(set(vals)) < 2:
            continue
        qpos = rng.randrange(n)
        qvar = chosen[qpos]
        ans = vals[qpos]
        items = list(zip(chosen, vals))
        group = f"bN|{','.join(chosen)}|{','.join(vals)}|{qvar}"
        rows.append(rec("C2_bindN", _bind_prompt(items, qvar), ans, group,
                        {"n_vars": n, "query_pos": qpos, "nearest": vals[-1],
                         "first": vals[0], "counter_role": "orig"}))
        # 反事实：交换两个变量的值，同一查询，答案改变
        if n >= 2:
            i, j = rng.sample(range(n), 2)
            swapped = list(vals)
            swapped[i], swapped[j] = swapped[j], swapped[i]
            if swapped[qpos] != ans:
                rows.append(rec("C2_counter", _bind_prompt(list(zip(chosen, swapped)), qvar),
                                swapped[qpos], group,
                                {"n_vars": n, "query_pos": qpos, "nearest": swapped[-1],
                                 "first": swapped[0], "counter_role": "swap"}))
    # 重新赋值：最后赋值必须胜出
    for _ in range(1200):
        v = rng.choice(VARS)
        vals = [rng.choice(DIGITS) for _ in range(3)]
        if len(set(vals)) < 2:
            continue
        seq = [(v, vals[0]), (rng.choice([x for x in VARS if x != v]), rng.choice(DIGITS)),
               (v, vals[1]), (v, vals[2])]
        rows.append(rec("C2_reassign", _bind_prompt(seq, v), vals[2],
                        f"re|{v}|{','.join(vals)}",
                        {"n_vars": len(seq), "query_pos": 2, "nearest": vals[-1],
                         "first": vals[0]}))
    # 间接引用：b=a
    for _ in range(800):
        a, b = rng.sample(VARS, 2)
        val = rng.choice(DIGITS)
        rows.append(rec("C2_indirect", f"{a}={val}\n{b}={a}\nGET {b}", val,
                        f"ind|{a}|{b}|{val}", {"n_vars": 2, "query_pos": 1,
                                                "nearest": val, "first": val}))
    return rows


def gen_c3(rng):
    rows = []
    ops = {
        "ADD": lambda a, b: a + b,
        "SUB": lambda a, b: max(0, a - b),
        "MUL": lambda a, b: a * b,
        "DIV": lambda a, b: a // b if b else 0,
    }
    for name, fn in ops.items():
        for _ in range(1400):
            if name == "DIV":
                b = rng.randint(1, 9)
                a = rng.randint(b, min(9, b * 3))  # 整除，商<=9
            else:
                a, b = rng.randint(0, 9), rng.randint(0, 9)
            ans = str(fn(a, b))
            if len(ans) > 2:
                continue
            rows.append(rec("C3_arith", f"{name} {a} {b}", ans,
                            f"{name}|{a}|{b}", {"op": name, "a": a, "b": b}))
        # 变量运算
        for _ in range(600):
            va, vb = rng.sample(VARS, 2)
            a = rng.randint(0, 5); b = rng.randint(0, 5)
            if name == "DIV":
                b = max(1, b); a = a * b
            ans = str(fn(a, b))
            if len(ans) > 2:
                continue
            prompt = f"{va}={a}\n{vb}={b}\n{name} {va} {vb}"
            rows.append(rec("C3_var_arith", prompt, ans,
                            f"{name}var|{a}|{b}|{va}{vb}",
                            {"op": name, "a": a, "b": b}))
    return rows


def split_dataset(rows, seed=SEED):
    """按 key 分流，反事实同组不拆散；冻结 SHA；跨集合 prompt 零重合。"""
    rng = random.Random(seed)
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["key"]].append(r)
    keys = list(groups)
    rng.shuffle(keys)
    splits = {"train": [], "val": [], "sealed": []}
    for k in keys:
        h = int(hashlib.sha256(k.encode()).hexdigest()[:8], 16) % 100
        s = "train" if h < 78 else "val" if h < 89 else "sealed"
        splits[s].extend(groups[k])
    # prompt 去重（跨集合）
    seen = set()
    for s in ("train", "val", "sealed"):
        clean = []
        for r in splits[s]:
            if r["prompt"] in seen:
                continue
            seen.add(r["prompt"])
            clean.append(r)
        splits[s] = clean
    return splits


def shortcut_baselines(rows):
    n = max(1, len(rows))
    fixed = Counter(r["answer"] for r in rows).most_common(1)[0][1] / n
    nearest = sum(1 for r in rows if r.get("meta", {}).get("nearest") == r["answer"]) / n
    first = sum(1 for r in rows if r.get("meta", {}).get("first") == r["answer"]) / n
    # 固定答案/最近/首个之外的随机猜测（按有效答案数）
    n_ans = len({r["answer"] for r in rows})
    return {"fixed_answer": round(fixed, 4), "nearest_assignment": round(nearest, 4),
            "first_assignment": round(first, 4),
            "random_guess": round(1.0 / max(1, n_ans), 4)}


def main():
    rng = random.Random(SEED)
    rows = gen_c0(rng) + gen_c2(rng) + gen_c3(rng)
    splits = split_dataset(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    frozen = {}
    for s, rr in splits.items():
        p = OUT / f"c1_{s}.jsonl"
        p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rr) + "\n", encoding="utf-8")
        frozen[s] = hashlib.sha256(p.read_bytes()).hexdigest()
    # 反事实测试（独立）：至少 200 对
    cf_groups = defaultdict(list)
    for r in rows:
        if r.get("meta", {}).get("counter_role") and r["task"] in ("C2_counter", "C2_bindN"):
            cf_groups[r["key"]].append(r)
    cf_pairs = []
    for k, g in cf_groups.items():
        orig = [x for x in g if x["meta"].get("counter_role") == "orig"]
        swap = [x for x in g if x["meta"].get("counter_role") == "swap"]
        if orig and swap and orig[0]["answer"] != swap[0]["answer"]:
            cf_pairs.append({"key": k, "orig": orig[0], "swap": swap[0]})
    (OUT / "c1_counterfactual.jsonl").write_text(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in cf_pairs) + "\n", encoding="utf-8")
    frozen["counterfactual"] = hashlib.sha256((OUT / "c1_counterfactual.jsonl").read_bytes()).hexdigest()

    audit = {
        "status": "DATA_GENERATED", "seed": SEED,
        "counts": {s: len(rr) for s, rr in splits.items()},
        "by_task": {s: dict(Counter(r["task"] for r in rr)) for s, rr in splits.items()},
        "counterfactual_pairs": len(cf_pairs),
        "shortcut_baselines_train": shortcut_baselines(splits["train"]),
        "shortcut_baselines_sealed": shortcut_baselines(splits["sealed"]),
        "frozen_sha256": frozen,
        "prompt_overlap": {},
        "data_deterministic": True,
    }
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i+1:]:
            pa = {r["prompt"] for r in splits[a]}
            pb = {r["prompt"] for r in splits[b]}
            audit["prompt_overlap"][f"{a}_{b}"] = len(pa & pb)
    (Path(__file__).resolve().parent / "C1_DATA_AUDIT.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"counts": audit["counts"], "cf_pairs": len(cf_pairs),
                      "tasks": audit["by_task"]["train"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
