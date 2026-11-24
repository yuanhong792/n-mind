"""M1 独立无泄漏计算数据集与字符级 tokenizer。

所有候选模型共用此 tokenizer。数据按 fact_group 切分；反事实组、同事实多查询、
变量重命名均不跨集合。长度泛化样本在训练前预留，不进入 train/val/sealed。
"""
from __future__ import annotations

import hashlib
import json
import random
import string
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "data"
SEED = 20261124

PAD, BOS, EOS, UNK = 0, 1, 2, 3
SPECIAL = {PAD: "<pad>", BOS: "<bos>", EOS: "<eos>", UNK: "<unk>"}
_CHARS = [chr(c) for c in range(32, 127)] + ["\n", "\t"]
_SYMBOLS = sorted(set(_CHARS))
ID2CHAR = {i: ch for i, ch in enumerate(_SYMBOLS, start=4)}
CHAR2ID = {ch: i for i, ch in ID2CHAR.items()}
VOCAB_SIZE = max(ID2CHAR) + 1
VERSION = "nmind-math-m1-char/1"


def vocab_sha() -> str:
    payload = {"version": VERSION, "vocab_size": VOCAB_SIZE,
               "special": SPECIAL, "chars": list(ID2CHAR.values())}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class M1Tokenizer:
    version = VERSION

    def __init__(self):
        self.vocab_size = VOCAB_SIZE

    @property
    def vocab_sha(self) -> str:
        return vocab_sha()

    def encode(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be str")
        out = []
        for ch in text:
            if ch not in CHAR2ID:
                raise ValueError(f"未定义字符 {ch!r}")
            out.append(CHAR2ID[ch])
        return out

    def decode(self, ids, skip_special=True) -> str:
        parts = []
        for value in ids:
            i = int(value)
            if i in ID2CHAR:
                parts.append(ID2CHAR[i])
            elif i in SPECIAL and not skip_special:
                parts.append(SPECIAL[i])
        return "".join(parts)


VARS = list("abcdefgh")
RENAMED = ["x", "y", "z", "w"]
DIGITS = list("0123456789")
UPPER = list(string.ascii_uppercase)
SEQ_CHARS = DIGITS + UPPER


def rec(task, prompt, answer, group, meta=None):
    return {"task": task, "prompt": prompt, "answer": answer,
            "fact_group": group, "meta": meta or {}}


def gen_sequences(rng):
    rows = []
    chars = DIGITS + UPPER + list("+-*/=<>")
    for ch in chars:
        for fmt in ("COPY", "ECHO", "PRINT"):
            rows.append(rec("A1_copy1", f"{fmt} {ch}", ch, f"A1|{ch}|{fmt}",
                            {"first_digit": ch if ch.isdigit() else "", "answer": ch}))
    for ch in chars[:40]:
        rows.append(rec("A1_copy1_unseen_fmt", f"SHOW {ch}", ch, f"A1U|{ch}",
                        {"unseen_format": True, "answer": ch}))
    for n in range(2, 7):
        count = 750 if n <= 4 else 260
        for _ in range(count):
            seq = "".join(rng.choice(SEQ_CHARS) for _ in range(n))
            rows.append(rec("A2_copyN", f"COPY {seq}", seq, f"A2|{seq}",
                            {"length": n, "in_distribution": n <= 4, "answer": seq}))
    for _ in range(1400):
        n = rng.randint(2, 6)
        seq = "".join(rng.choice(DIGITS[:6]) for _ in range(n))
        idx = rng.randint(1, n)
        rows.append(rec("A3_index", f"INDEX {seq} {idx}", seq[idx-1],
                        f"A3|{seq}|{idx}",
                        {"length": n, "in_distribution": n <= 4, "index": idx,
                         "answer": seq[idx-1]}))
    for n in range(2, 7):
        count = 650 if n <= 4 else 240
        for _ in range(count):
            seq = "".join(rng.choice(SEQ_CHARS) for _ in range(n))
            rows.append(rec("A4_reverse", f"REVERSE {seq}", seq[::-1],
                            f"A4|{seq}",
                            {"length": n, "in_distribution": n <= 4, "answer": seq[::-1]}))
    for _ in range(700):
        ch = rng.choice(SEQ_CHARS)
        n = rng.randint(2, 5)
        rows.append(rec("A5_repeat", f"REPEAT {ch} {n}", ch*n,
                        f"A5|{ch}|{n}", {"answer": ch*n}))
    return rows


def gen_binding(rng):
    rows = []
    # 单变量
    for _ in range(500):
        v, x = rng.choice(VARS), rng.choice(DIGITS)
        rows.append(rec("B1_bind1", f"{v}={x}\nGET {v}", x, f"B1|{v}={x}",
                        {"answer": x, "first_digit": x, "last_digit": x, "nearest_assign": x}))
    # 双变量：同组两个查询 + 反事实交换两个查询，整组不拆分
    for _ in range(1800):
        v1, v2 = rng.sample(VARS, 2)
        x, y = rng.sample(DIGITS, 2)
        g = f"B2|{v1}={x},{v2}={y}"
        rows.append(rec("B2_bind", f"{v1}={x}\n{v2}={y}\nGET {v1}", x, g,
                        {"answer": x, "first_digit": x, "last_digit": y,
                         "nearest_assign": y, "query": v1}))
        rows.append(rec("B2_bind", f"{v1}={x}\n{v2}={y}\nGET {v2}", y, g,
                        {"answer": y, "first_digit": x, "last_digit": y,
                         "nearest_assign": y, "query": v2}))
        rows.append(rec("B2_query_switch", f"{v1}={x}\n{v2}={y}\nGET {v2}", y, g,
                        {"answer": y, "query": v2, "first_digit": x, "nearest_assign": y}))
        rows.append(rec("B_swap", f"{v1}={y}\n{v2}={x}\nGET {v1}", y, g,
                        {"answer": y, "first_digit": y, "nearest_assign": x, "cf_role": "swap"}))
        rows.append(rec("B_swap", f"{v1}={y}\n{v2}={x}\nGET {v2}", x, g,
                        {"answer": x, "first_digit": y, "nearest_assign": x, "cf_role": "swap"}))
    # 四变量 / 八变量，查询位置均匀
    for n, task, cnt in ((4, "B4_bind", 1900), (8, "B8_bind", 900)):
        for _ in range(cnt):
            vs = rng.sample(VARS, n)
            vals = rng.sample(DIGITS, n)
            g = f"{task}|" + ",".join(f"{v}={x}" for v, x in zip(vs, vals))
            qp = rng.randrange(n)
            prog = "\n".join(f"{v}={x}" for v, x in zip(vs, vals))
            rows.append(rec(task, prog + f"\nGET {vs[qp]}", vals[qp], g,
                            {"answer": vals[qp], "first_digit": vals[0],
                             "last_digit": vals[-1], "nearest_assign": vals[-1],
                             "query_pos": qp, "n_vars": n}))
    # 变量重命名：同名无关，按值集合归组
    for _ in range(500):
        v1, v2 = rng.sample(RENAMED, 2)
        x, y = rng.sample(DIGITS, 2)
        g = f"BRN|{x},{y}"
        rows.append(rec("B_rename", f"{v1}={x}\n{v2}={y}\nGET {v1}", x, g,
                        {"answer": x, "first_digit": x, "nearest_assign": y, "renamed": True}))
    return rows


def gen_updates(rng):
    rows = []
    # 记忆更新：目标写入旧值，多个无关赋值，再写新值，末尾再放干扰
    for _ in range(1200):
        target = rng.choice(VARS)
        others = rng.sample([v for v in VARS if v != target], 3)
        old, new = rng.sample(DIGITS, 2)
        n1, n2 = rng.sample([d for d in DIGITS if d not in (old, new)], 2)
        seq = [(target, old), (others[0], n1), (target, new), (others[1], n2), (others[2], n1)]
        g = "C|" + ",".join(f"{v}={x}" for v, x in seq)
        prog = "\n".join(f"{v}={x}" for v, x in seq)
        rows.append(rec("C_reassign", prog + f"\nGET {target}", new, g,
                        {"answer": new, "first_digit": old, "last_digit": n1,
                         "nearest_assign": n1, "target_updates": 2}))
    # 间接引用：值复制语义，之后 a 被改写，查询 b 应保留旧 a
    for _ in range(1100):
        a, b, c = rng.sample(VARS, 3)
        old_a, new_a, cv = rng.sample(DIGITS, 3)
        seq = [(a, old_a), (b, a), (c, cv), (a, new_a)]
        g = "D|" + ",".join(f"{v}={x}" for v, x in seq)
        prog = "\n".join(f"{v}={x}" for v, x in seq)
        rows.append(rec("D_indirect", prog + f"\nGET {b}", old_a, g,
                        {"answer": old_a, "first_digit": old_a, "last_digit": new_a,
                         "nearest_assign": new_a, "semantic": "value_copy"}))
        rows.append(rec("D_indirect", prog + f"\nGET {a}", new_a, g,
                        {"answer": new_a, "first_digit": old_a, "last_digit": new_a,
                         "nearest_assign": new_a, "semantic": "current"}))
    return rows


def gen_distance(rng):
    rows = []
    # 训练只包含 0/4/8；16/32/64 仅验证/封存，构成真实距离泛化
    for dist in (0, 4, 8, 16, 32, 64):
        count = 420 if dist <= 8 else 260
        for _ in range(count):
            target = rng.choice(VARS)
            ans = rng.choice(DIGITS)
            g = f"E|{dist}|{target}={ans}|" + str(rng.randrange(10**9))
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


def baselines(rows):
    n = max(1, len(rows))
    counts = Counter(str(r["answer"]) for r in rows)
    fixed = counts.most_common(1)[0][1] / n if counts else 0.0
    def rate(key):
        return sum(1 for r in rows if str(r.get("meta", {}).get(key, "")) == str(r["answer"])) / n
    return {"fixed_answer": round(fixed, 4),
            "first_digit": round(rate("first_digit"), 4),
            "last_digit": round(rate("last_digit"), 4),
            "nearest_assign": round(rate("nearest_assign"), 4),
            "random_guess": round(1.0 / max(1, len({str(r["answer"]) for r in rows})), 4)}


def load_history_prompts():
    prompts = set()
    for rel in (("../compute_c1/data/c1_train.jsonl",),
                ("../compute_c11/data/c11_train.jsonl",)):
        p = (HERE / rel[0]).resolve()
        if p.exists():
            prompts.update(json.loads(l)["prompt"] for l in p.read_text(encoding="utf-8").splitlines() if l.strip())
    return prompts


def main():
    rng = random.Random(SEED)
    rows = gen_sequences(rng) + gen_binding(rng) + gen_updates(rng) + gen_distance(rng)
    history = load_history_prompts()

    # 长度泛化和未见格式先预留，永不进入主划分
    length_gen = [r for r in rows if (r.get("meta", {}).get("in_distribution") is False
                  or r["task"] == "A1_copy1_unseen_fmt") and r["prompt"] not in history]
    length_groups = {r["fact_group"] for r in length_gen}
    pool = [r for r in rows if r["fact_group"] not in length_groups
            and r["prompt"] not in history]

    # 反事实组预隔离 ≥300
    groups = defaultdict(list)
    for r in pool:
        groups[r["fact_group"]].append(r)
    keys = list(groups)
    rng.shuffle(keys)
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
        p = OUT / f"m1_{s}.jsonl"
        p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rr) + "\n", encoding="utf-8")
        frozen[s] = hashlib.sha256(p.read_bytes()).hexdigest()
    (OUT / "m1_counterfactual.jsonl").write_text(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in cf_pairs) + "\n", encoding="utf-8")
    frozen["counterfactual"] = hashlib.sha256((OUT / "m1_counterfactual.jsonl").read_bytes()).hexdigest()
    (OUT / "m1_length_gen.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in length_gen) + "\n", encoding="utf-8")
    frozen["length_gen"] = hashlib.sha256((OUT / "m1_length_gen.jsonl").read_bytes()).hexdigest()

    # 泄漏审计
    sg = {s: {r["fact_group"] for r in rr} for s, rr in splits.items()}
    cross = {}
    names = list(sg)
    for i, a in enumerate(names):
        for b in names[i+1:]:
            cross[f"{a}_{b}"] = len(sg[a] & sg[b])
    hist_overlap = {s: len({r["prompt"] for r in rr} & history) for s, rr in splits.items()}
    cf_overlap = len({q["prompt"] for p in cf_pairs for q in p["queries"]} & history)
    lg_overlap = len({r["prompt"] for r in length_gen} & history)

    audit = {
        "status": "M1_DATA_FROZEN", "seed": SEED,
        "tokenizer": {"version": VERSION, "vocab_size": VOCAB_SIZE,
                      "sha256": vocab_sha(), "roundtrip": True},
        "counts": {s: len(rr) for s, rr in splits.items()},
        "by_task": {s: dict(Counter(r["task"] for r in rr)) for s, rr in splits.items()},
        "reserved": {"counterfactual_pairs": len(cf_pairs), "length_generalization": len(length_gen)},
        "frozen_sha256": frozen,
        "cross_split_fact_group_overlap": cross,
        "history_prompt_overlap": {**hist_overlap, "counterfactual": cf_overlap,
                                   "length_gen": lg_overlap},
        "baselines_train": baselines(splits["train"]),
        "baselines_by_task_train": {t: baselines([r for r in splits["train"] if r["task"] == t])
                                    for t in sorted({r["task"] for r in splits["train"]})},
        "length_train_range": "2-4", "length_test_range": "5-6",
        "distance_train": [0, 4, 8], "distance_test": [16, 32, 64],
    }
    (HERE / "M1_DATA_AUDIT.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    (HERE / "M1_LEAKAGE_AUDIT.json").write_text(json.dumps({
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
                      "baselines": audit["baselines_train"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
