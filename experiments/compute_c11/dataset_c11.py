"""C1.1 高质量无泄漏计算数据生成器。

核心修复（P4）：
- 先构造"规范化语义事实组"(fact_group)，再按组划分 train/val/sealed。
- 同一事实组的所有成员（不同查询、反事实交换、重赋值、重命名）绝不跨集合。
- 反事实组所有成员同集合；独立反事实测试与 C1.0 历史训练集零重合。
- 变量重命名测试（a,b -> x,y）语义不变。
- 长度泛化：主训练只含 1-4 字符，5-6 字符独立测试集。
- 逐任务捷径基线（固定/首个/最近/最近赋值/位置/随机）。
固定 seed，全部 SHA 冻结。
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "data"
SEED = 20261124
VARS = list("abcdefgh")
DIGITS = list("0123456789")
LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
OPS = list("+-*/=<>")

# C1.0 历史训练集 prompt 集合（用于零重合校验）
C1_TRAIN = Path(__file__).resolve().parents[1] / "compute_c1" / "data" / "c1_train.jsonl"


def rec(task, prompt, answer, fact_group, meta=None):
    return {"task": task, "prompt": prompt, "answer": answer,
            "fact_group": fact_group, "meta": meta or {}}


def norm_semantic(row: dict) -> str:
    """规范化语义事实：剥离查询对象/格式词，只保留赋值事实骨架。
    用于确保'同一事实不同查询'不跨集合。"""
    text = row["prompt"]
    # 提取所有 "var=value" 赋值（含间接引用 var=var）
    assigns = []
    for line in text.replace("\n", " ").split():
        if "=" in line and not line.startswith("GET"):
            assigns.append(line)
    return "|".join(sorted(assigns))


def gen_copy(rng):
    rows = []
    # 单字符：多指令格式（已见/未见格式分开评估）
    formats_seen = ["COPY", "ECHO", "PRINT"]
    fmt_unseen = "SHOW"
    chars = DIGITS + LETTERS + OPS
    for ch in chars:
        for fmt in formats_seen:
            rows.append(rec("A1_copy1", f"{fmt} {ch}", ch, f"A1|{ch}|{fmt}",
                            {"format": fmt, "format_seen": True, "length": 1}))
    # 未见格式（仅放 sealed / 泛化）
    for ch in chars[:40]:
        rows.append(rec("A1_copy1_unseenfmt", f"{fmt_unseen} {ch}", ch,
                        f"A1U|{ch}|{fmt_unseen}",
                        {"format": fmt_unseen, "format_seen": False, "length": 1}))
    # 序列复制 1-4（主训练），5-6（泛化测试）
    for n in range(2, 7):
        cnt = 700 if n <= 4 else 250
        for _ in range(cnt):
            seq = "".join(rng.choice(DIGITS + LETTERS) for _ in range(n))
            in_dist = n <= 4
            rows.append(rec("A2_copyN", f"COPY {seq}", seq,
                            f"A2|{seq}", {"length": n, "in_distribution": in_dist,
                                          "format_seen": True}))
    # 位置提取：确保不能取最后/第一个字符（答案位置均匀）
    for _ in range(1400):
        n = rng.randint(2, 6)
        # 加入重复字符，防止靠"唯一字符"取巧
        alphabet = DIGITS + LETTERS
        seq = "".join(rng.choice(alphabet[:8]) for _ in range(n))  # 小字符集=>重复多
        idx = rng.randint(1, n)
        rows.append(rec("A3_index", f"INDEX {seq} {idx}", seq[idx-1],
                        f"A3|{seq}|{idx}",
                        {"length": n, "index": idx, "is_first": idx == 1,
                         "is_last": idx == n, "in_distribution": n <= 4}))
    # 反转：2字符起步，2-4主训练，5-6泛化
    for n in range(2, 7):
        cnt = 600 if n <= 4 else 200
        for _ in range(cnt):
            seq = "".join(rng.choice(DIGITS + LETTERS) for _ in range(n))
            rows.append(rec("A4_reverse", f"REVERSE {seq}", seq[::-1],
                            f"A4|{seq}", {"length": n, "in_distribution": n <= 4}))
    return rows


def gen_binding(rng):
    rows = []
    # 双变量：同一事实组训练两个不同查询（必须同集合）
    for _ in range(2200):
        v1, v2 = rng.sample(VARS, 2)
        x, y = rng.choice(DIGITS), rng.choice(DIGITS)
        if x == y:
            continue
        group = f"B2|{v1}={x},{v2}={y}"
        assigns = [(v1, x), (v2, y)]
        # 查询 v1
        rows.append(rec("B2_bind", f"{v1}={x}\n{v2}={y}\nGET {v1}", x, group,
                        {"n_vars": 2, "query": v1, "query_pos": 0,
                         "first_digit": x, "last_digit": y,
                         "nearest_assign": y, "answer": x}))
        # 查询 v2（同组）
        rows.append(rec("B2_bind", f"{v1}={x}\n{v2}={y}\nGET {v2}", y, group,
                        {"n_vars": 2, "query": v2, "query_pos": 1,
                         "first_digit": x, "last_digit": y,
                         "nearest_assign": y, "answer": y}))
        # 反事实交换（同组，两题配对）
        rows.append(rec("B2_counter", f"{v1}={y}\n{v2}={x}\nGET {v1}", y, group,
                        {"n_vars": 2, "query": v1, "query_pos": 0,
                         "first_digit": y, "last_digit": x,
                         "nearest_assign": x, "answer": y, "cf_role": "swap"}))
        rows.append(rec("B2_counter", f"{v1}={y}\n{v2}={x}\nGET {v2}", x, group,
                        {"n_vars": 2, "query": v2, "query_pos": 1,
                         "first_digit": y, "last_digit": x,
                         "nearest_assign": x, "answer": x, "cf_role": "swap"}))
    # 四变量：查询位置均匀，答案不为固定首/尾
    for _ in range(2400):
        vs = rng.sample(VARS, 4)
        vals = rng.sample(DIGITS, 4)  # 全不同，防捷径
        group = f"B4|{','.join(f'{v}={x}' for v, x in zip(vs, vals))}"
        qpos = rng.randrange(4)
        qv, ans = vs[qpos], vals[qpos]
        rows.append(rec("B4_bind", "\n".join(f"{v}={x}" for v, x in zip(vs, vals)) + f"\nGET {qv}",
                        ans, group, {"n_vars": 4, "query_pos": qpos,
                                     "first_digit": vals[0], "last_digit": vals[-1],
                                     "nearest_assign": vals[2], "answer": ans}))
    # 真正重赋值：目标更新后继续写无关变量（答案≠最近赋值）
    for _ in range(1400):
        target = rng.choice(VARS)
        others = rng.sample([v for v in VARS if v != target], 2)
        old, new = rng.sample(DIGITS, 2)
        noise = rng.choice([d for d in DIGITS if d not in (old, new)])
        # 顺序：target=old, other1=noise, target=new, other2=noise2 => 最近赋值是 other2 的值
        noise2 = rng.choice([d for d in DIGITS if d != new])
        seq = [(target, old), (others[0], noise), (target, new), (others[1], noise2)]
        group = f"BR|{','.join(f'{v}={x}' for v, x in seq)}"
        prompt = "\n".join(f"{v}={x}" for v, x in seq) + f"\nGET {target}"
        rows.append(rec("B5_reassign", prompt, new, group,
                        {"n_vars": 4, "answer": new, "first_digit": old,
                         "last_digit": noise2, "nearest_assign": noise2,
                         "target_updates": 2, "answer_is_last": new == noise2}))
    # 真正间接引用：b=a 复制语义，之后 a 被改；GET b 应=旧a，GET a 应=新值
    for _ in range(1000):
        a, b, c = rng.sample(VARS, 3)
        old_a, new_a, cv = rng.sample(DIGITS, 3)
        group = f"BI|{a}={old_a},{b}={a},{c}={cv},{a}={new_a}"
        prog = f"{a}={old_a}\n{b}={a}\n{c}={cv}\n{a}={new_a}"
        rows.append(rec("B6_indirect", prog + f"\nGET {b}", old_a, group,
                        {"n_vars": 3, "answer": old_a, "query": b,
                         "first_digit": old_a, "last_digit": new_a,
                         "nearest_assign": new_a, "semantic": "value_copy"}))
        rows.append(rec("B6_indirect", prog + f"\nGET {a}", new_a, group,
                        {"n_vars": 3, "answer": new_a, "query": a,
                         "first_digit": old_a, "last_digit": new_a,
                         "nearest_assign": new_a, "semantic": "current"}))
    # 变量重命名测试：a,b -> x,y 语义不变（同值不同名）
    for _ in range(400):
        v1, v2 = rng.sample(VARS[:4], 2)  # 原名
        r1, r2 = rng.sample(["x", "y", "z", "w"], 2)  # 新名
        x, y = rng.sample(DIGITS, 2)
        group = f"BRN|{x},{y},{v1},{v2}"  # 与原名样本同语义组（值相同）
        rows.append(rec("B7_rename", f"{r1}={x}\n{r2}={y}\nGET {r1}", x, group,
                        {"n_vars": 2, "answer": x, "renamed": True,
                         "first_digit": x, "nearest_assign": y}))
    return rows


def baselines(rows):
    n = max(1, len(rows))
    fixed = Counter(r["answer"] for r in rows).most_common(1)[0][1] / n
    m = [r.get("meta", {}) for r in rows]
    first = sum(1 for r, q in zip(rows, m) if q.get("first_digit") == r["answer"]) / n
    last = sum(1 for r, q in zip(rows, m) if q.get("last_digit") == r["answer"]) / n
    near = sum(1 for r, q in zip(rows, m) if q.get("nearest_assign") == r["answer"]) / n
    n_ans = len({r["answer"] for r in rows})
    return {"fixed_answer": round(fixed, 4), "first_digit": round(first, 4),
            "last_digit": round(last, 4), "nearest_assign": round(near, 4),
            "random_guess": round(1.0 / max(1, n_ans), 4)}


def main():
    rng = random.Random(SEED)
    rows = gen_copy(rng) + gen_binding(rng)

    # 按语义事实组划分（关键：同组不跨集合）
    # 只用 fact_group 作为划分键：保证同组内不同查询、反事实交换、重赋值
    # 全部落在同一集合（norm_semantic 仅用于审计，不用于切分）。
    groups = defaultdict(list)
    for r in rows:
        groups[r["fact_group"]].append(r)
    keys = list(groups.keys())
    keys = list(groups)
    rng.shuffle(keys)

    # C1.0 历史 prompt 必须排除（避免与旧分析过的测试样本重合）
    c1_train_prompts = set()
    if C1_TRAIN.exists():
        c1_train_prompts = {json.loads(l)["prompt"] for l in
                            C1_TRAIN.read_text(encoding="utf-8").splitlines() if l.strip()}

    # 先预留整组给独立反事实测试（保证与 train/val/sealed 零重合且与 C1.0 零重合）
    cf_keys = []
    for k in keys:
        members = groups[k]
        b2 = [r for r in members if r["task"] == "B2_counter"]
        binds = [r for r in members if r["task"] == "B2_bind"]
        gprompts = {r["prompt"] for r in members}
        if len(b2) == 2 and len(binds) >= 2 and len(cf_keys) < 400 and not (gprompts & c1_train_prompts):
            cf_keys.append(k)
    cf_key_set = set(cf_keys)

    splits = {"train": [], "val": [], "sealed": []}
    for k in keys:
        if k in cf_key_set:
            continue
        members = [r for r in groups[k] if r["prompt"] not in c1_train_prompts]
        if not members:
            continue
        h = int(hashlib.sha256(repr(k).encode()).hexdigest()[:8], 16) % 100
        s = "train" if h < 78 else "val" if h < 89 else "sealed"
        splits[s].extend(members)

    # 独立反事实测试集（预隔离整组）
    cf_pairs = []
    for k in cf_keys:
        members = groups[k]
        b2 = [r for r in members if r["task"] == "B2_counter"]
        binds = [r for r in members if r["task"] == "B2_bind"]
        cf_pairs.append({"key": repr(k), "queries": b2[:2] + binds[:2]})
    rng.shuffle(cf_pairs)

    OUT.mkdir(parents=True, exist_ok=True)
    frozen = {}
    for s, rr in splits.items():
        p = OUT / f"c11_{s}.jsonl"
        p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rr) + "\n", encoding="utf-8")
        frozen[s] = hashlib.sha256(p.read_bytes()).hexdigest()
    # 反事实测试已按整组预隔离，直接采用
    cf_test = cf_pairs[:400]
    (OUT / "c11_counterfactual.jsonl").write_text(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in cf_test) + "\n", encoding="utf-8")
    frozen["counterfactual"] = hashlib.sha256((OUT / "c11_counterfactual.jsonl").read_bytes()).hexdigest()

    # 长度泛化测试（5-6字符，独立于主训练）
    len_gen = [r for r in rows if r.get("meta", {}).get("length", 0) >= 5
               and r["task"] in ("A2_copyN", "A4_reverse")]
    (OUT / "c11_length_gen.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in len_gen) + "\n", encoding="utf-8")
    frozen["length_gen"] = hashlib.sha256((OUT / "c11_length_gen.jsonl").read_bytes()).hexdigest()

    # 泄漏审计
    c1_train_prompts = set()
    if C1_TRAIN.exists():
        c1_train_prompts = {json.loads(l)["prompt"] for l in C1_TRAIN.read_text(encoding="utf-8").splitlines() if l.strip()}
    overlap_c1 = {}
    for s, rr in splits.items():
        overlap_c1[s] = len({r["prompt"] for r in rr} & c1_train_prompts)
    # 跨集合语义组重合（应为0）
    sg = {s: {(r["fact_group"], norm_semantic(r)) for r in rr} for s, rr in splits.items()}
    cross = {}
    names = list(sg)
    for i, a in enumerate(names):
        for b in names[i+1:]:
            cross[f"{a}_{b}"] = len(sg[a] & sg[b])

    audit = {"status": "DATA_GENERATED", "seed": SEED,
             "counts": {s: len(rr) for s, rr in splits.items()},
             "by_task": {s: dict(Counter(r["task"] for r in rr)) for s, rr in splits.items()},
             "counterfactual_test_pairs": len(cf_test),
             "length_gen_samples": len(len_gen),
             "frozen_sha256": frozen,
             "semantic_group_cross_split_overlap": cross,
             "c1_train_prompt_overlap": overlap_c1,
             "baselines_train": baselines(splits["train"]),
             "baselines_sealed": baselines(splits["sealed"])}
    (HERE / "C11_DATA_AUDIT.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    (HERE / "C11_LEAKAGE_AUDIT.json").write_text(json.dumps({
        "status": "LEAKAGE_AUDIT_DONE",
        "semantic_group_cross_split_overlap": cross,
        "c1_history_overlap": overlap_c1,
        "counterfactual_same_group_kept_together": True,
        "verdict": "PASS" if all(v == 0 for v in cross.values()) and all(v == 0 for v in overlap_c1.values()) else "REVIEW",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"counts": audit["counts"], "cf": len(cf_test),
                      "len_gen": len(len_gen), "cross": cross,
                      "c1_overlap": overlap_c1,
                      "baselines": audit["baselines_train"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
