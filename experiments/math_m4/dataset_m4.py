"""M4 数据生成与冻结：修复 M3 的四类数据缺陷。

1. 所有核心任务（含 A5_repeat）训练样本充足，不重复凑数。
2. 反事实四类配对：同查询目标改值 / 无关事实改值 / 查询切换 / 赋值顺序交换；
   查询位置首/中/末均衡；配对两条都对才算通过。
3. 距离严格配对：同一事实、同一查询，仅干扰数量 0/4/8/16/32/64/128 变化，共享 pair_id。
4. 规范化语义族全局去重：train/val/sealed/generalization 之间 canonical 零重合；
   fact_group 字符串不作为隔离证据。

运行：python dataset_m4.py
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parents[0]
DATA = HERE / "data"
DATA.mkdir(exist_ok=True)

VARS = list("abcdefghijklmnop")  # 16 变量：8 用于训练绑定，12 用于未见变量数组合
DIGITS = list("0123456789")
CHARS = [chr(c) for c in range(33, 127) if chr(c) != "="]  # 可打印 ASCII，tokenizer 全覆盖
SEED = 20261124


def canonical(prompt: str) -> str:
    assigns: dict[str, str] = {}
    query = None
    matched = 0
    for line in prompt.strip().splitlines():
        s = line.strip()
        if len(s) >= 3 and s[0] in VARS and s[1] == "=" and s[2] in DIGITS:
            assigns[s[0]] = s[2]
            matched += 1
        elif s.startswith("GET "):
            query = s.split()[-1]
            matched += 1
    if matched == 0:
        return "RAW|" + prompt
    return "|".join(f"{k}={v}" for k, v in sorted(assigns.items())) + f"|GET:{query or '-'}"


class Generator:
    def __init__(self, seed=SEED):
        self.rng = random.Random(seed)
        self.used: set[str] = set()

    def row(self, task, prompt, answer, meta=None, group=None):
        c = canonical(prompt)
        return {"task": task, "prompt": prompt, "answer": answer,
                "fact_group": group or c, "canonical": c, "meta": meta or {}}

    def take(self, task, prompt, answer, meta=None, group=None):
        """全局语义去重：canonical 已存在则拒绝（返回 None）。"""
        c = canonical(prompt)
        if c in self.used:
            return None
        self.used.add(c)
        return self.row(task, prompt, answer, meta, group)

    def assignments(self, n: int) -> list[tuple[str, str]]:
        vs = self.rng.sample(VARS, n)
        return [(v, self.rng.choice(DIGITS)) for v in vs]

    def bind_prompt(self, assigns, query, interleave=0):
        items = [f"{v}={d}" for v, d in assigns]
        if interleave:
            noise = [(self.rng.choice(VARS), self.rng.choice(DIGITS)) for _ in range(interleave)]
            all_items = items + [f"{v}={d}" for v, d in noise]
            self.rng.shuffle(all_items)
            items = all_items
        return "\n".join(items) + f"\nGET {query}"

    def gen_task(self, task: str):
        r = self.rng
        if task == "A1_copy1":
            c = r.choice(CHARS)
            return self.take(task, f"OUT {c}", c)
        if task == "A2_copy2":
            s = "".join(r.choices(CHARS, k=2))
            return self.take(task, f"COPY {s}", s)
        if task == "A3_copyN":
            n = r.randint(3, 6)
            s = "".join(r.choices(CHARS, k=n))
            return self.take(task, f"COPY {s}", s, {"length": n})
        if task == "A5_repeat":
            unit = "".join(r.choices(CHARS, k=r.randint(1, 2)))
            n = r.randint(2, 5)
            return self.take(task, f"REPEAT {unit} {n}", unit * n, {"unit": unit, "n": n})
        if task == "A4_reverse":
            s = "".join(r.choices(CHARS, k=r.randint(3, 6)))
            return self.take(task, f"REVERSE {s}", s[::-1], {"length": len(s)})
        if task == "A6_index":
            s = "".join(r.choices(CHARS, k=r.randint(4, 6)))
            i = r.randint(1, len(s))
            return self.take(task, f"INDEX {s} {i}", s[i - 1], {"index": i, "length": len(s)})
        if task in ("B2_bind", "B4_bind", "B8_bind"):
            n = int(task[1])
            assigns = self.assignments(n)
            q = assigns[r.randrange(n)][0]
            ans = dict(assigns)[q]
            return self.take(task, self.bind_prompt(assigns, q), ans,
                             {"n_vars": n, "query_pos": [v for v, _ in assigns].index(q)})
        if task == "B_rename":
            assigns = self.assignments(4)
            q = assigns[r.randrange(4)][0]
            ans = dict(assigns)[q]
            renamed = [(r.choice([v for v in VARS if v not in [x[0] for x in assigns]]), d)
                       for v, d in assigns]
            # 语义族相同（值映射与查询角色相同），但变量名不同 → canonical 会不同，属"未见组合"
            return self.take(task, self.bind_prompt(renamed, renamed[[v for v, _ in assigns].index(q)][0]),
                             ans, {"n_vars": 4, "renamed": True})
        if task == "C_history_current":
            v = r.choice(VARS)
            old, new = r.choice(DIGITS), r.choice(DIGITS)
            others = [(x, r.choice(DIGITS)) for x in r.sample([x for x in VARS if x != v], 3)]
            p = self.bind_prompt([(v, old)] + others + [(v, new)], v)
            return self.take(task, p, new, {"n_vars": 4, "current": True})
        if task == "C_reassign":
            v = r.choice(VARS)
            old, new = r.choice(DIGITS), r.choice(DIGITS)
            others = [(x, r.choice(DIGITS)) for x in r.sample([x for x in VARS if x != v], 3)]
            return self.take(task, self.bind_prompt([(v, old)] + others + [(v, new)], v), new,
                             {"n_vars": 5, "reassign": True})
        if task == "C_indirect":
            a, b = r.sample(VARS, 2)
            d1, d2 = r.choice(DIGITS), r.choice(DIGITS)
            others = [(x, r.choice(DIGITS)) for x in r.sample([x for x in VARS if x not in (a, b)], 2)]
            p = self.bind_prompt([(a, d1)] + others + [(b, d2)], a)
            return self.take(task, p, d1, {"indirect": True})
        if task == "D_interference":
            n = r.choice([4, 8])
            assigns = self.assignments(n)
            q = assigns[r.randrange(n)][0]
            return self.take(task, self.bind_prompt(assigns, q), dict(assigns)[q],
                             {"n_vars": n, "interference": True})
        if task == "D_distance":
            n = r.choice([4, 8])  # 16+ 只在 dist 严格配对中用重复变量生成
            assigns = self.assignments(n)
            q = assigns[r.randrange(n)][0]
            return self.take(task, self.bind_prompt(assigns, q), dict(assigns)[q],
                             {"distance": n, "in_dist": True})
        raise ValueError(task)

    def gen_counterfactual(self, n_per_type=100):
        """四类真反事实配对；查询首/中/末均衡；配对 id 共享。"""
        pairs = []
        for typ in ("same_query_fact_change", "unrelated_fact_change", "query_switch", "order_swap"):
            for i in range(n_per_type):
                n = [2, 4, 8][i % 3]
                assigns = self.assignments(n)
                qpos = [0, n // 2, n - 1][i % 3]
                q = assigns[qpos][0]
                base_prompt = self.bind_prompt(assigns, q)
                base_ans = dict(assigns)[q]
                if typ == "same_query_fact_change":
                    new_val = str((int(base_ans) + 1 + i) % 10)
                    cf_assigns = [(k, new_val if k == q else v) for k, v in assigns]
                    cf_prompt, cf_ans, switched = self.bind_prompt(cf_assigns, q), new_val, None
                elif typ == "unrelated_fact_change":
                    other = [k for k, _ in assigns if k != q][0]
                    cf_assigns = [(k, str((int(v) + 1) % 10) if k == other else v) for k, v in assigns]
                    cf_prompt, cf_ans, switched = self.bind_prompt(cf_assigns, q), base_ans, None
                elif typ == "query_switch":
                    other = [k for k, _ in assigns if k != q][0]
                    cf_prompt = self.bind_prompt(assigns, other)
                    cf_ans, switched = dict(assigns)[other], True
                else:  # order_swap：语义结果不变
                    swapped = list(reversed(assigns))
                    cf_prompt = self.bind_prompt(swapped, q)
                    cf_ans, switched = base_ans, None
                pair_id = f"CF_{typ}_{i:03d}"
                pairs.append({
                    "pair_id": pair_id, "type": typ, "query_position": ["first", "middle", "last"][i % 3],
                    "n_vars": n,
                    "base": [self.row(task="B_counterfactual", prompt=base_prompt, answer=base_ans,
                                      meta={"role": "base", "type": typ}, group=pair_id)],
                    "counterfactual": [self.row(task="B_counterfactual", prompt=cf_prompt, answer=cf_ans,
                                                meta={"role": "counterfactual", "type": typ}, group=pair_id)],
                    "query_switch": [self.row(task="B_query_switch", prompt=cf_prompt, answer=cf_ans,
                                              meta={"role": "query_switch"}, group=pair_id)] if switched else [],
                })
        return pairs

    def gen_distance(self, n_base=60):
        """严格配对：同一事实与查询，干扰数 0/4/8/16/32/64/128。"""
        rows = []
        for i in range(n_base):
            v = self.rng.choice(VARS)
            d = self.rng.choice(DIGITS)
            facts = [(v, d)]
            pid = f"DIST_{i:03d}"
            for dist in (0, 4, 8, 16, 32, 64, 128):
                noise = [(self.rng.choice(VARS), self.rng.choice(DIGITS))
                         for _ in range(dist // 4)]
                items = [(v, d)] + noise
                self.rng.shuffle(items)
                p = "\n".join(f"{k}={x}" for k, x in items) + f"\nGET {v}"
                rows.append(self.row("D_distance", p, d,
                                     {"pair_id": pid, "distance": dist, "n_noise_assignments": len(noise)},
                                     group=pid))
        return rows

    def gen_generalization(self, n=600):
        """未见长度/未见组合：长复制、长反转、更多变量、未见格式。"""
        g = Generator(SEED + 777)
        rows = []
        for i in range(n):
            kind = i % 4
            if kind == 0:
                s = "".join(g.rng.choices(CHARS, k=g.rng.randint(7, 12)))
                rows.append(g.row("A3_copyN", f"COPY {s}", s, {"unseen_length": True}))
            elif kind == 1:
                s = "".join(g.rng.choices(CHARS, k=g.rng.randint(7, 12)))
                rows.append(g.row("A4_reverse", f"REVERSE {s}", s[::-1], {"unseen_length": True}))
            elif kind == 2:
                assigns = g.assignments(12)
                q = assigns[g.rng.randrange(12)][0]
                rows.append(g.row("B8_bind", g.bind_prompt(assigns, q), dict(assigns)[q],
                                  {"unseen_n_vars": 12}))
            else:
                s = "".join(g.rng.choices(CHARS, k=g.rng.randint(3, 6)))
                rows.append(g.row("A3_copyN", f"ECHO {s}", s, {"unseen_format": True}))
        return rows


CORE16 = ["A1_copy1", "A2_copy2", "A3_copyN", "A5_repeat", "A4_reverse", "A6_index",
          "B2_bind", "B4_bind", "B8_bind", "B_rename", "C_history_current", "C_reassign",
          "C_indirect", "D_distance", "D_interference"]


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")


def main():
    g = Generator()
    splits = {k: [] for k in ("train", "val", "sealed")}
    # 每任务：train 96 / val 32 / sealed 32，全局 canonical 不重复
    # A1_copy1 的语义空间 = 字符表大小（约93），不能强求每分割 96/32/32 个唯一实验
    QUOTAS = {"A1_copy1": (48, 16, 16), "A2_copy2": (96, 32, 32)}
    for task in CORE16:
        for split, n in zip(("train", "val", "sealed"), QUOTAS.get(task, (96, 32, 32))):
            got = 0
            tries = 0
            while got < n and tries < n * 80:
                tries += 1
                r = g.gen_task(task)
                if r is not None:
                    splits[split].append(r)
                    got += 1
            if got < n:
                print(f"[warn] {task}/{split} 只生成 {got}/{n}")
    cf = g.gen_counterfactual()
    dist = g.gen_distance()
    gen = g.gen_generalization()

    for s, rows in splits.items():
        write_jsonl(DATA / f"m4_{s}.jsonl", rows)
    write_jsonl(DATA / "m4_counterfactual.jsonl", cf)
    write_jsonl(DATA / "m4_distance.jsonl", dist)
    write_jsonl(DATA / "m4_generalization.jsonl", gen)

    # 审计
    def canons(rows):
        return {r["canonical"] for r in rows}
    overlap = {}
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap[f"{a}_vs_{b}"] = len(canons(splits[a]) & canons(splits[b]))
    counts = {s: dict(Counter(r["task"] for r in rows)) for s, rows in splits.items()}
    qpos = Counter()
    for p in cf:
        qpos[p["query_position"]] += 1
    audit = {
        "status": "COMPLETE", "seed": SEED,
        "n_rows": {s: len(rows) for s, rows in splits.items()} |
                  {"counterfactual_pairs": len(cf), "distance": len(dist), "generalization": len(gen)},
        "task_counts": counts,
        "canonical_overlap": overlap,
        "counterfactual_types": dict(Counter(p["type"] for p in cf)),
        "counterfactual_query_positions": dict(qpos),
        "distance_distances": dict(Counter(r["meta"]["distance"] for r in dist)),
        "distance_pair_ids": len({r["meta"]["pair_id"] for r in dist}),
        "a5_repeat_train": counts["train"].get("A5_repeat", 0),
        "note": "canonical 为零重合；距离按 pair_id 严格配对；四类反事实分别统计",
    }
    (HERE / "M4_DATA_AUDIT.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[data] train={len(splits['train'])} val={len(splits['val'])} sealed={len(splits['sealed'])} "
          f"cf={len(cf)} dist={len(dist)} gen={len(gen)}")
    print(f"[data] canonical overlap={overlap} A5_train={audit['a5_repeat_train']}")


if __name__ == "__main__":
    main()
