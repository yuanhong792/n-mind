"""M4/M41F 数据生成与冻结（M41F 语义修复版）。

保留 M4 冻结纪律；M41F 只修复任务语义：
- C_indirect：真实变量引用 + 值复制，源变量后续被覆盖；
- C_history_current：GET v~OLD / GET v~NEW 查询真实历史值/当前值；
- C_reassign：目标重赋值后追加随机无关赋值，破坏末尾数字捷径；
- B_rename：一对一变量重命名，禁止多个原变量映射到同名；
- 所有任务标签由独立参考解释器核验。

参考解释器仅用于数据生成/审计/评测，不参与模型推理。
输出写入 M41F_DATA_AUDIT.json，不覆盖历史 M4_DATA_AUDIT.json。
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from reference_interpreter import canonical as ref_canonical
from reference_interpreter import reference_answer

HERE = Path(__file__).resolve().parents[0]
DATA = HERE / "data"
DATA.mkdir(exist_ok=True)

VARS = list("abcdefghijklmnop")
DIGITS = list("0123456789")
CHARS = [chr(c) for c in range(33, 127) if chr(c) != "="]
SEED = 20261124


class Generator:
    def __init__(self, seed=SEED):
        self.rng = random.Random(seed)
        self.used: set[str] = set()

    def row(self, task, prompt, answer, meta=None, group=None):
        expected = reference_answer(task, prompt)
        if expected != answer:
            raise RuntimeError(f"参考解释器核验失败 {task}: expected={expected!r} got={answer!r} "
                               f"prompt={prompt!r}")
        c = ref_canonical(prompt)
        return {"task": task, "prompt": prompt, "answer": answer,
                "fact_group": group or c, "canonical": c, "meta": meta or {}}

    def take(self, task, prompt, answer, meta=None, group=None):
        c = ref_canonical(prompt)
        if c in self.used:
            return None
        expected = reference_answer(task, prompt)
        if expected != answer:
            raise RuntimeError(f"参考解释器核验失败 {task}: expected={expected!r} got={answer!r} "
                               f"prompt={prompt!r}")
        self.used.add(c)
        return self.row(task, prompt, answer, meta, group)

    def assignments(self, n: int):
        vs = self.rng.sample(VARS, n)
        return [(v, self.rng.choice(DIGITS)) for v in vs]

    def bind_prompt(self, assigns, query, interleave=0):
        items = [f"{v}={d}" for v, d in assigns]
        if interleave:
            noise = [(self.rng.choice(VARS), self.rng.choice(DIGITS))
                     for _ in range(interleave)]
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
            orig = [v for v, _ in assigns]
            pool = [v for v in VARS if v not in orig]
            renamed_vars = r.sample(pool, 4)  # 一对一，且不与原始变量名重合
            mapping = dict(zip(orig, renamed_vars))
            pos = r.randrange(4)
            q_orig = orig[pos]
            q_new = mapping[q_orig]
            ans = dict(assigns)[q_orig]
            renamed = [(mapping[v], d) for v, d in assigns]
            prompt = self.bind_prompt(renamed, q_new)
            return self.take(task, prompt, ans, {"n_vars": 4, "renamed": True,
                                                  "mapping": mapping})
        if task == "C_indirect":
            src, target = r.sample(VARS, 2)
            d1 = r.choice(DIGITS)
            d2 = r.choice([d for d in DIGITS if d != d1])
            noise_vars = r.sample([v for v in VARS if v not in (src, target)], 2)
            items = [(src, d1), (target, src)] + \
                    [(v, r.choice(DIGITS)) for v in noise_vars] + [(src, d2)]
            prompt = self.bind_prompt(items, target)
            return self.take(task, prompt, d1,
                             {"indirect_reference": True, "source": src, "target": target,
                              "copied_value": d1, "source_overwritten_to": d2})
        if task == "C_history_current":
            target = r.choice(VARS)
            old = r.choice(DIGITS)
            new = r.choice([d for d in DIGITS if d != old])
            others = [(v, r.choice(DIGITS)) for v in
                      r.sample([v for v in VARS if v != target], 3)]
            mode = r.choice(["OLD", "NEW"])
            items = [(target, old)] + others + [(target, new)]
            prompt = self.bind_prompt(items, f"{target}~{mode}")
            return self.take(task, prompt, old if mode == "OLD" else new,
                             {"mode": mode, "historical_value": old, "current_value": new,
                              "n_vars": 4})
        if task == "C_reassign":
            target = r.choice(VARS)
            old = r.choice(DIGITS)
            new = r.choice([d for d in DIGITS if d != old])
            used = [target] + r.sample([v for v in VARS if v != target], 3)
            others = [(v, r.choice(DIGITS)) for v in used[1:]]
            trailing_vars = r.sample([v for v in VARS if v not in used], 2)
            trailing = [(v, r.choice([d for d in DIGITS if d != new]))
                        for v in trailing_vars]
            items = [(target, old)] + others + [(target, new)] + trailing
            prompt = self.bind_prompt(items, target)
            return self.take(task, prompt, new,
                             {"reassign": True, "old_value": old, "new_value": new,
                              "trailing_unrelated_assignments": 2})
        if task == "D_interference":
            n = r.choice([4, 8])
            assigns = self.assignments(n)
            q = assigns[r.randrange(n)][0]
            return self.take(task, self.bind_prompt(assigns, q), dict(assigns)[q],
                             {"n_vars": n, "interference": True})
        if task == "D_distance":
            n = r.choice([4, 8])
            assigns = self.assignments(n)
            q = assigns[r.randrange(n)][0]
            return self.take(task, self.bind_prompt(assigns, q), dict(assigns)[q],
                             {"distance": n, "in_dist": True})
        raise ValueError(task)

    def gen_counterfactual(self, n_per_type=100):
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
                    new_val = str((int(base_ans) + 1 + (i % 9)) % 10)
                    cf_assigns = [(k, new_val if k == q else v) for k, v in assigns]
                    cf_prompt, cf_ans, switched = self.bind_prompt(cf_assigns, q), new_val, None
                elif typ == "unrelated_fact_change":
                    other = [k for k, _ in assigns if k != q][0]
                    cf_assigns = [(k, str((int(v) + 1) % 10) if k == other else v)
                                  for k, v in assigns]
                    cf_prompt, cf_ans, switched = self.bind_prompt(cf_assigns, q), base_ans, None
                elif typ == "query_switch":
                    other = [k for k, _ in assigns if k != q][0]
                    cf_prompt = self.bind_prompt(assigns, other)
                    cf_ans, switched = dict(assigns)[other], True
                else:
                    swapped = list(reversed(assigns))
                    cf_prompt = self.bind_prompt(swapped, q)
                    cf_ans, switched = base_ans, None
                pair_id = f"CF_{typ}_{i:03d}"
                pairs.append({
                    "pair_id": pair_id, "type": typ,
                    "query_position": ["first", "middle", "last"][i % 3],
                    "n_vars": n,
                    "base": [self.row("B_counterfactual", base_prompt, base_ans,
                                      {"role": "base", "type": typ}, pair_id)],
                    "counterfactual": [self.row("B_counterfactual", cf_prompt, cf_ans,
                                                {"role": "counterfactual", "type": typ}, pair_id)],
                    "query_switch": [self.row("B_query_switch", cf_prompt, cf_ans,
                                              {"role": "query_switch"}, pair_id)]
                    if switched else [],
                })
        return pairs

    def gen_distance(self, n_base=60):
        rows = []
        for i in range(n_base):
            v = self.rng.choice(VARS)
            d = self.rng.choice(DIGITS)
            pid = f"DIST_{i:03d}"
            for dist in (0, 4, 8, 16, 32, 64, 128):
                # 干扰变量不得覆盖目标事实，否则答案会被破坏。
                pool = [x for x in VARS if x != v]
                noise = [(self.rng.choice(pool), self.rng.choice(DIGITS))
                         for _ in range(dist // 4)]
                items = [(v, d)] + noise
                self.rng.shuffle(items)
                p = "\n".join(f"{k}={x}" for k, x in items) + f"\nGET {v}"
                rows.append(self.row("D_distance", p, d,
                                     {"pair_id": pid, "distance": dist,
                                      "n_noise_assignments": len(noise)}, pid))
        return rows

    def gen_generalization(self, n=600):
        g = Generator(SEED + 777)
        rows = []
        for i in range(n):
            kind = i % 4
            if kind == 0:
                s = "".join(g.rng.choices(CHARS, k=g.rng.randint(7, 12)))
                rows.append(g.row("A3_copyN", f"COPY {s}", s, {"unseen_length": True}))
            elif kind == 1:
                s = "".join(g.rng.choices(CHARS, k=g.rng.randint(7, 12)))
                rows.append(g.row("A4_reverse", f"REVERSE {s}", s[::-1],
                                  {"unseen_length": True}))
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
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                    encoding="utf-8")


def main():
    g = Generator()
    splits = {k: [] for k in ("train", "val", "sealed")}
    QUOTAS = {"A1_copy1": (48, 16, 16), "A2_copy2": (96, 32, 32)}
    for task in CORE16:
        for split, n in zip(("train", "val", "sealed"), QUOTAS.get(task, (96, 32, 32))):
            got, tries = 0, 0
            while got < n and tries < n * 200:
                tries += 1
                r = g.gen_task(task)
                if r is not None:
                    splits[split].append(r)
                    got += 1
            if got < n:
                raise RuntimeError(f"{task}/{split} 只生成 {got}/{n}，语义修复后唯一空间不足")

    cf = g.gen_counterfactual()
    dist = g.gen_distance()
    gen = g.gen_generalization()
    for s, rows in splits.items():
        write_jsonl(DATA / f"m4_{s}.jsonl", rows)
    write_jsonl(DATA / "m4_counterfactual.jsonl", cf)
    write_jsonl(DATA / "m4_distance.jsonl", dist)
    write_jsonl(DATA / "m4_generalization.jsonl", gen)

    def canons(rows):
        return {r["canonical"] for r in rows}
    overlap = {}
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap[f"{a}_vs_{b}"] = len(canons(splits[a]) & canons(splits[b]))
    audit = {
        "status": "COMPLETE", "seed": SEED,
        "n_rows": {s: len(rows) for s, rows in splits.items()} |
                  {"counterfactual_pairs": len(cf), "distance": len(dist),
                   "generalization": len(gen)},
        "task_counts": {s: dict(Counter(r["task"] for r in rows))
                        for s, rows in splits.items()},
        "canonical_overlap": overlap,
        "semantics": {
            "C_indirect": "变量引用值复制，源变量后续覆盖",
            "C_history_current": "GET v~OLD / GET v~NEW 查询历史/当前值",
            "C_reassign": "目标重赋值后追加无关赋值",
            "B_rename": "一对一变量重命名",
        },
        "reference_interpreter": "reference_interpreter.reference_answer（仅数据/评测）",
        "note": "M41F 重新冻结；历史 M4_DATA_AUDIT.json 与 M41 审计未覆盖。",
    }
    (HERE / "M41F_DATA_AUDIT.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[data-m41f] train={len(splits['train'])} val={len(splits['val'])} "
          f"sealed={len(splits['sealed'])} canonical_overlap={overlap}")


if __name__ == "__main__":
    main()
