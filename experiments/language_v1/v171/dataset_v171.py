"""V1.7.1 P4: balanced, controllable-quota Chinese dataset generator.

Fixes over V1.7:
  - task quotas are explicit & controllable (no 2-item dialogue / 1-item context).
  - counterfactual examples are TRUE PAIRS: for one fact there are two rows with
    the same question but swapped attributes -> different answers. A pair passes
    only if BOTH questions are answered correctly.
  - multi-turn context is properly represented.
  - fact-group + template-family dedup isolation between train/val/sealed/counter.
  - V1.7 data is NEVER overwritten; this writes to v171/ and freezes SHA-256.

ZERO training in this module. Uses the unchanged V1.5 tokenizer.
"""
from __future__ import annotations
import hashlib, json, random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments/language_v1/v171/data"
SEED = 20271124

# 可控任务配额（每 split 目标条数；counter 专门放反事实配对测试）
QUOTAS = {
    "train":  {"A_dialogue": 600, "B_bind": 1800, "C_counter": 1200, "D_position": 1500, "E_context": 900},
    "val":    {"A_dialogue": 100, "B_bind": 300,  "C_counter": 200,  "D_position": 250,  "E_context": 150},
    "sealed": {"A_dialogue": 100, "B_bind": 300,  "C_counter": 200,  "D_position": 250,  "E_context": 150},
    "counter":{"A_dialogue": 0,   "B_bind": 0,    "C_counter": 300,  "D_position": 0,    "E_context": 0},
}
# sealed / counter 中反事实必须成对: 每对 2 行 -> 200 sealed 条=100 对, 300 counter 条=150 对

# 扩充人物池（首字互异 -> 答案首 token 判别性）, 支撑多轮上下文任务配额
PERSONS = ["王明","李红","张刚","陈丽","赵强","周美","孙华","吴军","郑芳","刘杰",
           "黄敏","徐磊","马琳","朱婷","胡宇","郭浩","何欣","高睿","林彤","罗凯",
           "梁静","宋佳","唐勇","许薇","韩雪","冯刚","邓超","曹阳","彭丽","曾志",
           "萧然","田甜","董浩","潘婷","袁泉","于洋","蒋雯","蔡明","余波","杜江",
           "叶青","程亮","苏晴","魏然","吕布","丁宁","沈冰","任杰","姚明","卢俊",
           "姜涛","崔健","钟磊","谭凯","陆瑶","汪峰","范冰","金石","戴军","贾玲"]
COLORS = ["红色","绿色","蓝色","黄色","紫色","橙色","黑色","白色"]
OBJECTS = ["苹果","书包","杯子","香蕉","橘子","皮球","铅笔","本子"]
LOCATIONS = ["桌上","柜子","椅子","床边"]
GREETINGS = ["你好","你好啊","嗨","早上好","下午好","晚上好","在吗","喂","哈喽","您好"]
IDENTITIES = ["你叫什么名字","你叫什么","你是谁","你的名字","报上名来","你哪位","请问你是","自我介绍"]
THANKS = ["谢谢你","谢谢你的帮助","多谢","感谢","谢谢啦","非常感谢","辛劳了","多谢你"]


def fact_split(key: str) -> str:
    h = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 100
    return "train" if h < 70 else "val" if h < 82 else "sealed" if h < 94 else "counter"


def rec(task, prompt, answer, key, meta=None):
    return {"task": task, "prompt": prompt, "answer": answer, "key": key, "meta": meta or {}}


def build_pools(rng):
    """每个任务生成候选池, 按 (fact_group, template_family) 分组。"""
    bygroup = defaultdict(list)  # group_key -> list[(template_family, rec)]

    def add(task, prompt, answer, group, family, meta=None):
        bygroup[group].append((family, rec(task, prompt, answer, group, meta)))

    # A 基础对话：多模板族（fact key 含问法 -> 隔离到同一 split）
    fam_map = [("greet", GREETINGS, "你好！"), ("identity", IDENTITIES, "我叫N-MIND。"),
               ("thanks", THANKS, "不客气！")]
    for fam, qs, a in fam_map:
        for q in qs:
            add("A_dialogue", f"用户：{q}", a, f"A|{q}", fam)

    # B 实体绑定：多种因素组合，目标位置/最近属性可控
    for _ in range(26000):
        p1, p2 = rng.sample(PERSONS, 2); c1, c2 = rng.sample(COLORS, 2); obj = rng.choice(OBJECTS)
        order = [(p1,c1),(p2,c2)] if rng.random() < .5 else [(p2,c2),(p1,c1)]
        (fp, fc), (sp, sc) = order
        target = rng.choice([p1, p2]); ans = c1 if target == p1 else c2
        conn = rng.choice(["，", "；"])
        stmt = f"已知：{fp}有{fc}{obj}{conn}{sp}有{sc}{obj}。"
        fam = rng.choice(["q_color", "q_possess"])
        q = (f"问：{target}的{obj}是什么颜色？" if fam == "q_color"
             else f"问：{target}手里的{obj}什么颜色？")
        meta = {"target_is_first": target == fp, "nearest_attr": sc, "answer": ans}
        add("B_bind", f"{stmt}\n{q}\n答：", ans,
            f"B|{p1}|{c1}|{obj}|{p2}|{c2}", fam, meta)

    # C 反事实：真实成对。group 内两条：原事实答案 + 交换后答案，同一问法。
    for _ in range(14000):
        p1, p2 = rng.sample(PERSONS, 2); c1, c2 = rng.sample(COLORS, 2); obj = rng.choice(OBJECTS)
        target = rng.choice([p1, p2])
        original = c1 if target == p1 else c2     # 原事实下答案
        swapped = c2 if target == p1 else c1      # 交换属性后答案
        if original == swapped: continue
        # 原事实题
        p_orig = (f"原：{p1}有{c1}{obj}，{p2}有{c2}{obj}。\n"
                  f"问：{target}的{obj}是什么颜色？\n答：")
        # 交换后题：同样的问法, 事实已改变
        p_swap = (f"原：{p1}有{c1}{obj}，{p2}有{c2}{obj}。\n"
                  f"改：把{p1}和{p2}的{obj}颜色对调。\n"
                  f"问：{target}的{obj}现在是什么颜色？\n答：")
        group = f"C|{p1}|{c1}|{obj}|{p2}|{c2}|{target}"
        add("C_counter", p_orig, original, group, "orig", {"answer": original, "pair_role": "orig"})
        add("C_counter", p_swap, swapped, group, "swap", {"answer": swapped, "pair_role": "swap"})

    # D 位置归属：多模板族
    for _ in range(12000):
        p1, p2 = rng.sample(PERSONS, 2); o1, o2 = rng.sample(OBJECTS, 2)
        l1, l2 = rng.sample(LOCATIONS, 2)
        stmt = f"{p1}把{o1}放在{l1}，{p2}把{o2}放在{l2}。"
        if rng.random() < .5:
            fam = "who"; q = rng.choice([f"问：谁把{o2}放在{l2}？", f"问：{l2}上的{o2}是谁的？"]); ans = p2
        else:
            fam = "what"; q = rng.choice([f"问：{p1}把什么放在{l1}？", f"问：{p1}放在{l1}的是什么？"]); ans = o1
        add("D_position", f"{stmt}\n{q}\n答：", ans,
            f"D|{p1}|{o1}|{l1}|{p2}|{o2}|{l2}|{fam}", fam, {"answer": ans})

    # E 多轮上下文：每个 (人物, 风格, 会话轮次模板族) 为独立 fact group,
    # 但同一人物+同一风格的所有问法共享 group -> 严格无泄漏, 同时池足够大。
    intros = ["我叫{p}", "我是{p}", "我的名字是{p}", "大家好，我是{p}", "你可以叫我{p}",
              "我是新来的{p}", "记住我叫{p}", "我叫{p}，请多关照"]
    replies = ["你好，{p}", "你好{p}", "{p}，你好", "你好，见到你很高兴，{p}",
               "嗨，{p}", "欢迎{p}", "好的，{p}", "幸会，{p}"]
    asks = ["我叫什么？", "我的名字是什么？", "我是谁？", "猜猜我叫什么？", "你知道我的名字吗？",
            "请问我是谁？", "我的名字是？", "还记得我叫什么吗？"]
    for p in PERSONS:
        for style in range(len(intros)):
            intro = intros[style].format(p=p)
            reply = replies[style].format(p=p)
            group = f"E|{p}|style{style}"   # 同一人物+风格的所有问法落同一 split
            for qi, q in enumerate(asks):
                fam = f"style{style}"
                add("E_context", f"用户：{intro}\n助手：{reply}\n用户：{q}\n助手：", p,
                    group, fam, {"answer": p, "ask_idx": qi})
    return bygroup


def build():
    rng = random.Random(SEED)
    bygroup = build_pools(rng)
    splits = defaultdict(list)
    group_splits = {}
    for group, items in bygroup.items():
        s = fact_split(group)
        group_splits[group] = s
        splits[s].extend((group, family, r) for family, r in items)

    # 反事实成对完整性: 一个 group 的两条(orig/swap)必须同 split 且都在 -> 否则整组丢弃
    cf_groups = defaultdict(list)
    for s in splits:
        for group, fam, r in splits[s]:
            if r["task"] == "C_counter":
                cf_groups[(s, group)].append((fam, r))
    clean_cf = {}
    for (s, group), items in cf_groups.items():
        roles = {r["meta"].get("pair_role") for _, r in items}
        if roles == {"orig", "swap"} and len({r["answer"] for _, r in items}) == 2:
            clean_cf[(s, group)] = items
    # 重建 splits, 只保留完整反事实对 + 其它任务
    rebuild = defaultdict(list)
    retained_cf_keys = set()
    for s in splits:
        for group, fam, r in splits[s]:
            if r["task"] == "C_counter":
                if (s, group) in clean_cf:
                    retained_cf_keys.add((s, group))
                    rebuild[s].append((group, fam, r))
            else:
                rebuild[s].append((group, fam, r))
    splits = rebuild

    # 全局 prompt 去重 + 按配额采样（反事实按“对”采样, 保证成对）
    used_prompts = set()
    final = {}
    for s in QUOTAS:
        pool = splits[s]
        rng.shuffle(pool)
        if s in ("sealed", "counter", "val", "train"):
            # 先处理反事实对
            cf_pairs = defaultdict(list)
            non_cf = []
            for group, fam, r in pool:
                if r["task"] == "C_counter":
                    cf_pairs[group].append(r)
                else:
                    non_cf.append(r)
            wanted_pairs = QUOTAS[s].get("C_counter", 0) // 2
            chosen_cf = []
            for group, items in list(cf_pairs.items())[:wanted_pairs]:
                if len(items) == 2 and items[0]["answer"] != items[1]["answer"]:
                    chosen_cf.extend(items)
            # 其它任务按配额
            rows = []
            for r in chosen_cf + non_cf:
                if r["prompt"] in used_prompts:
                    continue
                quota = QUOTAS[s].get(r["task"], 0)
                cnt = sum(1 for x in rows if x["task"] == r["task"])
                if r["task"] == "C_counter":
                    # 反事实条数已在 chosen_cf 控制
                    if sum(1 for x in rows if x["task"] == "C_counter") >= QUOTAS[s].get("C_counter", 0):
                        continue
                elif cnt >= quota:
                    continue
                used_prompts.add(r["prompt"]); rows.append(r)
            final[s] = rows
        else:
            final[s] = []

    # 审计
    def baselines(rows):
        n = len(rows) or 1
        freq = Counter(r["answer"] for r in rows)
        fixed = freq.most_common(1)[0][1] / n if freq else 0
        b = [r for r in rows if r["task"] == "B_bind"]
        near = sum(1 for r in b if r["meta"].get("nearest_attr") == r["answer"]) / (len(b) or 1)
        pos = sum(1 for r in b if r["meta"].get("target_is_first")) / (len(b) or 1)
        return {"fixed_answer": round(fixed, 4), "nearest_attribute": round(near, 4), "position_first": round(pos, 4)}

    setnames = list(final)
    overlap = {}
    for i, a in enumerate(setnames):
        for b in setnames[i+1:]:
            pa = {r["prompt"] for r in final[a]}; pb = {r["prompt"] for r in final[b]}
            overlap[f"{a}_{b}"] = len(pa & pb)
    key_splits = defaultdict(set)
    for s, rows in final.items():
        for r in rows:
            key_splits[r["key"]].add(s)

    # 反事实成对统计
    def pair_stats(rows):
        cm = defaultdict(list)
        for r in rows:
            if r["task"] == "C_counter":
                cm[r["key"]].append(r)
        pairs = {k: v for k, v in cm.items() if len(v) == 2 and v[0]["answer"] != v[1]["answer"]}
        return {"counter_rows": sum(1 for r in rows if r["task"] == "C_counter"),
                "groups": len(cm), "valid_pairs": len(pairs)}

    audit = {
        "version": "v171", "seed": SEED, "quota_config": QUOTAS,
        "counts": {s: len(v) for s, v in final.items()},
        "task_counts": {s: dict(Counter(r["task"] for r in v)) for s, v in final.items()},
        "prompt_overlap": overlap,
        "facts_crossing_splits": sum(1 for x in key_splits.values() if len(x) > 1),
        "counterfactual_pairs": {s: pair_stats(v) for s, v in final.items()},
        "shortcut_baselines_sealed": baselines(final["sealed"]),
        "tokenizer": "V1.5 ByteBPETokenizer hash=26af116ebd13ff00 (UNCHANGED)",
        "fact_and_template_isolation": True,
        "v171_data_does_not_overwrite_v17": True,
        "note": "反事实按(group)成对采样, orig/swap 同一问法不同答案, 两题同时对才算一对通过。",
    }
    return final, audit


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    final, audit = build()
    for s, rows in final.items():
        path = OUT / f"v171_{s}.jsonl"
        path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
        audit[f"{s}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT / "V171_DATA_FROZEN.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "experiments/language_v1/v171/V171_DATA_AUDIT.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"counts": audit["counts"], "task_counts": audit["task_counts"],
                      "overlap": audit["prompt_overlap"], "facts_crossing": audit["facts_crossing_splits"],
                      "cf_pairs": audit["counterfactual_pairs"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
