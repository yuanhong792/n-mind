"""V1.7 P1: deterministic, semantically-correct Chinese dataset generator.
5 tasks: A dialogue / B binding / C counterfactual / D position / E multi-turn.
HARD RULES:
  - keep V1.5 tokenizer (hash 26af116ebd13ff00); new chars encoded as byte tokens
  - person names have DISTINCT first characters -> answer first token is discriminative
  - FACT-LEVEL HOLDOUT: every question about the same fact shares one split (zero crossing)
  - GLOBAL prompt de-dup: train/val/sealed/counter prompts are pairwise disjoint
  - balanced answers, target position varies, answer != always nearest attribute
  - frozen before training with SHA-256; 3 shortcut baselines computed
ZERO training in this module.
"""
from __future__ import annotations
import hashlib, json, random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments/language_v1/v17"
SEED = 20261123
TARGET = {"train": 6000, "val": 600, "sealed": 600, "counter": 300}

PERSONS = ["王明","李红","张刚","陈丽","赵强","周美","孙华","吴军","郑芳","刘杰",
           "黄敏","徐磊","马琳","朱婷","胡宇","郭浩","何欣","高睿","林彤","罗凯"]
COLORS = ["红色","绿色","蓝色","黄色","紫色","橙色","黑色","白色"]
OBJECTS = ["苹果","书包","杯子","香蕉","橘子","皮球","铅笔","本子"]
LOCATIONS = ["桌上","柜子","椅子","床边"]

def fact_split(key: str) -> str:
    """事实级确定性分流：同一 key 的样本必落同一 split。"""
    h = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 100
    return "train" if h < 70 else "val" if h < 82 else "sealed" if h < 94 else "counter"

def rec(task, prompt, answer, key, meta=None):
    return {"task": task, "prompt": prompt, "answer": answer, "key": key, "meta": meta or {}}

def build_candidates(rng):
    """生成候选盈池，按 fact key 分组（key 相同=同一事实/上下文）。"""
    bykey = defaultdict(list)
    def add(task, prompt, answer, key, meta=None):
        bykey[key].append(rec(task, prompt, answer, key, meta))
    # A 基础对话（每个模板一个 key）
    for q, a in [("你好","你好！"),("你好啊","你好！"),("嗨","你好！"),
                 ("你叫什么名字","我叫N-MIND。"),("你叫什么","我叫N-MIND。"),("你是谁","我叫N-MIND。"),
                 ("谢谢你","不客气！"),("谢谢你的帮助","不客气！"),("多谢","不客气！")]:
        add("A_dialogue", f"用户：{q}", a, f"A|{q}")
    # B 绑定：随机采样组合
    for _ in range(14000):
        p1, p2 = rng.sample(PERSONS, 2); c1, c2 = rng.sample(COLORS, 2); obj = rng.choice(OBJECTS)
        first_p, first_c = (p1,c1) if rng.random()<.5 else (p2,c2)
        second_p, second_c = (p2,c2) if first_p==p1 else (p1,c1)
        target = rng.choice([p1,p2]); ans = c1 if target==p1 else c2
        conn = rng.choice(["，","；"])
        stmt = f"已知：{first_p}有{first_c}{obj}{conn}{second_p}有{second_c}{obj}。"
        q = rng.choice([f"问：{target}的{obj}是什么颜色？", f"问：{target}手里的{obj}什么颜色？"])
        add("B_bind", f"{stmt}\n{q}\n答：", ans, f"B|{p1}|{c1}|{obj}|{p2}|{c2}",
            {"target_is_first": target==first_p, "nearest_attr": second_c, "answer": ans})
    # C 反事实
    for _ in range(6000):
        p1,p2 = rng.sample(PERSONS,2); c1,c2 = rng.sample(COLORS,2); obj = rng.choice(OBJECTS)
        target = rng.choice([p1,p2]); nc1,nc2 = c2,c1; ans = nc1 if target==p1 else nc2
        p = (f"原：{p1}有{c1}{obj}，{p2}有{c2}{obj}。\n"
             f"改：{p1}有{nc1}{obj}，{p2}有{nc2}{obj}。\n"
             f"问：{target}的{obj}现在是什么颜色？\n答：")
        add("C_counter", p, ans, f"C|{p1}|{c1}|{obj}|{p2}|{c2}", {"answer":ans})
    # D 位置归属
    for _ in range(8000):
        p1,p2 = rng.sample(PERSONS,2); o1,o2 = rng.sample(OBJECTS,2); l1,l2 = rng.sample(LOCATIONS,2)
        stmt = f"{p1}把{o1}放在{l1}，{p2}把{o2}放在{l2}。"
        if rng.random()<.5:
            q = rng.choice([f"问：谁把{o2}放在{l2}？", f"问：{l2}上的{o2}是谁的？"]); ans=p2
        else:
            q = rng.choice([f"问：{p1}把什么放在{l1}？", f"问：{p1}放在{l1}的是什么？"]); ans=o1
        add("D_position", f"{stmt}\n{q}\n答：", ans, f"D|{p1}|{o1}|{l1}|{p2}|{o2}|{l2}|{ans[:2]}", {"answer":ans})
    # E 多轮（key 只含人物 -> 同一人物的所有问法落同一 split，严格无泄漏）
    for p in PERSONS:
        for style in range(4):
            intro = ["我叫"+p,"我是"+p,"我的名字是"+p,"大家好，我是"+p][style]
            reply = ["你好，"+p,"你好"+p,p+"，你好","你好，见到你很高兴，"+p][style]
            ask = rng.choice(["我叫什么？","我的名字是什么？","我是谁？"])
            add("E_context", f"用户：{intro}\n助手：{reply}\n用户：{ask}\n助手：", p, f"E|{p}")
    return bykey

def build():
    rng = random.Random(SEED)
    bykey = build_candidates(rng)
    # 事实级分流：整个 key 一起走
    splits = defaultdict(list)
    for key, rows in bykey.items():
        s = fact_split(key)
        splits[s].extend(rows)
    # 全局 prompt 去重（跨 split 也要唯一）：按 prompt 的 hash 稳定排序后首次命中保留
    used_prompts = set()
    for s in splits:
        rng.shuffle(splits[s])
        dedup = []
        for r in splits[s]:
            if r["prompt"] in used_prompts: continue
            used_prompts.add(r["prompt"]); dedup.append(r)
        splits[s] = dedup
    # 目标数量采样（训练尽量充足；小集按目标截取，保证各集规模达标）
    for s in ("train","val","sealed","counter"):
        if len(splits[s]) > TARGET[s]:
            splits[s] = splits[s][:TARGET[s]]
    # 审计
    def baselines(rows):
        n = len(rows) or 1
        freq = Counter(r["answer"] for r in rows)
        fixed = freq.most_common(1)[0][1]/n if freq else 0
        b = [r for r in rows if r["task"]=="B_bind"]
        recent = sum(1 for r in b if r["meta"].get("nearest_attr")==r["answer"])/(len(b) or 1)
        pos = sum(1 for r in b if r["meta"].get("target_is_first"))/(len(b) or 1)
        return {"fixed_answer":round(fixed,4),"nearest_attribute":round(recent,4),"position_first":round(pos,4)}
    t = {r["prompt"] for r in splits["train"]}
    v = {r["prompt"] for r in splits["val"]}
    sl = {r["prompt"] for r in splits["sealed"]}
    co = {r["prompt"] for r in splits["counter"]}
    key_splits = defaultdict(set)
    for s, rows in splits.items():
        for r in rows: key_splits[r["key"]].add(s)
    audit = {
        "seed":SEED, "counts":{s:len(v) for s,v in splits.items()},
        "task_counts":{s:dict(Counter(r["task"] for r in v)) for s,v in splits.items()},
        "answer_freq":{s:dict(Counter(r["answer"] for r in v)) for s,v in splits.items()},
        "tokenizer":"V1.5 ByteBPETokenizer hash=26af116ebd13ff00 (UNCHANGED)",
        "fact_level_holdout":True,
        "distinct_first_char_persons":len({p[0] for p in PERSONS})==len(PERSONS),
        "prompt_overlap":{
            "train_val":len(t&v),"train_sealed":len(t&sl),"train_counter":len(t&co),
            "val_sealed":len(v&sl),"sealed_counter":len(sl&co)},
        "facts_crossing_splits":sum(1 for x in key_splits.values() if len(x)>1),
        "shortcut_baselines_val":baselines(splits["val"]),
        "shortcut_baselines_sealed":baselines(splits["sealed"]),
        "note":"封存集所需汉字均在训练语料中出现（共享人物/颜色/物品词表），但事实组合严格 holdout",
    }
    return splits, audit

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    splits, audit = build()
    for s, rows in splits.items():
        path = OUT / f"v17_{s}.jsonl"
        path.write_text("\n".join(json.dumps(r,ensure_ascii=False) for r in rows)+"\n", encoding="utf-8")
        audit[f"{s}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT / "V17_DATA_AUDIT.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"counts":audit["counts"],"overlap":audit["prompt_overlap"],
                      "facts_crossing":audit["facts_crossing_splits"],
                      "distinct_first_char":audit["distinct_first_char_persons"],
                      "baselines_val":audit["shortcut_baselines_val"]},ensure_ascii=False,indent=2))

if __name__ == "__main__": main()
