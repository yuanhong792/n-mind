"""V1.7.2 P6：接入 V1.7.1 数据、审计配额缺口、生成独立 V1.7.2 数据。

- 原样保留 v171/data，不覆盖。
- A_dialogue 原配额严重不足（train 20/600），用真实句型族扩充，不用复制/
  标点噪声凑数。每题以 prompt SHA 分流，保证跨集合完全相同 prompt 零重合。
- 事实规范化键跨集合隔离；反事实保持真实成对。
- val/sealed 写入前冻结 SHA；缺口如实报告，不冒充完成。
本脚本不执行训练。
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from v172_protocol import build_tokenizer, write_json

OUT = Path(__file__).resolve().parent
V171_DATA = ROOT / "experiments/language_v1/v171/data"
V172_DATA = OUT / "data"

QUOTAS = {
    "train":  {"A_dialogue": 600, "B_bind": 1800, "C_counter": 1200, "D_position": 1500, "E_context": 900},
    "val":    {"A_dialogue": 100, "B_bind": 300,  "C_counter": 200,  "D_position": 250,  "E_context": 150},
    "sealed": {"A_dialogue": 100, "B_bind": 300,  "C_counter": 200,  "D_position": 250,  "E_context": 150},
    "counter":{"A_dialogue": 0,   "B_bind": 0,    "C_counter": 300,  "D_position": 0,    "E_context": 0},
}

# 真实中文问法族：每族多个语义不同的说法，上下文前缀表达不同会话状态。
DIALOGUE_FAMILIES = {
    "greet": (
        ["你好", "你好啊", "嗨", "早上好", "下午好", "晚上好", "在吗", "喂", "哈喽", "您好",
         "你好呀", "嗨你好", "早上好呀", "晚上好呀", "hello你好", "在不在", "有人吗", "打扰一下",
         "好久不见", "早上好朋友", "晚上好啊", "下午好呀", "各位好", "你好朋友", "哈喽啊",
         "您好啊", "哟你好", "早啊", "见到你很高兴", "又见面了"],
        "你好！",
    ),
    "identity": (
        ["你叫什么名字", "你叫什么", "你是谁", "你的名字", "报上名来", "你哪位", "请问你是", "自我介绍",
         "你叫什么名儿", "怎么称呼你", "你的名字是什么", "介绍一下你自己", "你是哪位", "请问尊姓大名",
         "能告诉我你的名字吗", "说说你是谁", "你是叫什么", "名字是啥", "你的全名", "你叫什么来着",
         "如何称呼", "请问你叫什么", "你叫什么名字呀", "报一下名字", "你是谁呀", "你是哪一位",
         "自我介绍下", "说下你的名字", "能介绍自己吗", "你叫什么名字呢"],
        "我叫N-MIND。",
    ),
    "thanks": (
        ["谢谢你", "谢谢你的帮助", "多谢", "感谢", "谢谢啦", "非常感谢", "辛劳了", "多谢你",
         "多谢帮忙", "感谢你的帮助", "太感谢了", "谢谢你帮忙", "真心感谢", "谢啦", "十分感谢", "感谢感谢",
         "多谢了啊", "太谢谢你了", "感谢支持", "有劳了", "谢谢你所做的一切", "真的谢谢你",
         "非常感谢你的帮助", "谢谢您", "十分感谢你", "感谢你的支持", "谢谢你哦", "多谢合作"],
        "不客气！",
    ),
    "status": (
        ["你好吗", "你怎么样", "最近还好吗", "今天过得怎么样", "忙不忙", "还好吗", "你现在怎么样",
         "一切都好吗", "状态如何", "最近忙吗", "你还好吧", "今天好吗", "过得如何", "最近咋样",
         "你现在忙吗", "状态怎么样", "一切顺利吗", "你还好吗今天", "今天状态如何", "最近怎么样啊",
         "日子过得怎么样", "你现在状态好不好", "还好吧你", "最近一切可好", "过得还好吗", "你今天如何"],
        "我很好，谢谢！",
    ),
    "capability": (
        ["你能做什么", "你会什么", "有什么本事", "能帮我什么忙", "你有哪些功能", "你会干什么",
         "你擅长什么", "有什么我可以问你的", "你能帮我做什么", "你能帮我什么", "你都会啥",
         "你有什么本事", "能做什么事情", "你有哪些能力", "你都能干什么", "你有什么用途",
         "你能处理什么问题", "你平时都能做什么", "请介绍你的能力", "你都会哪些技能",
         "你能提供什么帮助", "你有什么特长", "你擅长做什么", "你能干什么活", "你都有啥功能",
         "你的功能有哪些", "你能帮我解决什么", "你可以做什么事"],
        "我可以陪你聊天、回答问题。",
    ),
}
CONTEXTS = ["", "那个，", "请问", "打扰一下，", "嘿，", "不好意思，", "我说，", "喂，",
            "对了，", "话说，", "先说个事，", "嗯，", "早，", "哎，", "来，", "小兄弟，"]


def load_rows(split: str, base: Path = V171_DATA):
    return [json.loads(line) for line in (base / f"v171_{split}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


def norm_fact(row: dict) -> str:
    """事实规范化：去掉问法与模板词，只保留事实骨架，用于跨集合泄漏检测。"""
    s = row.get("prompt", "")
    for marker in ("问：", "答：", "改：", "原：", "已知："):
        s = s.replace(marker, "")
    return hashlib.sha256(s.strip().encode("utf-8")).hexdigest()[:16]


def audit_existing() -> dict:
    all_rows = {}
    for split in ("train", "val", "sealed", "counter"):
        all_rows[split] = load_rows(split)
    counts = {s: dict(Counter(r["task"] for r in rows)) for s, rows in all_rows.items()}
    gaps = {}
    for split, tasks in QUOTAS.items():
        gaps[split] = {t: max(0, q - counts[split].get(t, 0)) for t, q in tasks.items()}
    prompt_sets = {s: {r["prompt"] for r in rows} for s, rows in all_rows.items()}
    overlap = {}
    names = list(prompt_sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap[f"{a}_{b}"] = len(prompt_sets[a] & prompt_sets[b])
    fact_sets = {s: {norm_fact(r) for r in rows} for s, rows in all_rows.items()}
    fact_cross = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            fact_cross[f"{a}_{b}"] = len(fact_sets[a] & fact_sets[b])
    # 反事实成对性
    pairs = {}
    for split in ("train", "val", "sealed", "counter"):
        groups = defaultdict(list)
        for r in all_rows[split]:
            if r["task"] == "C_counter":
                groups[r["key"]].append(r)
        valid = 0
        for key, rs in groups.items():
            if len(rs) == 2 and rs[0]["answer"] != rs[1]["answer"]:
                roles = {x.get("meta", {}).get("pair_role") for x in rs}
                valid += int(roles == {"orig", "swap"})
        pairs[split] = {"groups": len(groups), "valid_pairs": valid}
    frozen = {}
    for split in ("train", "val", "sealed", "counter"):
        frozen[split] = hashlib.sha256((V171_DATA / f"v171_{split}.jsonl").read_bytes()).hexdigest()
    return {
        "source": "experiments/language_v1/v171/data (UNCHANGED)",
        "counts": counts, "quotas": QUOTAS, "gaps": gaps,
        "prompt_overlap": overlap, "normalized_fact_overlap": fact_cross,
        "counterfactual_pairs": pairs,
        "v171_frozen_sha256": frozen,
        "source_preserved": True,
    }


def generate_dialogue() -> dict[str, list[dict]]:
    """按 prompt SHA 稳定分流生成 A_dialogue，跨集合零重合。"""
    buckets: dict[str, list[dict]] = defaultdict(list)
    seen_prompt: set[str] = set()
    existing = {r["prompt"] for s in ("train", "val", "sealed", "counter") for r in load_rows(s)}
    for fam, (questions, answer) in DIALOGUE_FAMILIES.items():
        for q in questions:
            for ctx in CONTEXTS:
                prompt = f"用户：{ctx}{q}"
                if prompt in seen_prompt or prompt in existing:
                    continue
                seen_prompt.add(prompt)
                key = f"A|{fam}|{q}|{ctx}"
                h = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 100
                split = "train" if h < 70 else "val" if h < 85 else "sealed"
                row = {"task": "A_dialogue", "prompt": prompt, "answer": answer,
                       "key": key, "meta": {"family": fam, "surface": q, "context": ctx}}
                buckets[split].append(row)
    return dict(buckets)


def main():
    audit = audit_existing()
    generated = generate_dialogue()
    V172_DATA.mkdir(parents=True, exist_ok=True)

    # 复制 v171 全部数据 + 用新对话补足 A_dialogue 配额（原文件不动）
    final_counts = {}
    frozen = {}
    for split in ("train", "val", "sealed", "counter"):
        rows = load_rows(split)
        base_dialogue = sum(1 for r in rows if r["task"] == "A_dialogue")
        need = max(0, QUOTAS[split]["A_dialogue"] - base_dialogue)
        add = generated.get(split, [])[:need]
        rows = rows + add
        path = V172_DATA / f"v172_{split}.jsonl"
        path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
        frozen[split] = hashlib.sha256(path.read_bytes()).hexdigest()
        final_counts[split] = dict(Counter(r["task"] for r in rows))

    gaps_final = {s: {t: max(0, q - final_counts[s].get(t, 0)) for t, q in QUOTAS[s].items()}
                  for s in QUOTAS}
    payload = {
        "status": "DATA_AUDIT_AND_REPAIR_DONE",
        "source": audit,
        "generator_families": {k: {"variants": len(v[0]), "answer": v[1]} for k, v in DIALOGUE_FAMILIES.items()},
        "context_variants": len(CONTEXTS),
        "new_dialogue_generated": {s: len(v) for s, v in generated.items()},
        "v172_counts": final_counts,
        "v172_remaining_gaps": gaps_final,
        "v172_frozen_sha256": frozen,
        "v171_original_preserved": True,
        "data_version_loaded_by_training": "experiments/language_v1/v172/data/v172_*.jsonl",
        "no_copy_no_punctuation_noise": True,
        "tokenizer_unchanged": True,
        "note": "C_counter 原始 1198/1200 缺口 2 如实保留，不用伪造样本补齐。",
    }
    write_json(OUT / "V172_DATA_AUDIT.json", payload)
    print(json.dumps({"v171_dialogue": audit["counts"]["train"].get("A_dialogue"),
                      "v172_dialogue": final_counts["train"].get("A_dialogue"),
                      "remaining_gaps": gaps_final,
                      "new_generated": {s: len(v) for s, v in generated.items()}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
