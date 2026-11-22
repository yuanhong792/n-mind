"""中文受限微语言多轮对话生成器（M5-Language 任务书 P1）：独立确定性 oracle。

世界：实体（人/物）、位置（1-6号位置）、颜色；操作（放置/移动/涂色）；
问题（位置/颜色/数量/比较/否定/指代"它"/计算/查资料/不可答）。
答案全部由世界模拟器**独立计算**（oracle），模型训练/评测只看文本。

预登记（运行前已定，禁止看结果后修改）：
  * 模板分 TRAIN（训练/测试同式）与 PARA（ood_paraphrase 专用未见问法）两套；
  * 意图采样权重；实体/物件 train 与 ood_name 两个不相交集合；
  * 七个 split（train/dev/sealed_test/ood_name/ood_paraphrase/ood_composition/ood_length）
    用不同 seed 与实体组合隔离，pair/场景级不重叠；
  * manifest 记录全部 seed、模板清单、各 split sha256 与逐类型多数答案基线。
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

GENERATOR_VERSION = "dialogue-v1"

TRAIN_PERSONS = ["小明", "小红", "小刚", "小丽"]
OOD_PERSONS = ["大伟", "阿强", "小芳", "老王"]
TRAIN_OBJECTS = ["杯子", "书本", "钥匙", "皮球"]
OOD_OBJECTS = ["盒子", "帽子"]
POSITIONS = [f"{i}号位置" for i in range(1, 7)]
COLORS = ["红色", "蓝色", "绿色", "黄色"]
FACTS = {"图书馆闭馆日": "周四", "仓库盘点日": "每月最后一个周五"}
UNANSWERABLE = ["月球上有几只兔子？", "下周的彩票号码是什么？", "火星上有没有外卖？"]

ACK = "好的。"
ABSTAIN = "我不知道。"

# ---- 预登记模板：TRAIN（训练与常规测试同式） --------------------------------
T_STATE_POS = ["{e}在{p}。", "{e}正位于{p}。"]
T_STATE_COLOR = ["{e}是{c}的。", "{e}的颜色是{c}。"]
T_MOVE = ["把{e}移动到{p}。", "把{e}放到{p}。"]
T_PAINT = ["把{e}涂成{c}。"]
T_QA_POS = ["{e}在哪里？", "{e}在哪儿？"]
T_QA_POS_REF = ["它在哪里？", "它在哪儿？"]
T_QA_COLOR = ["{e}是什么颜色？", "{e}什么颜色？"]
T_QA_COLOR_REF = ["它是什么颜色？"]
T_QA_COUNT = ["{p}有几样东西？"]
T_QA_COMPARE = ["{e1}的位置编号比{e2}大吗？"]
T_QA_NEG = ["{e}不在{p}吗？"]
T_CALC = ["请计算：{a}加{b}等于多少？", "算一下{a}乘{b}等于多少。"]
T_LOOKUP = ["查资料：{k}？", "帮我查一下{k}。"]

# ---- 预登记模板：PARA（ood_paraphrase 专用，训练中绝不出现） -----------------
T_STATE_POS_PARA = ["{e}被放在了{p}。"]
T_QA_POS_PARA = ["{e}现在位于什么地方？", "{e}呆在哪里呢？"]
T_QA_POS_REF_PARA = ["它现在位于什么地方？"]
T_QA_COLOR_PARA = ["{e}的颜色是什么？", "说说{e}是什么颜色的。"]
T_MOVE_PARA = ["请把{e}挪到{p}。"]

INTENT_WEIGHTS = {            # 预登记的意图采样权重（和为 1.0）
    "stmt_pos": 0.20, "stmt_color": 0.10, "move": 0.08, "paint": 0.02,
    "qa_pos": 0.13, "qa_pos_ref": 0.09, "qa_color": 0.09, "qa_color_ref": 0.06,
    "qa_count": 0.05, "qa_compare": 0.05, "qa_neg": 0.05,
    "tool_calc": 0.04, "tool_lookup": 0.03, "unanswerable": 0.03,
}
QUESTION_INTENTS = {"qa_pos", "qa_pos_ref", "qa_color", "qa_color_ref", "qa_count",
                    "qa_compare", "qa_neg", "tool_calc", "tool_lookup", "unanswerable"}


@dataclass
class Turn:
    role: str                    # user | assistant
    text: str
    intent: str = ""


@dataclass
class Dialogue:
    dialogue_id: str
    split: str
    turns: list[Turn] = field(default_factory=list)


@dataclass
class _World:
    positions: dict[str, str] = field(default_factory=dict)
    colors: dict[str, str] = field(default_factory=dict)

    def copy(self) -> "_World":
        return _World(positions=dict(self.positions), colors=dict(self.colors))


def _pick(rng: random.Random, templates: list[str], **kw) -> str:
    return rng.choice(templates).format(**kw)


def _last_entity_in(text: str, entities: list[str]) -> str | None:
    hit = None
    for e in entities:
        if e in text:
            hit = e
    return hit


def _question_answer(world: _World, intent: str, rng: random.Random,
                     entities: list[str], last_entity: str | None) -> tuple[str, str, str | None]:
    """构造一个问题轮：返回 (user_text, oracle_answer, 提及实体)。"""
    known = [e for e in entities if e in world.positions or e in world.colors]
    positioned = [e for e in known if e in world.positions]
    if intent == "qa_pos":
        e = rng.choice(positioned)
        return _pick(rng, T_QA_POS, e=e), f"{e}在{world.positions[e]}。", e
    if intent == "qa_pos_ref":
        e = last_entity if last_entity in world.positions else rng.choice(positioned)
        return _pick(rng, T_QA_POS_REF), f"{e}在{world.positions[e]}。", e
    if intent == "qa_color":
        cand = [e for e in known if e in world.colors] or known
        e = rng.choice(cand)
        c = world.colors.get(e, "未知")
        return _pick(rng, T_QA_COLOR, e=e), f"{e}是{c}的。", e
    if intent == "qa_color_ref":
        e = last_entity or rng.choice(known)
        c = world.colors.get(e, "未知")
        return _pick(rng, T_QA_COLOR_REF), f"{e}是{c}的。", e
    if intent == "qa_count":
        p = rng.choice(POSITIONS)
        n = sum(1 for v in world.positions.values() if v == p)
        return _pick(rng, T_QA_COUNT, p=p), f"有{n}样。", None
    if intent == "qa_compare":
        positioned = [e for e in known if e in world.positions]
        if len(positioned) >= 2:
            e1, e2 = rng.sample(positioned, 2)
        else:
            e1 = rng.choice(known)
            others = [e for e in known if e != e1] or known
            e2 = rng.choice(others)
        n1 = int(world.positions.get(e1, "1号位置")[:1] or 1)
        n2 = int(world.positions.get(e2, "1号位置")[:1] or 1)
        ans = "是的。" if n1 > n2 else "不是。"
        return _pick(rng, T_QA_COMPARE, e1=e1, e2=e2), ans, e1
    if intent == "qa_neg":
        e = rng.choice([e for e in known if e in world.positions])
        if rng.random() < 0.5:
            p = world.positions[e]                      # 其实就在 → "不对，...在..."
            return _pick(rng, T_QA_NEG, e=e, p=p), f"不对，{e}在{p}。", e
        others = [p for p in POSITIONS if p != world.positions[e]]
        p = rng.choice(others)
        return _pick(rng, T_QA_NEG, e=e, p=p), f"是的，{e}不在{p}。", e
    if intent == "tool_calc":
        a, b = rng.randint(2, 12), rng.randint(2, 12)
        if rng.random() < 0.5:
            return _pick(rng, T_CALC, a=a, b=b), f"TOOL_CALCULATOR {a}+{b}", None
        return _pick(rng, T_CALC, a=a, b=b), f"TOOL_CALCULATOR {a}*{b}", None
    if intent == "tool_lookup":
        k = rng.choice(list(FACTS))
        return _pick(rng, T_LOOKUP, k=k), f"TOOL_LOOKUP {k}", None
    if intent == "unanswerable":
        return rng.choice(UNANSWERABLE), ABSTAIN, None
    raise ValueError(f"未知意图 {intent!r}")


def _statement(world: _World, intent: str, rng: random.Random, entities: list[str],
               para: bool) -> tuple[str, str | None]:
    """陈述/操作轮：返回 (user_text, 提及实体)。世界状态就地更新（oracle 环境）。"""
    e = rng.choice(entities)
    if intent == "stmt_pos":
        p = rng.choice(POSITIONS)
        world.positions[e] = p
        tpl = T_STATE_POS_PARA if para else T_STATE_POS
        return _pick(rng, tpl, e=e, p=p), e
    if intent == "stmt_color":
        c = rng.choice(COLORS)
        world.colors[e] = c
        return _pick(rng, T_STATE_COLOR, e=e, c=c), e
    if intent == "move":
        p = rng.choice(POSITIONS)
        if e not in world.positions:
            world.positions[e] = rng.choice(POSITIONS)
        world.positions[e] = p
        tpl = T_MOVE_PARA if para else T_MOVE
        return _pick(rng, tpl, e=e, p=p), e
    if intent == "paint":
        c = rng.choice(COLORS)
        world.colors[e] = c
        return _pick(rng, T_PAINT, e=e, c=c), e
    raise ValueError(f"未知陈述意图 {intent!r}")


def build_dialogue(split: str, dialogue_id: str, *, n_turns: int, persons: list[str],
                   objects: list[str], seed: int, para: bool = False) -> Dialogue:
    """构造一条多轮对话。用户轮意图按预登记权重采样；答案由 oracle 独立计算。"""
    rng = random.Random(seed)
    world = _World()
    entities = persons + objects
    d = Dialogue(dialogue_id=dialogue_id, split=split)
    last_entity: str | None = None
    questions = 0
    intents = list(INTENT_WEIGHTS)
    weights = [INTENT_WEIGHTS[i] for i in intents]
    for _ in range(n_turns):
        intent = rng.choices(intents, weights=weights, k=1)[0]
        if intent in ("qa_pos_ref", "qa_color_ref") and last_entity is None:
            intent = "qa_pos"
        if intent in ("qa_pos", "qa_pos_ref", "qa_compare", "qa_neg", "qa_count") \
                and not world.positions:
            intent = "stmt_pos"                     # 世界还没有位置事实，先铺事实
        if intent in ("qa_color", "qa_color_ref") and not world.colors:
            intent = "stmt_color"                   # 世界还没有颜色事实，先铺事实
        if intent in QUESTION_INTENTS:
            text, answer, mentioned = _question_answer(world, intent, rng, entities, last_entity)
            d.turns.append(Turn(role="user", text=text, intent=intent))
            d.turns.append(Turn(role="assistant", text=answer, intent=intent))
            questions += 1
            if mentioned:
                last_entity = mentioned
        else:
            text, mentioned = _statement(world, intent, rng, entities, para)
            d.turns.append(Turn(role="user", text=text, intent=intent))
            d.turns.append(Turn(role="assistant", text=ACK, intent=intent))
            if mentioned:
                last_entity = mentioned
    if questions == 0:                                # 保证每条对话至少一个问题
        if world.positions:
            intent = "qa_pos"
        elif world.colors:
            intent = "qa_color"
        else:
            intent = "unanswerable"
        text, answer, mentioned = _question_answer(world, intent, rng, entities, last_entity)
        d.turns.append(Turn(role="user", text=text, intent=intent))
        d.turns.append(Turn(role="assistant", text=answer, intent=intent))
    return d


@dataclass
class SplitSpec:
    name: str
    n_dialogues: int
    persons: list[str]
    objects: list[str]
    turns_range: tuple[int, int]
    seed: int
    para: bool = False


SPLITS: dict[str, SplitSpec] = {
    "train": SplitSpec("train", 300, TRAIN_PERSONS, TRAIN_OBJECTS, (2, 4), 20261231),
    "dev": SplitSpec("dev", 60, TRAIN_PERSONS, TRAIN_OBJECTS, (2, 4), 20261232),
    "sealed_test": SplitSpec("sealed_test", 80, TRAIN_PERSONS, TRAIN_OBJECTS, (2, 4), 20261233),
    "ood_name": SplitSpec("ood_name", 60, OOD_PERSONS, OOD_OBJECTS, (2, 4), 20261234),
    "ood_paraphrase": SplitSpec("ood_paraphrase", 60, TRAIN_PERSONS, TRAIN_OBJECTS,
                                 (2, 4), 20261235, para=True),
    "ood_composition": SplitSpec("ood_composition", 60, TRAIN_PERSONS, TRAIN_OBJECTS,
                                  (4, 6), 20261236),
    "ood_length": SplitSpec("ood_length", 40, TRAIN_PERSONS, TRAIN_OBJECTS, (6, 8), 20261237),
}


def build_split(spec: SplitSpec) -> list[Dialogue]:
    out: list[Dialogue] = []
    lo, hi = spec.turns_range
    for i in range(spec.n_dialogues):
        n = random.Random(spec.seed * 31 + i).randint(lo, hi)
        out.append(build_dialogue(spec.name, f"{spec.name}-{i:04d}",
                                   n_turns=n, persons=spec.persons, objects=spec.objects,
                                   seed=spec.seed * 1000 + i, para=spec.para))
    return out


def dialogue_to_dict(d: Dialogue) -> dict[str, Any]:
    return {"dialogue_id": d.dialogue_id, "split": d.split,
             "turns": [{"role": t.role, "text": t.text, "intent": t.intent}
                        for t in d.turns]}


def dialogue_text_for_vocab(d: Dialogue) -> str:
    return "\n".join(f"用户：{t.text}" if t.role == "user" else f"助手：{t.text}"
                      for t in d.turns)


def sha256_of(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True)
                           .encode("utf-8")).hexdigest()[:16]


def majority_answers(split_dialogues: list[Dialogue]) -> dict[str, str]:
    """逐意图的多数答案基线（任务书 P1：难度自检 + P4 门槛对照）。"""
    by_intent: dict[str, dict[str, int]] = {}
    for d in split_dialogues:
        for t in d.turns:
            if t.role == "assistant" and t.intent in QUESTION_INTENTS:
                by_intent.setdefault(t.intent, {})
                by_intent[t.intent][t.text] = by_intent[t.intent].get(t.text, 0) + 1
    return {intent: max(counter, key=counter.get)
            for intent, counter in by_intent.items()}


def build_all(data_dir: str | Path, *, include_sealed: bool = True) -> dict[str, Any]:
    """生成全部 split、落盘并返回 manifest（不可变数据工件）。"""
    data_dir = Path(data_dir)
    (data_dir / "splits").mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"generator_version": GENERATOR_VERSION,
                                 "intent_weights": INTENT_WEIGHTS,
                                 "entities": {"train_persons": TRAIN_PERSONS,
                                               "ood_persons": OOD_PERSONS,
                                               "train_objects": TRAIN_OBJECTS,
                                               "ood_objects": OOD_OBJECTS},
                                 "templates": {"train": {"qa_pos": T_QA_POS, "qa_pos_ref": T_QA_POS_REF,
                                                          "qa_color": T_QA_COLOR, "move": T_MOVE,
                                                          "state_pos": T_STATE_POS},
                                                "paraphrase_only": {"qa_pos": T_QA_POS_PARA,
                                                                     "qa_pos_ref": T_QA_POS_REF_PARA,
                                                                     "qa_color": T_QA_COLOR_PARA,
                                                                     "move": T_MOVE_PARA,
                                                                     "state_pos": T_STATE_POS_PARA}},
                                 "splits": {}}
    for name, spec in _split_order(include_sealed):
        dialogues = build_split(spec)
        payload = [dialogue_to_dict(d) for d in dialogues]
        (data_dir / "splits" / f"{name}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        manifest["splits"][name] = {
            "seed": spec.seed, "n_dialogues": len(dialogues),
            "turns_range": list(spec.turns_range),
            "para": spec.para, "sha256": sha256_of(payload),
            "majority_answers": majority_answers(dialogues),
            "n_question_turns": sum(1 for d in dialogues for t in d.turns
                                     if t.role == "assistant" and t.intent in QUESTION_INTENTS),
        }
    manifest["manifest_hash"] = sha256_of(manifest["splits"])
    (data_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    return manifest


def _split_order(include_sealed: bool):
    order = ["train", "dev"] + (["sealed_test"] if include_sealed else []) + \
        ["ood_name", "ood_paraphrase", "ood_composition", "ood_length"]
    return [(n, SPLITS[n]) for n in order]
