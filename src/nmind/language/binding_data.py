"""中文绑定域任务数据（R2 P1）：A/B/C 三级的确定性生成 + 独立 oracle + 封存/泄漏审计。

为什么需要这个模块（R2 任务书 §0/§3）：R1 的练习族（arithmetic / json_extract / grid_rules）
的字符集与 stable checkpoint 的 31 token 词表几乎不相交，模型读到的是整串 <unk>，
"练 81 次、神经验证 0 次"是分布错配的必然结果，不是训练次数不足。R2 因此把可学的
新技能限制在**同一个 stable 词表能读写的域**里，即 M5.2 的中文位置绑定域。

口径：
  * A 组（已知技能复现）的 prompt 与 M5.2 `run_m52.build_s0` **逐字节一致**（同模板、同 seed 序列），
    否则"A 复现"没有可比性；A 的封存集就是 M5.2 当年记录的 s0 sealed（seed 20261403，
    该 checkpoint 的 free-gen EM=1.0）。M5.2 模板里确认语与下一个 `用户：` 同一行
    （`…1号位置。\n好的。用户：小明在哪里？\n助手：`），这里照抄，不做"更漂亮"的改写。
  * B 组（相邻新技能）= 对 A 只改一个可识别变量（问法 / 实体值 / 位置值 / 两次陈述取最新），
    全部字符必须命中词表，oracle 独立于模型输出。
  * C 组（困难技能：两实体绑定 + 查询先陈述者）只做诊断，不作训练目标；
    除 EM 外必须报**反事实跟随率**（M5.2 已证明 EM 会掩盖"答最近陈述位置"的近因启发式）。
  * 词表外的族（需要 把/移动/他 等字符）如实登记为 UNSUPPORTED_VOCAB，不硬凑近义说法。
  * 每族按 split 隔离：train（学习流）/ dev / sealed（评测，永不进学习流）/ unseen_combo /
    counterfactual，seed 互不相同，并在 Gate-A 里报告与父模型训练文本的重叠率。
  * 注意：A 的模板空间只有 4 位置 × 4 确认语 = 16 条不同字符串，父模型见过全部 16 条，
    所以"A 封存集 EM=1.0"只能证明**复现**，不能证明泛化——泛化证据来自 B 的未见值探针。
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from typing import Any

ENTITIES = ("小明", "小红")
POSITIONS = ("1号位置", "2号位置", "3号位置", "4号位置")          # 父模型见过的取值
# 词表里只有数字 1/2/3/4：5-8 号位置无法在不改词表的前提下表达（见 UNSUPPORTED_VOCAB）
NEW_POSITIONS = ("5号位置", "6号位置", "7号位置", "8号位置")
# unseen_entity_qa 的学习流/测试流按位置对半切，保证"未见"在输入指纹层面成立（见其 builder）
ENTITY_TRAIN_POSITIONS = POSITIONS[:2]
ENTITY_HELDOUT_POSITIONS = POSITIONS[2:]
ACKS = ("好的。", "明白了。", "收到。")
ACK_PROB = 0.7                                                     # M5.2：30% 无确认语
R2_BASE_SEED = 20261601                                            # 与 M5.2(202614xx) 分开的号段
M52_SEED = 20261401                                                # 父模型训练所用 seed
M52_A_SEALED_SEED = 20261403                                       # M52 s0 sealed 的 seed

ORACLE_ID = "binding_oracle_v1"

FAMILIES: dict[str, dict[str, Any]] = {
    "position_qa": {"tier": "A", "label": "单实体位置问答（已知技能复现）",
                     "change_from_a": None, "neural_eval": True},
    "position_qa_paraphrase": {"tier": "B", "label": "同义问法（只改问句形式）",
                                "change_from_a": "问句 小明在哪里？ → 小明的位置在哪里？",
                                "neural_eval": True},
    "unseen_entity_qa": {"tier": "B", "label": "第二个实体做同样绑定（只改实体值）",
                          "change_from_a": "陈述与查询实体 小明 → 小红", "neural_eval": True},
    "unseen_position_qa": {"tier": "X", "label": "未见位置取值（词表外对照，不训练）",
                            "change_from_a": "位置 1-4号 → 5-8号", "neural_eval": False},
    "latest_statement_update": {"tier": "B", "label": "两次陈述取最新（事实更新，词表内写法）",
                                 "change_from_a": "上下文增加第二条同实体陈述，答案=最新",
                                 "neural_eval": True},
    "binding_swap": {"tier": "C", "label": "两实体绑定 + 查询先陈述者（近因陷阱）",
                      "change_from_a": "两实体上下文，查询第一个实体", "neural_eval": True},
    "combo_probe": {"tier": "C", "label": "跨族组合泛化探针（未见过的 陈述数×实体×问法×确认语 格子）",
                     "change_from_a": "多个变量同时变化，只诊断不训练", "neural_eval": False},
}

# 词表外：这些族要表达的字符不在 stable 词表里，强行训练只会把语言头拽向 <unk>
UNSUPPORTED_VOCAB = {
    "unseen_position_values": ("5", "6", "7", "8"),   # "小明在5号位置。"（数字只有 1-4）
    "update_command": ("把", "移", "动"),              # "把小明移动到5号位置。"
    "coreference_pronoun": ("他",),                    # "他在哪里？"
    "negation_query": ("不", "是"),                    # "小明不在1号位置，对吗？"
    "unmentioned_abstain": ("不", "知", "道"),         # 未陈述实体该拒答："不知道"
}

_ASCII_RE = re.compile(r"[A-Za-z0-9]")


@dataclass(frozen=True)
class BindingTask:
    family: str
    split: str                       # train | dev | sealed | unseen_combo | counterfactual
    prompt: str                      # 完整模型输入（以 "助手：" 结尾，与 runtime 桥接一致）
    target: str                      # oracle 答案（由 slots 计算，绝不含模型输出）
    context: str                     # 陈述部分（反事实配对键：同上下文不同问句）
    question: str
    slots: tuple[tuple[str, str], ...]
    fingerprint: str

    @property
    def tier(self) -> str:
        return FAMILIES[self.family]["tier"]

    def key(self) -> str:
        return f"{self.family}|{self.split}|{self.fingerprint}"

    def to_dict(self) -> dict[str, Any]:
        return {"family": self.family, "tier": self.tier, "split": self.split,
                "prompt": self.prompt, "target": self.target, "question": self.question,
                "context": self.context, "slots": dict(self.slots),
                "fingerprint": self.fingerprint, "oracle_id": ORACLE_ID}


def _fp(prompt: str, target: str) -> str:
    return hashlib.sha256(f"{prompt}→{target}".encode("utf-8")).hexdigest()[:16]


def _s0_prompt(statement: str, ack: str, question: str) -> str:
    """M5.2 s0 模板：`用户：{陈述}。\n{确认语}用户：{问句}\n助手：`（statement 自带句号）。"""
    return f"用户：{statement}\n{ack}用户：{question}\n助手："


def _s1_prompt(statements: str, question: str) -> str:
    """M5.2 s1 模板（多陈述同行）：用户：{陈述串}\n{问句}\n助手："""
    return f"用户：{statements}\n{question}\n助手："


def _mk(family: str, split: str, prompt: str, context: str, question: str,
        target: str, **slots: str) -> BindingTask:
    return BindingTask(family=family, split=split, prompt=prompt, target=target,
                       context=context, question=question,
                       slots=tuple(sorted(slots.items())), fingerprint=_fp(prompt, target))


def _ack(rng: random.Random, i: int) -> str:
    """M5.2 口径：确认语按 (i//4)%3 轮转，30% 概率为空。"""
    return ACKS[(i // 4) % 3] if rng.random() < ACK_PROB else ""


# ---------------------------------------------------------------- 各族确定性生成

def build_position_qa(n: int, seed: int, split: str = "sealed") -> list[BindingTask]:
    """A：与 M5.2 build_s0 逐字节一致（位置严格轮转 + 确认语轮转/30% 空）。"""
    rng = random.Random(seed)
    out: list[BindingTask] = []
    for i in range(n):
        p = POSITIONS[i % 4]
        ack = _ack(rng, i)
        q = "小明在哪里？"
        stmt = f"小明在{p}。"
        out.append(_mk("position_qa", split, _s0_prompt(stmt, ack, q), f"用户：{stmt}", q,
                        f"小明在{p}。", entity="小明", position=p, query_entity="小明", ack=ack))
    return out


def build_paraphrase(n: int, seed: int, split: str = "sealed") -> list[BindingTask]:
    """B：只改问句形式（"小明的位置在哪里？"——的/位/置 均在词表内）。"""
    rng = random.Random(seed)
    out: list[BindingTask] = []
    for i in range(n):
        p = POSITIONS[i % 4]
        ack = _ack(rng, i)
        q = "小明的位置在哪里？"
        stmt = f"小明在{p}。"
        out.append(_mk("position_qa_paraphrase", split, _s0_prompt(stmt, ack, q),
                        f"用户：{stmt}", q, f"小明在{p}。", entity="小明", position=p,
                        query_entity="小明", ack=ack))
    return out


def build_unseen_entity(n: int, seed: int, split: str = "sealed") -> list[BindingTask]:
    """B：只改实体值（小红；父模型 S0 阶段从没见过小红被陈述）。

    R2 预登记补充（写于 Gate-A 之后、Gate-B 之前）：本族模板空间只有
    4 位置 × 4 确认语态 = **16 个不同输入**。若学习流和测试集共用全部 4 个位置，
    "未见任务"就退化成"背过的位置"，§6 的泛化条件无法证明。因此把位置轴显式对半切：
      train            -> 1/2 号位置（学习流）
      sealed           -> 3/4 号位置（与学习流在输入指纹层面**零重叠**，才是真正的未见）
    实体轴（小红）对父模型仍是全新，"对父模型未见 + 对学习流未见"两个条件叠加，
    才构成可辩护的泛化证据。代价：训练与测试各自只覆盖一半位置，报告里必须写明。
    """
    pool = ENTITY_TRAIN_POSITIONS if split == "train" else ENTITY_HELDOUT_POSITIONS
    rng = random.Random(seed)
    out: list[BindingTask] = []
    for i in range(n):
        p = pool[i % len(pool)]
        ack = _ack(rng, i)
        q = "小红在哪里？"
        stmt = f"小红在{p}。"
        out.append(_mk("unseen_entity_qa", split, _s0_prompt(stmt, ack, q), f"用户：{stmt}", q,
                        f"小红在{p}。", entity="小红", position=p, query_entity="小红", ack=ack))
    return out


def build_unseen_position(n: int, seed: int, split: str = "sealed") -> list[BindingTask]:
    """X（词表外对照）：位置取值改成 5-8 号——stable 词表只有数字 1/2/3/4，
    因此这些题面编码后必然出现 <unk>。保留它是为了把 R1 的失败机制量化重现在
    同一父模型上（对照"分布错配"与"能力不足"），**不作为神经学习目标**。"""
    rng = random.Random(seed)
    out: list[BindingTask] = []
    for i in range(n):
        p = NEW_POSITIONS[i % 4]
        ack = _ack(rng, i)
        q = "小明在哪里？"
        stmt = f"小明在{p}。"
        out.append(_mk("unseen_position_qa", split, _s0_prompt(stmt, ack, q), f"用户：{stmt}", q,
                        f"小明在{p}。", entity="小明", position=p, query_entity="小明", ack=ack))
    return out


def build_latest_update(n: int, seed: int, split: str = "sealed") -> list[BindingTask]:
    """B：同一实体两次陈述，答案取最新一条（纯词表内写法，不用 把/移动）。"""
    rng = random.Random(seed)
    out: list[BindingTask] = []
    for i in range(n):
        p1 = POSITIONS[i % 4]
        p2 = POSITIONS[(i + 1) % 4]
        q = "小明在哪里？"
        stmts = f"小明在{p1}。小明在{p2}。"
        out.append(_mk("latest_statement_update", split, _s1_prompt(stmts, q),
                        f"用户：{stmts}", q, f"小明在{p2}。", entity="小明", position=p2,
                        old_position=p1, query_entity="小明"))
    return out


def build_binding_swap(n: int, seed: int, split: str = "sealed") -> list[BindingTask]:
    """C：两实体绑定；12 组合轮转，**每个组合先后被两个实体各查一次**（反事实配对）。

    M5.2 的 `build_s1` 用 `i % len(combos)` 配 `i % 2` 做查询实体交替，但 12 是偶数，
    于是同一组合永远配到同一实体——每个具体上下文只有一种问法，反事实配对集是空的。
    这里改成 `(i // 12) % 2`：组合走满一轮后翻转查询实体，同上下文两问都有。
    """
    combos = [(a, b) for a in POSITIONS for b in POSITIONS if a != b]
    out: list[BindingTask] = []
    for i in range(n):
        p1, p2 = combos[i % len(combos)]
        q_ent = ENTITIES[(i // len(combos)) % 2]
        p_of = {"小明": p1, "小红": p2}
        stmts = f"小明在{p1}。小红在{p2}。"
        q = f"{q_ent}在哪里？"
        out.append(_mk("binding_swap", split, _s1_prompt(stmts, q), f"用户：{stmts}", q,
                        f"{q_ent}在{p_of[q_ent]}。", entity_a="小明", position_a=p1,
                        entity_b="小红", position_b=p2, query_entity=q_ent))
    return out


def build_binding_swap_wide(n_pairs: int, seed: int, split: str = "counterfactual"
                             ) -> list[BindingTask]:
    """C 族宽域反事实封存集（Gate-B 005 方案 C）：上下文空间扩到 96 个唯一组合。

    为什么需要：build_binding_swap 的 context 由 (位置对) 决定，空间上限 12 ⇒
    counterfactual_pairs 最多 12 对，达不到 005 的"≥80 个封存配对"要求。宽域在
    **不改动 oracle 口径**的前提下扩上下文维度：12 位置组合 × 2 陈述顺序 × 4 确认语
    变体（好的/明白了/收到/无）= 96 个唯一上下文，每上下文先后按两个实体各查一次
    → 每上下文恰好 1 个反事实对，上限 96 对。

    slot 语义与 counterfactual_pairs 的 recency 计算对齐：entity_a/position_a =
    **先陈述**实体与其位置，entity_b/position_b = **后陈述**实体与其位置
    （recency_target = entity_b 在 position_b，由 counterfactual_pairs 统一计算）。
    oracle 与 build_binding_swap 同一公式（bd.oracle("binding_swap", **slots)）。
    """
    if n_pairs > 96:
        raise ValueError(f"宽域上下文空间上限 96 对，n_pairs={n_pairs} 超限")
    combos = [(a, b) for a in POSITIONS for b in POSITIONS if a != b]
    acks = ("好的。", "明白了。", "收到。", "")
    contexts = []
    for p1, p2 in combos:
        for order in ("ab", "ba"):
            for ack in acks:
                contexts.append((p1, p2, order, ack))
    rng = random.Random(seed)
    rng.shuffle(contexts)
    contexts = contexts[:n_pairs]
    out: list[BindingTask] = []
    for p_a, p_b, order, ack in contexts:
        first_ent, second_ent = ("小明", "小红") if order == "ab" else ("小红", "小明")
        stmts = f"{first_ent}在{p_a}。{second_ent}在{p_b}。"
        ctx = f"用户：{stmts}\n{ack}"
        for q_ent in ENTITIES:
            q = f"{q_ent}在哪里？"
            prompt = f"{ctx}用户：{q}\n助手："
            slots = {"entity_a": first_ent, "position_a": p_a,
                      "entity_b": second_ent, "position_b": p_b, "query_entity": q_ent}
            out.append(_mk("binding_swap", split, prompt, ctx, q,
                            oracle("binding_swap", **slots), **slots))
    return out


def build_combo_probe(n: int, seed: int, split: str = "unseen_combo", *,
                      exclude: list[BindingTask]) -> list[BindingTask]:
    """跨族组合泛化探针（等价于 build_unseen_combo("combo_probe", …)）：
    单陈述/双陈述 × 两实体 × 两问法 × 四确认语里没进过任何族学习流/评测集的格子。"""
    rng = random.Random(seed)
    cells = grid_cells("combo_probe")
    rng.shuffle(cells)
    seen = {t.prompt for t in exclude}
    out: list[BindingTask] = []
    for c in cells:
        t = render_grid_task("combo_probe", split, **c)
        if t.prompt in seen or t.fingerprint in {o.fingerprint for o in out}:
            continue
        out.append(t)
        if len(out) >= n:
            break
    return out


def slot_statements(**slots: str) -> tuple[str, ...]:
    """把槽位还原成陈述串（统一模板：`用户：{陈述…}\\n{确认语}用户：{问句}\\n助手：`）。"""
    if slots.get("entity_b"):
        return (f"{slots['entity_a']}在{slots['position_a']}。",
                f"{slots['entity_b']}在{slots['position_b']}。")
    if slots.get("old_position"):
        return (f"{slots['entity']}在{slots['old_position']}。",
                f"{slots['entity']}在{slots['position']}。")
    return (f"{slots['entity']}在{slots['position']}。",)


def render_grid_task(family: str, split: str, **slots: str) -> BindingTask:
    """统一渲染：同一套模板既能生成族内样本，也能生成组合泛化探针。"""
    stmts = slot_statements(**slots)
    ack = slots.get("ack", "")
    form = slots.get("form", "在哪里？")
    q = f"{slots.get('query_entity', slots.get('entity', slots.get('entity_a')))}{form}"
    prompt = "用户：" + "".join(stmts) + f"\n{ack}用户：{q}\n助手："
    target = oracle(family, **slots)
    return _mk(family, split, prompt, "用户：" + "".join(stmts), q, target, **slots)


# 组合泛化探针的槽位网格：实体 × 位置 × 问法 × 确认语（**全部限制在词表内**：
# 数字只有 1-4，5-8 号位置不进网格，改由 unseen_position_qa 族作为词表外对照单独测量）
GRID_FORMS = ("在哪里？", "的位置在哪里？")
GRID_ACKS = ("",) + ACKS
GRID_POSITIONS = POSITIONS


def grid_cells(family: str) -> list[dict[str, str]]:
    """按族裁剪的槽位网格（只保证字符在词表内，不代表已训练）。"""
    cells: list[dict[str, str]] = []
    ents = ENTITIES if family in ("unseen_entity_qa", "binding_swap") else ("小明",)
    poss = POSITIONS if family in ("position_qa", "unseen_position_qa") else GRID_POSITIONS
    forms = ("的位置在哪里？",) if family == "position_qa_paraphrase" else \
        ("在哪里？",) if family in ("position_qa", "unseen_position_qa") else GRID_FORMS
    if family == "binding_swap":
        for p1 in POSITIONS:
            for p2 in POSITIONS:
                if p1 == p2:
                    continue
                for qe in ENTITIES:
                    cells.append({"entity_a": "小明", "position_a": p1, "entity_b": "小红",
                                   "position_b": p2, "query_entity": qe})
        return cells
    if family == "latest_statement_update":
        for p1 in POSITIONS:
            for p2 in POSITIONS:
                if p1 == p2:
                    continue
                for a in GRID_ACKS:
                    cells.append({"entity": "小明", "query_entity": "小明",
                                   "old_position": p1, "position": p2, "ack": a})
        return cells
    if family == "combo_probe":
        for e in ENTITIES:
            for p in POSITIONS:
                for f in GRID_FORMS:
                    for a in GRID_ACKS:
                        cells.append({"entity": e, "position": p, "query_entity": e,
                                       "form": f, "ack": a})
                        for p2 in POSITIONS:
                            if p2 != p:
                                cells.append({"entity": e, "position": p2, "old_position": p,
                                               "query_entity": e, "form": f, "ack": a})
        return cells
    for e in ents:
        for p in poss:
            for f in forms:
                for a in GRID_ACKS:
                    cells.append({"entity": e, "position": p, "query_entity": e,
                                   "form": f, "ack": a})
    return cells


def build_unseen_combo(family: str, n: int, seed: int, *,
                       exclude: list[BindingTask]) -> list[BindingTask]:
    """组合泛化探针：该族槽位网格里**未出现在学习流/评测集**的组合，确定性取样。

    任务书 §3.2 第 7 项要求报 unseen entity / unseen combination / unseen length。
    模板空间只有 16 条的族（A）在这里只会给出"确认语×位置的未见组合"，
    取值本身仍是父模型见过的 1-4 号位置——报告里按事实区分，不冒充组合泛化。
    """
    rng = random.Random(seed)
    seen = {t.prompt for t in exclude}
    cells = grid_cells(family)
    rng.shuffle(cells)
    out: list[BindingTask] = []
    used: set[str] = set()
    for c in cells:
        t = render_grid_task(family, "unseen_combo", **c)
        if t.prompt in seen or t.fingerprint in used:
            continue
        used.add(t.fingerprint)
        out.append(t)
        if len(out) >= n:
            break
    return out


BUILDERS = {
    "position_qa": build_position_qa,
    "position_qa_paraphrase": build_paraphrase,
    "unseen_entity_qa": build_unseen_entity,
    "unseen_position_qa": build_unseen_position,
    "latest_statement_update": build_latest_update,
    "binding_swap": build_binding_swap,
}

# 每族 split → (规模, seed)。A 的 sealed 沿用 M5.2 seed，保证与其记录 EM=1.0 同源可比。
SPLIT_PLAN: dict[str, dict[str, tuple[int, int]]] = {
    "position_qa": {"sealed": (80, M52_A_SEALED_SEED), "dev": (24, R2_BASE_SEED + 1)},
    "position_qa_paraphrase": {"train": (48, R2_BASE_SEED + 10), "sealed": (40, R2_BASE_SEED + 11)},
    "unseen_entity_qa": {"train": (48, R2_BASE_SEED + 20),      # 位置 1/2（学习流）
                          "sealed": (40, R2_BASE_SEED + 21)},     # 位置 3/4（输入层面与学习流零重叠）
                          # 不再设 unseen_combo：train∪sealed 已覆盖 16 个组合，探针必然为空
    "unseen_position_qa": {"train": (48, R2_BASE_SEED + 30), "sealed": (40, R2_BASE_SEED + 31)},
    "latest_statement_update": {"train": (48, R2_BASE_SEED + 40), "sealed": (40, R2_BASE_SEED + 41),
                                 "unseen_combo": (24, R2_BASE_SEED + 42)},
    "binding_swap": {"sealed": (60, R2_BASE_SEED + 50),
                      "counterfactual": (60, R2_BASE_SEED + 51)},
    "combo_probe": {"unseen_combo": (32, R2_BASE_SEED + 60)},
}


def _family_exclude(family: str) -> list[BindingTask]:
    """探针要排除的"已覆盖格子"：本族 train+sealed；combo_probe 排除所有其他族的 train+sealed。"""
    if family != "combo_probe":
        return [t for s in ("train", "sealed") if s in SPLIT_PLAN[family]
                for t in build_family(family, s)]
    out: list[BindingTask] = []
    for other in SPLIT_PLAN:
        if other == "combo_probe":
            continue
        out += [t for s in ("train", "sealed", "unseen_combo", "counterfactual")
                 if s in SPLIT_PLAN[other] for t in build_family(other, s)]
    return out


def build_family(family: str, split: str, *, n: int | None = None,
                 seed: int | None = None) -> list[BindingTask]:
    """按 SPLIT_PLAN 取默认规模/seed（也可显式覆盖，用于 CLI 探针）。

    `unseen_combo` 是组合网格探针：只保留**没进过学习流/评测集**的格子，
    格子空间已被覆盖时如实返回更少的样本（报告里写 n 与原因，不硬凑）。
    """
    if family not in FAMILIES:
        raise KeyError(f"未知绑定族：{family}")
    plan = SPLIT_PLAN.get(family, {}).get(split)
    if plan is None and (n is None or seed is None):
        raise KeyError(f"族 {family} 未预登记 split={split}")
    size = n if n is not None else plan[0]
    sd = seed if seed is not None else plan[1]
    if split == "unseen_combo":
        return build_unseen_combo(family, size, sd, exclude=_family_exclude(family))
    if family not in BUILDERS:
        raise KeyError(f"族 {family} 没有 split={split} 的生成器")
    return BUILDERS[family](size, sd, split)


def build_all(*, only: tuple[str, ...] | None = None) -> dict[str, dict[str, list[BindingTask]]]:
    out: dict[str, dict[str, list[BindingTask]]] = {}
    for family in SPLIT_PLAN:
        if only and family not in only:
            continue
        out[family] = {split: build_family(family, split) for split in SPLIT_PLAN[family]}
    return out


# ---------------------------------------------------------------- 独立 oracle

def oracle(family: str, **slots: str) -> str:
    """由槽位确定性算出唯一答案，不读模型输出、不读学习流标签。"""
    if family == "binding_swap":
        p_of = {slots["entity_a"]: slots["position_a"], slots["entity_b"]: slots["position_b"]}
        return f"{slots['query_entity']}在{p_of[slots['query_entity']]}。"
    return f"{slots['query_entity']}在{slots['position']}。"


def oracle_for(task: BindingTask) -> str:
    """独立重算并与任务标签比对（Gate-A 用它验证"标签没被模型答案污染"）。"""
    return oracle(task.family, **dict(task.slots))


def normalize(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def check(task: BindingTask, predicted: str) -> bool:
    """精确匹配（去空白后），与 learning_loop.free_gen_em 的判据一致。"""
    return normalize(predicted) == normalize(task.target)


# ---------------------------------------------------------------- 词表 / 泄漏审计

def oov_tokens(tok, text: str) -> list[str]:
    """列出编码后落到 <unk> 的原始 token（与 CharTokenizer 的分词规则完全对齐）。"""
    out: list[str] = []
    buf = ""
    for ch in text:
        if _ASCII_RE.match(ch):
            buf += ch
            continue
        if buf:
            if buf.lower() not in tok.vocab:
                out.append(buf)
            buf = ""
        if ch != " " and ch not in tok.vocab:
            out.append(ch)
    if buf and buf.lower() not in tok.vocab:
        out.append(buf)
    return out


def unk_rate(tok, text: str) -> float:
    ids = tok.encode(text)
    return round(sum(1 for i in ids if i == 3) / max(1, len(ids)), 4)


def vocab_report(tok, tasks: list[BindingTask]) -> dict[str, Any]:
    oov: set[str] = set()
    for t in tasks:
        oov |= set(oov_tokens(tok, t.prompt)) | set(oov_tokens(tok, t.target))
    n = max(1, len(tasks))
    return {"n_tasks": len(tasks),
            "oov_tokens": sorted(oov),
            "tasks_with_any_oov": sum(1 for t in tasks
                                       if oov_tokens(tok, t.prompt) or oov_tokens(tok, t.target)),
            "prompt_unk_rate_mean": round(sum(unk_rate(tok, t.prompt) for t in tasks) / n, 4),
            "target_unk_rate_mean": round(sum(unk_rate(tok, t.target) for t in tasks) / n, 4),
            "in_vocab": not oov,
            "vocab_tokens": sorted(k for k in tok.vocab if not k.startswith("<")),
            "vocab_size": tok.vocab_size,
            "vocab_hash": tok.vocab_hash(),
            "unsupported_vocab_families": {
                fam: {"required_chars": sorted(set(cs)),
                      "oov_in_this_vocab": sorted(set(cs) - set(tok.vocab))}
                for fam, cs in UNSUPPORTED_VOCAB.items()}}


def parent_train_prompts() -> list[str]:
    """父模型（m52_s0_checkpoint）见过的文本：M5.2 s0 train(400)+dev(80)，S0 阶段只用这两批。"""
    return [t.prompt for t in build_position_qa(400, M52_SEED, "train")] + \
        [t.prompt for t in build_position_qa(80, M52_SEED + 1, "dev")]


def leakage(tok, tasks: list[BindingTask], parent_prompts: list[str]) -> dict[str, Any]:
    """提示级重叠：B/C 的 sealed/unseen 若已在父模型训练文本里出现，就不能当"未见样本"。"""
    par = set(parent_prompts)
    hits = [t.prompt for t in tasks if t.prompt in par]
    return {"n_tasks": len(tasks), "n_prompts_in_parent_train": len(hits),
            "overlap_rate": round(len(hits) / max(1, len(tasks)), 4),
            "distinct_prompts": len({t.prompt for t in tasks}),
            "distinct_targets": len({t.target for t in tasks}),
            "distinct_parent_prompts": len(par),
            "note": "A 组模板空间只有 4 位置×4 确认语=16 条，必然与父模型训练文本重叠；"
                    "因此 A 的封存集只证明复现，泛化证据来自未见取值/未见实体的 B 组探针"}


def counterfactual_pairs(tasks: list[BindingTask]) -> list[dict[str, Any]]:
    """同一上下文的两个查询实体配对：判"跟随查询实体"还是"跟随最近陈述"。"""
    by_ctx: dict[str, list[BindingTask]] = {}
    for t in tasks:
        by_ctx.setdefault(t.context, []).append(t)
    pairs: list[dict[str, Any]] = []
    for ctx, group in by_ctx.items():
        if len(group) < 2:
            continue
        # 同一 (上下文, 查询实体) 在轮转里会重复出现，反事实配对只保留一条
        uniq: dict[str, BindingTask] = {}
        for t in group:
            uniq.setdefault(dict(t.slots).get("query_entity", ""), t)
        group = list(uniq.values())
        if len(group) < 2:
            continue
        qs = sorted(group, key=lambda t: t.slots)
        s0 = dict(qs[0].slots)
        # 近因捷径的答案 = 上下文里**最后陈述**的那个实体
        recency = (f"{s0['entity_b']}在{s0['position_b']}。" if s0.get("entity_b")
                   else f"{s0['entity']}在{s0['position']}。")
        pairs.append({"context": ctx,
                       "recency_target": recency,
                       "tasks": [{"query_entity": dict(t.slots).get("query_entity", ""),
                                   "target": t.target, "prompt": t.prompt,
                                   "fingerprint": t.fingerprint} for t in qs]})
    return pairs


def follow_verdict(pair: dict[str, Any], preds: dict[str, str]) -> dict[str, Any]:
    """反事实跟随：两条回答是否都按"被问实体"绑定（而不是都答最近陈述的那个）。"""
    items = pair["tasks"]
    recency = pair["recency_target"]
    detail = []
    for it in items:
        p = preds.get(it["fingerprint"], "")
        detail.append({"query_entity": it["query_entity"], "target": it["target"],
                        "predicted": p, "correct": normalize(p) == normalize(it["target"]),
                        "recency_shortcut": normalize(p) == normalize(recency)
                        and normalize(it["target"]) != normalize(recency)})
    return {"context": pair["context"], "items": detail,
            "followed_query_entity": all(d["correct"] for d in detail)}


def answer_space(family: str) -> list[str]:
    """随机基线的答案空间（由槽位枚举，不读模型）。"""
    ents = ("小明", "小红") if family in ("unseen_entity_qa", "binding_swap") else ("小明",)
    poss = NEW_POSITIONS if family == "unseen_position_qa" else POSITIONS
    return [f"{e}在{p}。" for e in ents for p in poss]


def fingerprint_set(tasks: list[BindingTask]) -> set[str]:
    return {t.fingerprint for t in tasks}


def slot_coverage(tasks: list[BindingTask]) -> dict[str, Any]:
    """(实体, 位置) 组合覆盖报告（任务书 §3.2 第 7 项）。"""
    combos: set[tuple[str, str]] = set()
    for t in tasks:
        s = dict(t.slots)
        if t.family == "binding_swap":
            combos |= {(s["entity_a"], s["position_a"]), (s["entity_b"], s["position_b"])}
        else:
            combos.add((s["entity"], s["position"]))
    return {"distinct_combos": len(combos), "combos": sorted(f"{e}|{p}" for e, p in combos),
            "entities_seen": sorted({e for e, _p in combos}),
            "positions_seen": sorted({p for _e, p in combos})}
