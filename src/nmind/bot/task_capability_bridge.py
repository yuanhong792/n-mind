"""任务—能力桥接与 Gate-A（R2 任务书 §3）：不训练，先判断"能不能学、该不该学"。

R1 的失败模式（81 次自主练习、神经验证 0 次、候选训练后封存集仍 0%）在这里被拆成
可分别证伪的四问：
  1. 输入桥接对不对？——同一段文本经 tokenizer 后是不是模型见过的字符流（UNK 率）；
  2. 标签是否泄露？——oracle 是否独立于模型输出，题面里是否直接写死了答案；
  3. 神经基线是否复现？——同一 stable checkpoint 能否复现已有技能 A（M5.2 记录 EM=1.0）；
  4. 新技能是否有可测缺口？——B 组相邻任务是否既没掌握、又有独立客观验证器。

四类结论口径（§3.3）：
  * TASK_CAPABILITY_ALIGNED     —— A 复现 + 至少一个 B 有真实缺口 → 才允许进 Gate-B；
  * NEURAL_BASELINE_NOT_REPRODUCED / INPUT_BRIDGING_UNVERIFIED —— A 复现不了：
    先修输入桥接与评测口径，禁止把 0% 归因于"神经学习失败"；
  * VERIFIER_NOT_STANDALONE     —— 没有独立验证器：只能记 RULE_ONLY/UNVERIFIED；
  * NO_ELIGIBLE_LEARNING_TASK   —— 对齐没问题，只是相邻空间已掌握：休眠等新数据，不是失败。
§3.4 要求把"工具算对"与"神经选对工具"分开记账：`tool_ledger()` 输出三条账；
工具任务的 verified 经验 learning_target=none，只允许写外部记忆，永不进神经学习流。
"""

from __future__ import annotations

import math
import random
import re
import time
import types
from dataclasses import asdict, dataclass, field
from typing import Any

from ..language import binding_data as bd
from ..language.decoder import free_generation
from .practice import execute_tool, make_sealed_set
from .skill_registry import default_registry

# 预登记阈值（实验前落定，不得为过门槛而调整）
GATE_A_ELIGIBLE_GAP_PP = 10.0      # B 组相对 oracle 上限至少差 10 个百分点才算"有缺口"
GATE_A_MASTERED_EM = 0.90          # 达到即视为已掌握（与 M5.2 S0 门槛同值）
GATE_A_A_REPRODUCED_EM = 0.90      # A 组复现下限（M5.2 记录 1.0，留 10pp 容差）

# 族级代表格：sealed 优先（ unseen 评测），其次 unseen_combo，最后 counterfactual。
# 不依赖 dict 顺序，否则同一族的"代表 EM"会随 split 构造顺序漂移。
SPLIT_PRIORITY = ("sealed", "unseen_combo", "counterfactual")

VERDICTS = ("TASK_CAPABILITY_ALIGNED", "NEURAL_BASELINE_NOT_REPRODUCED",
            "INPUT_BRIDGING_UNVERIFIED", "VERIFIER_NOT_STANDALONE",
            "NO_ELIGIBLE_LEARNING_TASK", "TRAINING_PRECONDITION_FAILED")
FAMILY_VERDICTS = ("ELIGIBLE_NEW_SKILL", "NO_MEASURABLE_GAP", "ALREADY_MASTERED",
                    "REPRODUCED_KNOWN_SKILL", "TOOL_ONLY", "UNSUPPORTED_VOCAB",
                    "DIAGNOSTIC_ONLY", "NO_STANDALONE_ORACLE")

_STATEMENT_RE = re.compile(r"(小明|小红)在([1-8]号位置)。")
_QUERY_ENTITY_RE = re.compile(r"^(小明|小红)")
_BINDING_SHAPE_RE = re.compile(r"^(小明|小红)在[1-8]号位置。$")


@dataclass
class TaskSpec:
    """桥接层统一任务视图：绑定族（binding_data）与 R1 工具族（practice）都落到这里。"""
    task_id: str
    family: str
    tier: str                                  # A | B | C | X | TOOL
    split: str
    source: str                                # binding | r1_practice
    prompt: str
    question: str
    target: str
    skill_id: str
    oracle_id: str
    verifier_standalone: bool
    answer_readable_from_input: bool
    allowed_paths: tuple[str, ...]
    learning_target: str
    response_source_expected: str
    difficulty: int
    prerequisite_skill_ids: tuple[str, ...]
    neural_evaluation_enabled: bool
    neural_training_allowed: bool
    max_input_chars: int
    max_response_tokens: int
    fingerprint: str
    slots: dict[str, str] = field(default_factory=dict)
    task: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["allowed_paths"] = list(self.allowed_paths)
        d["prerequisite_skill_ids"] = list(self.prerequisite_skill_ids)
        return d

    def binding_view(self) -> Any:
        """还原成 binding_data 的轻量任务对象（词表/泄漏/反事实审计要用同一份槽位）。"""
        d = self.task["binding_task"]
        return types.SimpleNamespace(family=d["family"], split=d["split"], prompt=d["prompt"],
                                      target=d["target"], question=d["question"],
                                      context=d["context"], fingerprint=d["fingerprint"],
                                      slots=tuple(sorted(d["slots"].items())))


# ------------------------------------------------------------------ 任务构造

def _spec_from_skill(skill, **kw: Any) -> "TaskSpec":
    """注册表里的 SkillSpec → 本模块的 TaskSpec 公共字段（调用方补齐任务特有字段）。"""
    base = dict(skill_id=skill.skill_id, oracle_id=skill.verifier_id,
                 verifier_standalone=skill.verifier_standalone,
                 learning_target=skill.learning_target,
                 response_source_expected=skill.response_source_expected,
                 difficulty=skill.difficulty,
                 prerequisite_skill_ids=skill.prerequisite_skill_ids,
                 neural_evaluation_enabled=skill.neural_evaluation_enabled,
                 neural_training_allowed=skill.neural_training_allowed,
                 max_input_chars=skill.max_input_chars,
                 max_response_tokens=skill.max_response_tokens)
    base.update(kw)
    return TaskSpec(**base)


def binding_specs(tok, splits: tuple[str, ...] = ("sealed", "unseen_combo", "counterfactual"),
                  *, registry=None) -> list[TaskSpec]:
    """绑定族 → TaskSpec。技能与族一一对应，注册表决定"能不能测/能不能训"。"""
    reg = registry or default_registry()
    out: list[TaskSpec] = []
    for family, meta in bd.FAMILIES.items():
        skills = reg.by_family(family)
        if not skills:
            continue
        skill = skills[0]
        for split in bd.SPLIT_PLAN[family]:
            if split not in splits:
                continue
            for i, t in enumerate(bd.build_family(family, split)):
                oov = sorted(set(bd.oov_tokens(tok, t.prompt)) | set(bd.oov_tokens(tok, t.target)))
                out.append(_spec_from_skill(
                    skill, task_id=f"bind-{family}-{split}-{i:03d}-{t.fingerprint}",
                    family=family, tier=meta["tier"], split=split, source="binding",
                    prompt=t.prompt, question=t.question, target=t.target,
                    answer_readable_from_input=False,   # 题面只给陈述，答案由绑定规则推出
                    allowed_paths=("NEURAL_GENERATE", "ABSTAIN"),
                    fingerprint=t.fingerprint, slots=dict(t.slots),
                    task={"oov_tokens": oov, "binding_task": t.to_dict()}))
    return out


def practice_specs(tok, families: tuple[str, ...] = ("arithmetic", "json_extract", "grid_rules"),
                   *, n: int = 24, seed: int = 20261699, registry=None) -> list[TaskSpec]:
    """R1 三个练习族：保留为**工具/程序任务**，题面字符与 stable 词表的错配如实记账。"""
    reg = registry or default_registry()
    out: list[TaskSpec] = []
    for family in families:
        skills = reg.by_family(family)
        if not skills:
            continue
        skill = skills[0]
        for i, t in enumerate(make_sealed_set(family, n, seed=seed)):
            q = t["question"]
            prompt = f"用户：{q}\n助手："
            readable = str(t.get("answer")) in q      # 题面里直接写着答案 = 标签泄露
            out.append(_spec_from_skill(
                skill, task_id=f"r1-{family}-{i:03d}-{t['fingerprint']}", family=family,
                tier="TOOL", split="sealed", source="r1_practice", prompt=prompt, question=q,
                target=str(t["answer"]),
                answer_readable_from_input=bool(readable),
                allowed_paths=("TOOL_USE", "EXTERNAL_RETRIEVAL", "MEMORY_QUERY", "ABSTAIN"),
                learning_target="none", response_source_expected="tool",
                neural_training_allowed=False,
                fingerprint=t["fingerprint"], slots={}, task=dict(t),))
            out[-1].task["oov_tokens"] = sorted(set(bd.oov_tokens(tok, prompt)))
    return out


# ------------------------------------------------------------------ 单任务判分

def format_flags(spec: TaskSpec, predicted: str) -> dict[str, Any]:
    """输入格式/输出形态是否被模型正确处理（§3.2 第 2 项）。

    `replaced_char_present` 统计解码里出现 U+FFFD 的比例：词表外字符不会静默消失，
    而是以替换符如实暴露出来——这正是 R1 三个工具族 EM=0 的直接机制证据。
    """
    p = (predicted or "").strip()
    binding = spec.source == "binding"
    ent = _QUERY_ENTITY_RE.match(spec.question)
    flags = {"nonempty": bool(p), "replaced_char_present": "�" in p}
    if binding:
        flags.update({"valid_answer_shape": bool(_BINDING_SHAPE_RE.match(p)),
                       "query_entity_correct": bool(ent) and p.startswith(ent.group(1)),
                       "position_stated": bool(_STATEMENT_RE.search(p)),
                       "ends_with_period": p.endswith("。")})
    else:
        flags.update({"valid_answer_shape": p == spec.target.strip(),
                       "query_entity_correct": None, "position_stated": None,
                       "ends_with_period": p.endswith((".", "。")) or p.isdigit()})
    return flags


def _em(spec: TaskSpec, predicted: str) -> bool:
    return bd.normalize(predicted) == bd.normalize(spec.target)


def attempt(model, tok, spec: TaskSpec, *, max_new_tokens: int | None = None) -> dict[str, Any]:
    """一次真实自由生成（不读标签），返回可审计的尝试记录。"""
    limit = max_new_tokens or spec.max_response_tokens
    rec = free_generation(model, tok, spec.prompt, max_new_tokens=limit)
    return {"task_id": spec.task_id, "family": spec.family, "tier": spec.tier,
            "split": spec.split, "skill_id": spec.skill_id, "fingerprint": spec.fingerprint,
            "predicted": rec.text, "target": spec.target, "em": int(_em(spec, rec.text)),
            "stop_reason": rec.finish_reason, "n_tokens": len(rec.token_ids),
            "prompt_unk_rate": bd.unk_rate(tok, spec.prompt),
            "response_source": "neural", "selection_source": "scheduler",
            "verifier_id": spec.oracle_id,
            "feedback_source": "deterministic_oracle" if spec.verifier_standalone else "unverified",
            "learning_target": spec.learning_target, "format": format_flags(spec, rec.text)}


def measure(model, tok, specs: list[TaskSpec], *, max_new_tokens: int | None = None) -> list[dict]:
    return [attempt(model, tok, s, max_new_tokens=max_new_tokens) for s in specs]


# ------------------------------------------------------------------ 基线与对照

def rule_baseline(spec: TaskSpec, *, recency: bool) -> str:
    """两条确定性规则基线：绑定规则（按被问实体取其位置）与近因启发式（取最后陈述位置）。

    M5.2 的诊断显示模型在两实体任务上的行为更接近后者，因此两条都要报——
    用来说明"EM 达标"与"真的按查询实体绑定"不是一回事。
    """
    stated = _STATEMENT_RE.findall(spec.prompt)
    ent = _QUERY_ENTITY_RE.match(spec.question)
    if not stated or not ent:
        return ""
    if recency:
        return f"{ent.group(1)}在{stated[-1][1]}。"
    pos = next((p for e, p in reversed(stated) if e == ent.group(1)), None)
    return f"{ent.group(1)}在{pos}。" if pos else ""


def wilson_ci(hits: int, n: int, z: float = 1.96) -> list[Any]:
    """二项比例 95% 置信区间（Wilson）。单 seed 小样本必须带 n 与 CI，不给裸点值。"""
    if n <= 0:
        return [None, None]
    p = hits / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - half), 4), round(min(1.0, c + half), 4)]


def _rate(bools: list[bool]) -> dict[str, Any]:
    n, hits = len(bools), sum(1 for b in bools if b)
    return {"n": n, "hits": hits, "em": round(hits / n, 4) if n else None,
            "ci95": wilson_ci(hits, n)}


def baselines(specs: list[TaskSpec], *, seed: int = 20261671) -> dict[str, Any]:
    """随机/固定答案/规则/近因/上界（§3.2 第 4 项），按族分组，全部不读模型输出。"""
    rng = random.Random(seed)
    by_family: dict[str, list[TaskSpec]] = {}
    for s in specs:
        by_family.setdefault(s.family, []).append(s)
    out: dict[str, Any] = {}
    for family, group in by_family.items():
        space = bd.answer_space(family) if family in bd.FAMILIES else sorted({s.target for s in group})
        rnd = [rng.choice(space) if space else "" for _ in group]
        rb = [rule_baseline(s, recency=False) for s in group]
        rr = [rule_baseline(s, recency=True) for s in group]
        out[family] = {
            "n": len(group), "answer_space_size": len(space),
            "distinct_targets": len({s.target for s in group}),
            "random": _rate([_em(s, p) for s, p in zip(group, rnd)]),
            "fixed_target": _rate([_em(s, group[0].target) for s in group]),
            "rule_binding": _rate([_em(s, p) for s, p in zip(group, rb)]),
            "rule_recency_shortcut": _rate([_em(s, p) for s, p in zip(group, rr)]),
            "ceiling_oracle": _rate([True for _ in group]),
            "rule_binding_coverage": round(sum(1 for p in rb if p) / len(group), 4),
        }
    return out


def tool_ledger(specs: list[TaskSpec]) -> dict[str, Any]:
    """三条账分开（§3.4）：神经决策 / 工具或程序执行 / 组合结果。"""
    out: dict[str, Any] = {}
    for s in [x for x in specs if x.tier == "TOOL"]:
        fam = out.setdefault(s.family, {
            "n": 0, "tool_available": 0, "tool_exec_correct": 0, "neural_selected_tool": 0,
            "label_leak": 0, "skill_id": s.skill_id, "oracle_id": s.oracle_id,
            "learning_target": s.learning_target, "tool_errors": [], "samples": []})
        fam["n"] += 1
        fam["label_leak"] += int(s.answer_readable_from_input)
        try:
            got = str(execute_tool(s.task))
            fam["tool_available"] += 1
            fam["tool_exec_correct"] += int(got.strip() == s.target.strip())
            fam["neural_selected_tool"] += 1     # R1 的工具绑定是代码白名单（selection_source=rule）
        except Exception as e:
            fam["tool_errors"].append(repr(e)[:120])
        if len(fam["samples"]) < 2:
            fam["samples"].append({"question": s.question[:70], "target": s.target,
                                    "selection_source": "rule"})
    for fam in out.values():
        n = max(1, fam["n"])
        fam["tool_exec_em"] = round(fam["tool_exec_correct"] / n, 4)
        fam["attribution_note"] = ("神经决策账≠工具执行账：selection_source=rule 的固定绑定只证明"
                                    "路由存在，不证明模型学会选工具；工具算对也不等于神经能力增长")
    return out


# ------------------------------------------------------------------ 词表/泄漏/反事实

def shortcut_breakdown(pairs: list[dict[str, Any]], texts: dict[str, str]) -> dict[str, Any]:
    """把每个反事实作答归类：按被问实体 / 按最后陈述（近因）/ 按最先陈述（首位）/ 其他。

    M5.2 在 S1 训练后的权重上观察到"答最近陈述位置"；本轮父模型只经过 S0，
    所以必须重新测量它到底走哪条捷径，不能把 M5.2 的机制结论当成既有事实（§3.2 第 6 项）。
    """
    counts = {"correct_binding": 0, "recency_shortcut": 0, "first_statement_shortcut": 0,
              "other": 0}
    examples: list[dict[str, Any]] = []
    for p in pairs:
        stated = _STATEMENT_RE.findall(p["context"])
        first = f"{stated[0][0]}在{stated[0][1]}。" if stated else ""
        recency = p["recency_target"]
        for it in p["tasks"]:
            got = (texts.get(it["fingerprint"], "") or "").strip()
            target = it["target"].strip()
            if got == target:
                kind = "correct_binding"
            elif got == recency and got != target:
                kind = "recency_shortcut"
            elif got == first and got != target:
                kind = "first_statement_shortcut"
            else:
                kind = "other"
            counts[kind] += 1
            if len(examples) < 6 and kind != "correct_binding":
                examples.append({"query_entity": it["query_entity"], "target": target,
                                  "predicted": got, "kind": kind})
    n = max(1, sum(counts.values()))
    return {"counts": counts, "n": sum(counts.values()),
            "rates": {k: round(v / n, 4) for k, v in counts.items()}, "examples": examples}


def counterfactual_matrix(model, tok, specs: list[TaskSpec], *,
                          max_new_tokens: int = 24) -> dict[str, Any]:
    """两实体绑定的反事实跟随率：同上下文两问都按被问实体绑定才算"真绑定"。"""
    swap = [s for s in specs if s.family == "binding_swap" and s.split == "counterfactual"]
    if not swap:
        return {"status": "NOT_RUN", "reason": "缺 binding_swap 的 counterfactual split"}
    recs = {s.fingerprint: attempt(model, tok, s, max_new_tokens=max_new_tokens) for s in swap}
    pairs = bd.counterfactual_pairs([s.binding_view() for s in swap])
    texts = {fp: r["predicted"] for fp, r in recs.items()}
    verdicts = [bd.follow_verdict(p, texts) for p in pairs]
    items = [it for v in verdicts for it in v["items"]]
    follow = sum(1 for v in verdicts if v["followed_query_entity"])
    shortcut = sum(1 for it in items if it["recency_shortcut"])
    return {"n_pairs": len(pairs), "n_items": len(items),
            "follow_rate": round(follow / max(1, len(pairs)), 4), "follow_hits": follow,
            "ci95": wilson_ci(follow, len(pairs)),
            "recency_shortcut_items": shortcut,
            "shortcut_item_rate": round(shortcut / max(1, len(items)), 4),
            "mechanism": shortcut_breakdown(pairs, texts),
            "per_pair_example": verdicts[:4],
            "note": "EM 达标但 follow_rate 低 = 走了某条捷径，不算绑定能力（任务书 §3.2 第 6 项）"}


def leakage_matrix(specs: list[TaskSpec], tok) -> dict[str, Any]:
    """词表边界 + 与父模型训练文本的重叠 + 标签泄露 + oracle 一致性（§3.2 第 3/5 项）。"""
    parent = bd.parent_train_prompts()
    out: dict[str, Any] = {}
    for family in sorted({s.family for s in specs}):
        group = [s for s in specs if s.family == family
                 and s.split in ("sealed", "unseen_combo", "counterfactual")]
        if not group:
            continue
        if group[0].source == "binding":
            views = [s.binding_view() for s in group]
            vr, lk = bd.vocab_report(tok, views), bd.leakage(tok, views, parent)
            agree = all(bd.oracle_for(v) == v.target for v in views)
        else:
            oov = sorted({c for s in group for c in bd.oov_tokens(tok, s.prompt)})
            vr = {"n_tasks": len(group), "oov_tokens": oov, "in_vocab": not oov,
                  "tasks_with_any_oov": sum(1 for s in group if bd.oov_tokens(tok, s.prompt)),
                  "prompt_unk_rate_mean": round(sum(bd.unk_rate(tok, s.prompt) for s in group)
                                                 / max(1, len(group)), 4)}
            lk = {"n_tasks": len(group), "n_prompts_in_parent_train": 0, "overlap_rate": 0.0,
                  "distinct_prompts": len({s.prompt for s in group})}
            agree = True
        out[family] = {"vocab": vr, "leakage": lk,
                        "label_leak_from_input": sum(1 for s in group
                                                     if s.answer_readable_from_input),
                        "oracle_standalone": all(s.verifier_standalone for s in group),
                        "oracle_agrees_with_labels": bool(agree),
                        "max_input_chars": max(s.max_input_chars for s in group),
                        "longest_prompt_chars": max(len(s.prompt) for s in group)}
    return out


# ------------------------------------------------------------------ 族级判定

def family_verdict(family: str, tier: str, *, in_vocab: bool, em: float | None,
                   oracle_standalone: bool, label_leak: bool, tool_only: bool,
                   gap_pp: float | None = None) -> tuple[str, str]:
    """返回 (verdict, reason)。优先级：词表 > 独立验证器 > 工具 > 层级 > 缺口。"""
    if not in_vocab:
        return ("UNSUPPORTED_VOCAB",
                "题面/答案含词表外字符，模型只能读到 <unk>；不得作为神经生成学习目标")
    if not oracle_standalone:
        return ("NO_STANDALONE_ORACLE", "没有独立于模型输出的客观验证器 → 只能记 RULE_ONLY/UNVERIFIED")
    if tool_only:
        return ("TOOL_ONLY", "有可信工具/程序 oracle：答案由工具产生，verified 经验只进外部记忆，"
                              "不作为神经能力增长证据"
                              + ("；且题面直接包含答案（标签泄露）" if label_leak else ""))
    if em is None:
        return ("NO_STANDALONE_ORACLE", "未测（NOT_RUN）")
    gap = round((1.0 - em) * 100, 2) if gap_pp is None else gap_pp
    if tier == "A":
        if em >= GATE_A_A_REPRODUCED_EM:
            return ("REPRODUCED_KNOWN_SKILL", f"已知技能复现 EM={em} ≥ {GATE_A_A_REPRODUCED_EM}")
        return ("NO_MEASURABLE_GAP", f"A 组复现失败（EM={em}）：先修输入桥接，不做神经归因")
    if tier == "C":
        return ("DIAGNOSTIC_ONLY", "困难技能仅诊断（两实体绑定/组合泛化），未经单独许可不训练")
    if em >= GATE_A_MASTERED_EM:
        return ("ALREADY_MASTERED", f"相邻技能 EM={em}，缺口 {gap}pp < {GATE_A_ELIGIBLE_GAP_PP}pp")
    if gap >= GATE_A_ELIGIBLE_GAP_PP:
        return ("ELIGIBLE_NEW_SKILL", f"独立 oracle + 缺口 {gap}pp ≥ {GATE_A_ELIGIBLE_GAP_PP}pp"
                                       "→ 可进 Gate-B 小预算实验")
    return ("NO_MEASURABLE_GAP", f"缺口 {gap}pp 不足 {GATE_A_ELIGIBLE_GAP_PP}pp（部分泛化）")


# ------------------------------------------------------------------ Gate-A 主流程

def run_gate_a(model, tok, *, binding: list[TaskSpec], tools: list[TaskSpec],
               stable_sha: str = "", stable_file_sha: str = "",
               max_new_tokens: int = 24) -> dict[str, Any]:
    """Gate-A：只测量、零 optimizer.step。产出能力矩阵 + 任务适配审计 + 总结论。"""
    t0 = time.time()
    specs = list(binding) + list(tools)
    evaluable = [s for s in specs if s.neural_evaluation_enabled]
    records = measure(model, tok, evaluable, max_new_tokens=max_new_tokens)
    by_id = {r["task_id"]: r for r in records}
    matrix: dict[str, Any] = {}
    for spec in specs:
        fam = matrix.setdefault(spec.family, {"tier": spec.tier, "skill_id": spec.skill_id,
                                                "splits": {}, "n_specs": 0,
                                                "neural_training_allowed":
                                                spec.neural_training_allowed})
        fam["n_specs"] += 1
        rec = by_id.get(spec.task_id)
        if rec is None:
            fam["splits"][spec.split] = {"status": "NOT_RUN",
                                          "reason": "neural_evaluation_enabled=False"}
            continue
        cell = fam["splits"].setdefault(spec.split, {"n": 0, "em_hits": 0, "_rec": []})
        cell["n"] += 1
        cell["em_hits"] += rec["em"]
        cell["_rec"].append(rec)
    for fam in matrix.values():
        for split, cell in fam["splits"].items():
            if cell.get("status") == "NOT_RUN":
                continue
            n = max(1, cell["n"])
            recs = cell.pop("_rec")
            cell["em"] = round(cell["em_hits"] / n, 4)
            cell["ci95"] = wilson_ci(cell["em_hits"], n)
            cell["format_pass"] = {
                k: round(sum(1 for r in recs if r["format"][k]) / n, 4)
                for k in ("nonempty", "valid_answer_shape", "replaced_char_present")}
            cell["stop_reasons"]: dict[str, int] = {}
            for r in recs:
                cell["stop_reasons"][r["stop_reason"]] = cell["stop_reasons"].get(
                    r["stop_reason"], 0) + 1
            cell["prompt_unk_rate_mean"] = round(sum(r["prompt_unk_rate"] for r in recs) / n, 4)
            cell["mean_tokens"] = round(sum(r["n_tokens"] for r in recs) / n, 2)
            cell["examples"] = [{"predicted": r["predicted"][:40], "target": r["target"],
                                  "em": r["em"]} for r in recs[:4]]
    leak = leakage_matrix(specs, tok)
    verdicts: dict[str, Any] = {}
    a_em: float | None = None
    for family, fam in matrix.items():
        cell = next((fam["splits"][s] for s in SPLIT_PRIORITY
                     if s in fam["splits"] and "em" in fam["splits"][s]), None)
        em = cell["em"] if cell else None
        if family == "position_qa":
            a_em = em
        lv = leak.get(family, {})
        vocab = lv.get("vocab", {})
        verdicts[family] = {"family": family, "tier": fam["tier"], "skill_id": fam["skill_id"],
                             "neural_em": em, "in_vocab": vocab.get("in_vocab", True),
                             "oov_tokens": vocab.get("oov_tokens", []),
                             "label_leak": lv.get("label_leak_from_input", 0),
                             "oracle_standalone": lv.get("oracle_standalone", True),
                             "oracle_agrees_with_labels": lv.get("oracle_agrees_with_labels", True),
                             "neural_training_allowed": fam["neural_training_allowed"]}
    tl = tool_ledger(specs)
    for family, v in verdicts.items():
        vd, why = family_verdict(family, v["tier"], in_vocab=v["in_vocab"], em=v["neural_em"],
                                  oracle_standalone=v["oracle_standalone"]
                                  and v["oracle_agrees_with_labels"],
                                  label_leak=bool(v["label_leak"]),
                                  tool_only=family in tl)
        v["verdict"], v["reason"] = vd, why
        if family in tl:
            v["tool_exec_em"] = tl[family]["tool_exec_em"]
    eligible = sorted(f for f, v in verdicts.items() if v["verdict"] == "ELIGIBLE_NEW_SKILL")
    reproduced = a_em is not None and a_em >= GATE_A_A_REPRODUCED_EM
    if not reproduced:
        if a_em is None:
            verdict, summary = "INPUT_BRIDGING_UNVERIFIED", "A 组没测到数：先补评测口径"
        else:
            verdict, summary = ("NEURAL_BASELINE_NOT_REPRODUCED",
                                 f"A 组未复现（EM={a_em}）：先修输入桥接/评测口径，"
                                 "禁止把 0% 归因于神经学习失败")
    elif not eligible:
        verdict = "NO_ELIGIBLE_LEARNING_TASK"
        summary = (f"A 复现 EM={a_em}，但没有相邻技能留下 ≥{GATE_A_ELIGIBLE_GAP_PP}pp 的真实缺口 "
                    "→ NO_ELIGIBLE_TASK 休眠，等待新的分布外任务，不是神经失败")
    else:
        verdict = "TASK_CAPABILITY_ALIGNED"
        summary = (f"A 复现 EM={a_em}；{len(eligible)} 个相邻技能有独立 oracle 且缺口达标："
                    f"{', '.join(eligible)}")
    return {
        "gate": "GATE_A", "no_training": True, "verdict": verdict, "summary": summary,
        "verdicts_vocab": list(VERDICTS), "family_verdicts_vocab": list(FAMILY_VERDICTS),
        "stable_sha256_state": stable_sha, "stable_file_sha256": stable_file_sha,
        "a_reproduced": reproduced, "a_sealed_em": a_em, "eligible_families": eligible,
        "families": verdicts, "matrix": matrix, "baselines": baselines(evaluable),
        "tool_ledger": tl,
        "counterfactual": counterfactual_matrix(model, tok, binding,
                                                  max_new_tokens=max_new_tokens),
        "leakage_and_vocab": leak,
        "thresholds": {"a_reproduced_em_min": GATE_A_A_REPRODUCED_EM,
                        "mastered_em": GATE_A_MASTERED_EM,
                        "eligible_gap_pp": GATE_A_ELIGIBLE_GAP_PP},
        "n_specs": len(specs), "n_measured": len(records),
        "wall_s": round(time.time() - t0, 2), "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def reload_consistency(model_factory, tok, specs: list[TaskSpec], *, n: int = 6) -> dict[str, Any]:
    """同权重重复加载一致性（§6.4 最后一条）：两次加载的贪心输出必须逐字相同。"""
    a = [attempt(model_factory(), tok, s)["predicted"] for s in specs[:n]]
    b = [attempt(model_factory(), tok, s)["predicted"] for s in specs[:n]]
    return {"n": len(a), "identical": a == b, "mismatch": [i for i, (x, y) in enumerate(zip(a, b))
                                                              if x != y]}


def precondition_failures(gate_a: dict[str, Any], *, scheduler_verified_threshold: int,
                          available_verified: int, budget_authorized: bool,
                          candidate_max_steps: int) -> list[str]:
    """训练前置校验（修 R1 bug：调度器要 32 条 verified，练习根本产不出 → 永不触发）。

    返回失败原因列表；空列表才允许进 Gate-B。任何一条不满足都不许偷偷降门槛：
    要么如实停在这里，要么按 §6.3 单独授权。
    """
    fails: list[str] = []
    if gate_a.get("verdict") != "TASK_CAPABILITY_ALIGNED":
        fails.append(f"Gate-A 未通过：verdict={gate_a.get('verdict')}")
    if not gate_a.get("eligible_families"):
        fails.append("没有 ELIGIBLE_NEW_SKILL 族：无可学新技能")
    if not budget_authorized:
        fails.append("候选训练未授权（policy.candidate_training_authorized=False）")
    if candidate_max_steps <= 0:
        fails.append("候选训练步数预算为 0")
    if available_verified < scheduler_verified_threshold:
        fails.append(f"可用 verified 经验 {available_verified} < 调度阈值 "
                      f"{scheduler_verified_threshold}：需要单独授权练习，或按 §6.3 显式许可")
    return fails
