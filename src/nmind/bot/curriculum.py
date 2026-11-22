"""自主学习技能选择器（R2 任务书 §4）：把 Gate-A 测出来的真实缺口变成"下一轮练什么"。

与 R1 `GoalManager` 的分工：GoalManager 管**练习族白名单**（arithmetic/json_extract/grid_rules），
本模块管**技能级可学资格**（A/B/C/X/TOOL + 已注册 skill_id）。两者都只写 `selection_source=rule`
或 `scheduler`，都不宣称是"学习型探索策略"。

七个评分因子（§4.2，全部有出处，不靠拍脑袋）：
  * learning_gap        —— Gate-A 实测的 oracle 上限差（越大越该学）
  * prerequisite_ready  —— 前置技能的实测掌握度（不足则相邻技能也学不会）
  * verifier_available  —— 独立客观验证器是否存在（§2/§6.4 硬门槛）
  * dataset_quality     —— 学习流规模/去重/与封存集不重叠（数据本身可不可用）
  * novelty             —— 与父模型训练文本、与近期已练任务的重叠度（越低越新）
  * expected_cost       —— 该族每轮预计耗时（预算感知）
  * forgetting_risk     —— 与已掌握技能的分布距离（越大越容易冲掉旧技能）

资格硬门槛（分数再高也不能选）：tier 必须 B、`neural_training_allowed=True`、
验证器独立、缺口 ≥ 预登记阈值、词表内。全被卡住 → `NO_ELIGIBLE_TASK` 休眠（§4.4），
这是"对齐没问题、只是没东西可学"，不等于神经学习失败。

探索比例 `exploration_fraction` 有界且确定性（种子=cycle index），来源只能是代码/配置；
外部网页文本与用户消息一律不能改白名单、预算、阈值（§4.5）。
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from typing import Any

NO_ELIGIBLE_TASK = "NO_ELIGIBLE_TASK"
POLICY_SOURCES = ("code", "config", "user_authorized")


@dataclass
class CurriculumConfig:
    min_gap_pp: float = 10.0                  # 与 bridge.GATE_A_ELIGIBLE_GAP_PP 同值（预登记）
    exploration_fraction: float = 0.2         # 有界确定性探索，不超过 1/5 轮次
    max_attempts_per_skill: int = 8
    min_dataset_rows: int = 16                # 学习流太少谈不上"可靠验证条件"
    expected_cycle_seconds: float = 20.0
    weights: dict[str, float] = field(default_factory=lambda: {
        "learning_gap": 0.35, "prerequisite_ready": 0.20, "verifier_available": 0.15,
        "dataset_quality": 0.10, "novelty": 0.10, "expected_cost": 0.05,
        "forgetting_risk": 0.05})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SkillAssessment:
    skill_id: str
    family: str
    tier: str
    eligible: bool
    gate_a_verdict: str
    neural_em: float | None
    gap_pp: float
    prerequisite_ready: float
    verifier_available: float
    dataset_quality: float
    novelty: float
    expected_cost: float
    forgetting_risk: float
    template_space: int = 0
    stream_disjoint: bool = True
    reasons: list[str] = field(default_factory=list)
    rejections: list[str] = field(default_factory=list)
    components: dict[str, float] = field(default_factory=dict)
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def guard_policy_source(value: Any, *, source: str, what: str) -> Any:
    """策略字段只接受代码/本地配置/显式授权三个来源；其余（网页正文、用户消息）一律拒绝。"""
    if source not in POLICY_SOURCES:
        raise ValueError(f"{what} 只能来自 {POLICY_SOURCES}，实为 {source!r}：外部文本不得提权")
    return value


class SkillSelector:
    def __init__(self, cfg: CurriculumConfig | None = None, *,
                 allowed_skill_ids: tuple[str, ...] = ()) -> None:
        self.cfg = cfg or CurriculumConfig()
        self.allowed_skill_ids = tuple(allowed_skill_ids)     # 白名单来自 policy（代码/配置）

    # ---------------------------------------------------------------- 评分

    def _components(self, a: SkillAssessment) -> dict[str, float]:
        w = self.cfg.weights
        return {
            "learning_gap": max(0.0, min(1.0, a.gap_pp / 100.0)) * w["learning_gap"],
            "prerequisite_ready": a.prerequisite_ready * w["prerequisite_ready"],
            "verifier_available": a.verifier_available * w["verifier_available"],
            "dataset_quality": a.dataset_quality * w["dataset_quality"],
            "novelty": a.novelty * w["novelty"],
            # 成本与遗忘风险是"越小越好"，取反后加权
            "expected_cost": (1.0 - max(0.0, min(1.0, a.expected_cost))) * w["expected_cost"],
            "forgetting_risk": (1.0 - max(0.0, min(1.0, a.forgetting_risk))) * w["forgetting_risk"],
        }

    def score(self, a: SkillAssessment) -> float:
        a.components = self._components(a)
        a.score = round(sum(a.components.values()), 4)
        return a.score

    # ---------------------------------------------------------------- 评估

    def assess(self, *, gate_a: dict[str, Any], registry: Any,
               dataset_stats: dict[str, dict[str, Any]] | None = None,
               tracker: Any = None, recent_families: tuple[str, ...] = ()) -> list[SkillAssessment]:
        """把 Gate-A 的族级判定 + 数据/台账统计合成可评分的评估列表。"""
        dataset_stats = dataset_stats or {}
        fams = gate_a.get("families", {})
        em_by_skill = {(v.get("skill_id") or f): float(v.get("neural_em") or 0.0)
                       for f, v in fams.items()}
        out: list[SkillAssessment] = []
        for family, v in fams.items():
            specs = registry.by_family(family)
            skill = next((s for s in specs if s.skill_id == v.get("skill_id")),
                         specs[0] if specs else None)
            skill_id = v.get("skill_id") or (skill.skill_id if skill else "")
            ds = dataset_stats.get(family, {})
            gap = round((1.0 - float(v.get("neural_em") or 0.0)) * 100, 2)
            prereqs = list(skill.prerequisite_skill_ids) if skill else []
            ready = round(sum(em_by_skill.get(p, 0.0) for p in prereqs) / len(prereqs), 4) \
                if prereqs else 1.0
            novelty = round(1.0 - max(0.0, min(1.0, float(ds.get("parent_overlap_rate", 0.0)))), 4)
            if family in recent_families:
                novelty = round(novelty * 0.5, 4)      # 近期已练 → 新颖度折半
            a = SkillAssessment(
                skill_id=skill_id, family=family, tier=v["tier"],
                eligible=False, gate_a_verdict=v["verdict"], neural_em=v["neural_em"],
                gap_pp=gap, prerequisite_ready=ready,
                verifier_available=float(bool(v["oracle_standalone"]
                                              and v["oracle_agrees_with_labels"])),
                dataset_quality=round(min(1.0, ds.get("n_train", 0) / self.cfg.min_dataset_rows) *
                                      (1.0 - ds.get("train_sealed_overlap_rate", 0.0)), 4),
                novelty=novelty,
                expected_cost=round(ds.get("est_cycle_s", 0.0) / self.cfg.expected_cycle_seconds, 4),
                forgetting_risk=round(ds.get("distribution_distance", 0.2), 4),
                template_space=int(ds.get("template_space", 0)),
                stream_disjoint=bool(ds.get("stream_disjoint", False)),
                reasons=[f"Gate-A={v['verdict']}", f"实测 EM={v['neural_em']}", f"缺口 {gap}pp"])
            a.rejections = self._rejections(a, v)
            a.eligible = not a.rejections
            self.score(a)
            out.append(a)
        out.sort(key=lambda x: (-x.score, x.skill_id))
        return out

    def _rejections(self, a: SkillAssessment, v: dict[str, Any]) -> list[str]:
        """资格硬门槛：任一不过即不可选，且必须留下可审计的拒绝原因（§4.4）。"""
        r: list[str] = []
        if self.allowed_skill_ids and a.skill_id not in self.allowed_skill_ids:
            r.append(f"skill_id 不在策略白名单：{a.skill_id}")
        if v["verdict"] == "UNSUPPORTED_VOCAB":
            r.append("题面含词表外字符（UNK 流），不能作神经目标")
        if v["verdict"] == "TOOL_ONLY":
            r.append("工具/程序任务：答案由工具产生，只允许 memory_only")
        if v["verdict"] == "DIAGNOSTIC_ONLY":
            r.append("困难技能仅诊断，未经单独许可不得训练")
        if v["verdict"] == "REPRODUCED_KNOWN_SKILL":
            r.append("A 组是已知技能，重复训练只会制造虚假收益")
        if v["verdict"] == "ALREADY_MASTERED":
            r.append(f"相邻技能已掌握（EM={a.neural_em}），无可靠缺口")
        if not a.verifier_available:
            r.append("缺独立客观验证器（或 oracle 与标签不一致）")
        if a.dataset_quality <= 0.0:
            r.append("学习流数据不足（预登记 train 行为 0 或与未见测试集完全重叠）")
        if not a.stream_disjoint:
            r.append(f"学习流与未见测试集在输入指纹层面不互斥（模板空间仅 {a.template_space} 个"
                     "不同输入）→ 无法区分'泛化'与'背过'，本轮不训练")
        if a.gap_pp < self.cfg.min_gap_pp:
            r.append(f"缺口 {a.gap_pp}pp < 预登记阈值 {self.cfg.min_gap_pp}pp")
        if v["tier"] != "B":
            r.append(f"tier={v['tier']} ≠ B（只有相邻新技能允许自主训练）")
        if not v.get("neural_training_allowed", True):
            r.append("注册表声明该技能不允许神经训练")
        return r

    # ---------------------------------------------------------------- 选择

    def pick(self, assessments: list[SkillAssessment], *, cycle_index: int = 0,             exclude_families: tuple[str, ...] = ()) -> dict[str, Any]:
        """确定性选择 + 有界探索：探索只在 eligible 池内部换座位，绝不越过资格硬门槛。"""
        eligible = [a for a in assessments if a.eligible]
        pool = [a for a in eligible if a.family not in exclude_families] or eligible
        if not pool:
            return {"status": NO_ELIGIBLE_TASK, "selected": None,
                    "reason": f"没有满足资格硬门槛的技能（已排除 {list(exclude_families) or '无'}）",
                    "rejected_evidence": [
                        {"skill_id": a.skill_id, "family": a.family, "tier": a.tier,
                         "score": a.score, "rejections": a.rejections}
                        for a in sorted(assessments, key=lambda x: -x.score)][:8],
                    "selection_source": "rule"}
        rng = random.Random(20261600 + int(cycle_index))     # 种子=cycle 序号，可复现
        explore = rng.random() < self.cfg.exploration_fraction
        if explore:
            idx = int(cycle_index) % len(pool)
            chosen = pool[idx]
            why = (f"有界探索轮（exploration_fraction={self.cfg.exploration_fraction}，"
                   f"seed=cycle{cycle_index}）在 {len(pool)} 个合格技能里取第 {idx} 位")
        else:
            chosen, why = pool[0], "取评分最高者（贪心）"
        return {"status": "SELECTED", "selected": chosen.to_dict(),
                "selection_source": "rule", "exploration": bool(explore),
                "reason": why + f"；评分构成={chosen.components}",
                "ranking": [{"skill_id": a.skill_id, "family": a.family, "score": a.score,
                              "eligible": a.eligible} for a in assessments],
                "rejected_evidence": [{"skill_id": a.skill_id, "family": a.family,
                                        "tier": a.tier, "rejections": a.rejections}
                                       for a in assessments if not a.eligible][:8]}


def dataset_stats(tok: Any, gate_a: dict[str, Any], *, per_item_s: float,
                  rows_per_cycle: int = 24) -> dict[str, dict[str, Any]]:
    """每个族的学习流体检：规模、去重、与未见测试集/父模型训练文本的重叠、与父分布的距离、预计耗时。

    学习流一律按 `binding_data.SPLIT_PLAN` 的**预登记规模**取（不做 n 覆盖）：覆盖会把
    48 行压成 24 行，进而是模板空间饱和的假象——那正是 R1 把"背过 16 个格子"当成能力的坑。

    `distribution_distance` 只是**先验估计**（题面 token 集合与父模型训练文本的 Jaccard 距离），
    真正的遗忘率由 Gate-B 的 A 组封存集实测；报告里不得把这一列当成遗忘证据。
    """
    from nmind.language import binding_data as bd

    parent = bd.parent_train_prompts()
    parent_ids = set(tok.encode("\n".join(parent)))
    out: dict[str, dict[str, Any]] = {}
    for family in gate_a.get("families", {}):
        if family not in bd.FAMILIES:
            continue
        try:
            train = bd.build_family(family, "train")
        except Exception as exc:                       # 该族没有学习流（A/C 或词表外）
            out[family] = {"n_train": 0, "train_sealed_overlap_rate": 1.0,
                           "parent_overlap_rate": 1.0, "est_cycle_s": 0.0,
                           "distribution_distance": 0.0, "stream_disjoint": False,
                           "no_dataset_reason": f"{type(exc).__name__}: {exc}"}
            continue
        sealed = bd.build_family(family, "sealed")
        tr_fp, se_fp = bd.fingerprint_set(train), bd.fingerprint_set(sealed)
        lk = bd.leakage(tok, train, parent)
        overlap = round(len(tr_fp & se_fp) / max(1, len(tr_fp)), 4)
        ids = set().union(*[set(tok.encode(t.prompt)) for t in train]) if train else set()
        dist = round(1.0 - len(ids & parent_ids) / max(1, len(ids | parent_ids)), 4)
        out[family] = {
            "n_train": len(train), "n_unique_train": len(tr_fp),
            "n_eval": len(sealed), "n_unique_eval": len(se_fp),
            "eval_split": "sealed",
            "template_space": len(tr_fp | se_fp),
            "stream_disjoint": bool(train) and not (tr_fp & se_fp),
            "train_sealed_overlap_rate": overlap,
            "parent_overlap_rate": lk["overlap_rate"],
            "distinct_combos": bd.slot_coverage(train)["distinct_combos"],
            "distribution_distance": dist,
            "distribution_metric": "jaccard_token_distance_vs_parent_train(estimate)",
            "est_cycle_s": round(per_item_s * min(rows_per_cycle, len(train)), 3),
            "per_item_s": round(per_item_s, 5),
        }
    return out
