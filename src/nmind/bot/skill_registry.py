"""技能注册表（R2 任务书 §2）：SkillSpec 契约 + 四类归因枚举 + 内置技能声明。

目的：把"这件事到底是谁做出来的"变成结构化字段，而不是靠读日志猜。
R1 的做法是 capability_id 用 `arithmetic:neural` / `arithmetic:tool` 后缀区分，
"神经选对工具"和"工具算对结果"仍然记在同一族的账上；R2 拆成四个显式字段：

  * response_source   —— 这段文本是谁产出的：neural / tool / retrieval / rule / abstain
  * selection_source  —— 这一步动作为什么发生：neural / rule / user / scheduler
  * feedback_source   —— 判定正确的依据从哪来：deterministic_oracle / externally_verified /
                         user_approved / unverified
  * learning_target   —— 这条经验允许更新什么：neural_generation / neural_tool_selection /
                         memory_only / none

硬口径（写进注册表而不是靠注释）：
  * `neural_evaluation_enabled=False` 的技能不得进入神经评测矩阵；
  * `neural_training_allowed=False` 的技能，其 verified 经验不得进入学习流（工具任务、
    词表外任务都属此类）；
  * 未知能力一律 `ABSTAIN` + `learning_target=none`，不允许"看起来像会答"；
  * 外部网页文本与用户消息都不能新增/修改 SkillSpec（注册表只由代码与本地配置构造，
    `source="code"` 才可信）。
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

RESPONSE_SOURCES = ("neural", "tool", "retrieval", "rule", "abstain")
SELECTION_SOURCES = ("neural", "rule", "user", "scheduler")
FEEDBACK_SOURCES = ("deterministic_oracle", "externally_verified", "user_approved", "unverified")
LEARNING_TARGETS = ("neural_generation", "neural_tool_selection", "memory_only", "none")
TIERS = ("A", "B", "C", "X", "TOOL")
CAPABILITY_STATES = ("NEURAL_MASTERED", "NEURAL_PARTIAL", "TOOL_VERIFIED", "MEMORY_ONLY",
                     "RULE_ONLY", "UNSUPPORTED", "UNTESTED", "ABSTAIN")

SKILL_ID_RE = re.compile(r"^skill\.[a-z0-9_]+\.[a-z0-9_]+$")


@dataclass(frozen=True)
class SkillSpec:
    skill_id: str
    label: str
    tier: str                                        # A 已知复现 / B 相邻新技能 / C 困难诊断 / X 词表外 / TOOL 工具
    task_family: str
    input_schema: dict[str, str] = field(default_factory=dict)
    output_schema: dict[str, str] = field(default_factory=dict)
    difficulty: int = 1                              # 1..5（相对父模型，不是相对人类）
    prerequisite_skill_ids: tuple[str, ...] = ()
    verifier_id: str = "none"
    verifier_standalone: bool = False                # oracle 是否独立于模型输出（§6.4）
    allowed_response_sources: tuple[str, ...] = ("neural", "abstain")
    allowed_tools: tuple[str, ...] = ()
    safe_online_sources: tuple[str, ...] = ()        # 空 = 不允许联网
    neural_evaluation_enabled: bool = True
    neural_training_allowed: bool = True
    learning_target: str = "neural_generation"
    response_source_expected: str = "neural"
    max_input_chars: int = 48
    max_response_tokens: int = 24
    capability_state: str = "UNTESTED"               # Gate-A 后由桥接层回填
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["prerequisite_skill_ids"] = list(self.prerequisite_skill_ids)
        d["allowed_response_sources"] = list(self.allowed_response_sources)
        d["allowed_tools"] = list(self.allowed_tools)
        d["safe_online_sources"] = list(self.safe_online_sources)
        return d

    def check(self) -> list[str]:
        """契约自检：返回违规说明列表（空 = 合法）。注册进表前必须通过。"""
        errs: list[str] = []
        if not SKILL_ID_RE.match(self.skill_id):
            errs.append(f"skill_id 形如 skill.<域>.<名>：{self.skill_id}")
        if self.tier not in TIERS:
            errs.append(f"tier 必须 ∈ {TIERS}，实为 {self.tier}")
        if not self.input_schema or not self.output_schema:
            errs.append(f"{self.skill_id} 缺 input/output schema")
        if any(s not in RESPONSE_SOURCES for s in self.allowed_response_sources):
            errs.append(f"{self.skill_id} allowed_response_sources 含未知来源")
        if self.response_source_expected not in RESPONSE_SOURCES:
            errs.append(f"{self.skill_id} response_source_expected 非法")
        if self.learning_target not in LEARNING_TARGETS:
            errs.append(f"{self.skill_id} learning_target 非法")
        if self.capability_state not in CAPABILITY_STATES:
            errs.append(f"{self.skill_id} capability_state 非法")
        if self.tier in ("TOOL", "X") and self.neural_training_allowed:
            errs.append(f"{self.skill_id} 为 {self.tier} 级，不得允许神经训练")
        if self.tier in ("C", "X") and self.neural_training_allowed:
            errs.append(f"{self.skill_id} 诊断/词表外族未经单独许可不得训练")
        if self.neural_training_allowed and not self.verifier_standalone:
            errs.append(f"{self.skill_id} 训练需要独立于模型输出的验证器")
        if self.max_input_chars <= 0 or self.max_response_tokens <= 0:
            errs.append(f"{self.skill_id} 输入/输出预算必须为正")
        return errs


class SkillRegistry:
    """代码构造的技能表；查询、校验、序列化。不做任何模型调用。"""

    def __init__(self, specs: list[SkillSpec] | None = None, *, source: str = "code") -> None:
        self.source = source
        self._specs: dict[str, SkillSpec] = {}
        self.rejected: list[dict[str, Any]] = []
        for s in (specs or []):
            self.register(s)

    def register(self, spec: SkillSpec) -> bool:
        errs = spec.check()
        if errs:
            self.rejected.append({"skill_id": spec.skill_id, "errors": errs})
            return False
        if spec.skill_id in self._specs:
            self.rejected.append({"skill_id": spec.skill_id, "errors": ["重复注册，已忽略"]})
            return False
        self._specs[spec.skill_id] = spec
        return True

    def get(self, skill_id: str) -> SkillSpec | None:
        return self._specs.get(skill_id)

    def by_family(self, family: str) -> list[SkillSpec]:
        return [s for s in self._specs.values() if s.task_family == family]

    def by_tier(self, *tiers: str) -> list[SkillSpec]:
        return [s for s in self._specs.values() if s.tier in tiers]

    def trainable(self) -> list[SkillSpec]:
        return [s for s in self._specs.values() if s.neural_training_allowed]

    def evaluable(self) -> list[SkillSpec]:
        return [s for s in self._specs.values() if s.neural_evaluation_enabled]

    def with_state(self, skill_id: str, state: str) -> SkillSpec:
        """回填能力状态（Gate-A/B 用）；返回新实例，注册表按 skill_id 替换。"""
        old = self._specs.get(skill_id)
        if old is None:
            raise KeyError(skill_id)
        new = SkillSpec(**{**asdict(old), "capability_state": state})
        self._specs[skill_id] = new
        return new

    def abstain(self, reason: str = "未注册能力") -> dict[str, Any]:
        """未知能力的唯一合法出口：拒答 + 不学习（任务书 §2 要求显式 UNSUPPORTED/ABSTAIN）。"""
        return {"response_source": "abstain", "answer": "", "capability_state": "ABSTAIN",
                "learning_target": "none", "reason": reason, "selection_source": "rule",
                "feedback_source": "unverified"}

    def to_json(self) -> dict[str, Any]:
        return {"format": "nmind-skill-registry/1", "source": self.source,
                "counts": {"skills": len(self._specs), "rejected": len(self.rejected),
                            "by_tier": {t: len(self.by_tier(t)) for t in TIERS}},
                "vocabularies": {"response_source": list(RESPONSE_SOURCES),
                                  "selection_source": list(SELECTION_SOURCES),
                                  "feedback_source": list(FEEDBACK_SOURCES),
                                  "learning_target": list(LEARNING_TARGETS),
                                  "capability_state": list(CAPABILITY_STATES)},
                "skills": [s.to_dict() for s in self._specs.values()],
                "rejected": self.rejected}

    def save(self, path: str | Path) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")
        return str(p)

    @classmethod
    def from_json(cls, path: str | Path) -> "SkillRegistry":
        blob = json.loads(Path(path).read_text(encoding="utf-8"))
        if blob.get("format") != "nmind-skill-registry/1":
            raise ValueError(f"不是 nmind 技能表（format={blob.get('format')!r}）")
        reg = cls(source=blob.get("source", "unknown"))
        for d in blob.get("skills", []):
            kw = dict(d)
            for k in ("prerequisite_skill_ids", "allowed_response_sources", "allowed_tools",
                        "safe_online_sources"):
                if k in kw:
                    kw[k] = tuple(kw[k])
            reg.register(SkillSpec(**kw))
        return reg


# ------------------------------------------------------------------ 内置技能声明

def _binding(skill_id: str, family: str, label: str, tier: str, *, difficulty: int,
             prereq: tuple[str, ...], notes: str, max_input_chars: int = 64,
             evaluable: bool = True, trainable: bool = True,
             expected: str = "neural") -> SkillSpec:
    return SkillSpec(
        skill_id=skill_id, label=label, tier=tier, task_family=family,
        input_schema={"context": "str(陈述句：实体+位置)", "question": "str(位置问句)",
                       "format": "用户：{context}\\n{question}\\n助手："},
        output_schema={"answer": "str(“{实体}在{位置}。”)", "stop": "eos|newline|budget"},
        difficulty=difficulty, prerequisite_skill_ids=prereq,
        verifier_id="binding_oracle_v1", verifier_standalone=True,
        allowed_response_sources=("neural", "abstain") if trainable else ("neural", "tool",
                                                                            "rule", "abstain"),
        allowed_tools=(), safe_online_sources=(),
        neural_evaluation_enabled=evaluable, neural_training_allowed=trainable,
        learning_target="neural_generation" if trainable else "none",
        response_source_expected=expected, max_input_chars=max_input_chars,
        capability_state="UNTESTED", notes=notes)


def builtin_skills() -> list[SkillSpec]:
    """R2 内置技能：绑定域 A/B/C/X + R1 的工具族（保留为工具能力，不再当神经目标）。"""
    a = "skill.binding.position_qa"
    specs = [
        _binding(a, "position_qa", "单实体位置问答（已知技能）", "A", difficulty=1,
                 prereq=(), evaluable=True, trainable=False,
                 notes="A 组只用于复现与遗忘检查；父模型已掌握，重复训练只会制造虚假收益"),
        _binding("skill.binding.paraphrase", "position_qa_paraphrase", "同义问法位置问答", "B",
                 difficulty=2, prereq=(a,),
                 notes="相对 A 只改问句形式；问句字符须全在 stable 词表内"),
        _binding("skill.binding.unseen_entity", "unseen_entity_qa", "第二实体位置绑定", "B",
                 difficulty=2, prereq=(a,),
                 notes="相对 A 只改实体值（小红）；父模型 S0 阶段没见过小红被陈述"),
        _binding("skill.binding.latest_update", "latest_statement_update", "两次陈述取最新", "B",
                 difficulty=3, prereq=(a,),
                 notes="词表内写法的事实更新（不用 把/移动）；答案取最后一条陈述"),
        _binding("skill.binding.two_entity_swap", "binding_swap", "两实体绑定+近因陷阱", "C",
                 difficulty=4, prereq=(a, "skill.binding.unseen_entity"), trainable=False,
                 evaluable=True,
                 notes="诊断族：除 EM 外必须报反事实跟随率；未经单独许可不得训练"),
        _binding("skill.binding.combo_probe", "combo_probe", "跨族组合泛化探针", "C",
                 difficulty=5, prereq=(a,), trainable=False, evaluable=True,
                 notes="多变量同时变化，只用于诊断 unseen 组合，不作学习目标"),
        _binding("skill.binding.unseen_position", "unseen_position_qa", "未见位置取值", "X",
                 difficulty=3, prereq=(a,), trainable=False, evaluable=True,
                 expected="abstain",
                 notes="词表只有数字 1-4：5-8 号位置必然出现 <unk>，保留为分布错配对照"),
    ]
    tool_common = dict(input_schema={"question": "str"}, output_schema={"answer": "str"},
                        difficulty=2, verifier_standalone=True, neural_evaluation_enabled=True,
                        neural_training_allowed=False, learning_target="none",
                        response_source_expected="tool", capability_state="TOOL_VERIFIED",
                        max_input_chars=160, max_response_tokens=24)
    specs += [
        SkillSpec(skill_id="skill.tool.arithmetic", label="算术（计算器工具）", tier="TOOL",
                  task_family="arithmetic", allowed_tools=("calculator",),
                  allowed_response_sources=("tool", "rule", "abstain"),
                  prerequisite_skill_ids=(), verifier_id="arithmetic_oracle_v1",
                  safe_online_sources=(), notes=R1_TOOL_NOTE, **tool_common),
        SkillSpec(skill_id="skill.tool.json_extract", label="JSON 字段抽取", tier="TOOL",
                  task_family="json_extract", allowed_tools=("json_extractor",),
                  allowed_response_sources=("tool", "retrieval", "rule", "abstain"),
                  prerequisite_skill_ids=(), verifier_id="json_field_oracle_v1",
                  safe_online_sources=(), notes=R1_TOOL_NOTE, **tool_common),
        SkillSpec(skill_id="skill.tool.grid_rules", label="网格规则（只读程序）", tier="TOOL",
                  task_family="grid_rules", allowed_tools=(),
                  allowed_response_sources=("rule", "abstain"), prerequisite_skill_ids=(),
                  verifier_id="grid_rules_program_oracle_v1", safe_online_sources=(),
                  notes=R1_TOOL_NOTE, **tool_common),
    ]
    return specs


R1_TOOL_NOTE = (
    "R1 练习族：题面含大量词表外字符（阿拉伯多位数、拉丁标识符、英文标点），"
    "父模型读到的是 <unk> 流。有可信外部工具/程序时可验证成功，"
    "但其 verified 经验只允许写 memory_only，不得作为神经生成学习目标（任务书 §3.4/§6.4）。")


def default_registry() -> SkillRegistry:
    return SkillRegistry(builtin_skills())
