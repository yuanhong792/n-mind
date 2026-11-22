"""经验桥：练习结果 → VerifiedExperience → 学习流（M5 任务书 experience_bridge）。

* 只有 verification=verified 的经验能进入神经学习流（rejected/uncertain 隔离）；
* 来源（source_id/content_sha256/model_checkpoint_sha256）全程保留；
* 神经尝试失败但工具成功的经验同样 verified——但 outcome_source=tool，
  永不归属神经核心。

R2 把"永不归属神经核心"变成**代码级拦截**：每条经验带 learning_target，
`to_learning_rows()` 只收 `learning_target="neural_generation"` 的行。
R1 的候选训练之所以 0 成长，就是因为工具/规则经验也进了 `to_learning_rows()`，
把一个词表都表达不了的任务喂给了同一个 checkpoint。
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class VerifiedExperience:
    experience_id: str
    task_family: str
    observation: dict
    action: dict
    feedback: dict
    verification_status: str          # verified | rejected | uncertain
    validator_id: str
    source_id: str
    content_sha256: str
    model_checkpoint_sha256: str
    skill_id: str = ""
    response_source: str = "unknown"
    feedback_source: str = "unverified"
    learning_target: str = "none"


def build_experience(task: dict, *, neural_attempt: str, tool_result: str | None,
                     verification: str, validator_id: str,
                     model_sha: str, exp_id: str,
                     skill_id: str = "", response_source: str = "unknown",
                     feedback_source: str = "unverified",
                     learning_target: str = "none") -> VerifiedExperience:
    blob = json.dumps({"q": task["question"], "a": task["answer"]},
                       ensure_ascii=False, sort_keys=True)
    return VerifiedExperience(
        experience_id=exp_id, task_family=task["family"],
        observation={"question": task["question"]},
        action={"neural_attempt": neural_attempt},
        feedback={"tool_result": tool_result, "oracle": task["answer"],
                  "feedback_source": feedback_source},
        verification_status=verification, validator_id=validator_id,
        source_id=f"practice:{task['family']}",
        content_sha256=hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16],
        model_checkpoint_sha256=model_sha, skill_id=skill_id,
        response_source=response_source, feedback_source=feedback_source,
        learning_target=learning_target)


def to_learning_rows(experiences: list[VerifiedExperience]
                     ) -> tuple[list[str], list[str], dict[str, int]]:
    """verified 且 learning_target=neural_generation 的经验 → 语言训练行（prompt, answer）。

    第三个返回值是被挡掉的行数分类计数（skipped_reasons），调用方必须把它写进日志：
    "为什么这轮没训练"要能回答，不能只报 0。
    """
    prompts, answers = [], []
    skipped: dict[str, int] = {"not_verified": 0, "wrong_learning_target": 0,
                                "wrong_feedback_source": 0}
    for e in experiences:
        if e.verification_status != "verified":
            skipped["not_verified"] += 1
            continue
        if e.learning_target != "neural_generation":
            skipped["wrong_learning_target"] += 1
            continue
        if e.feedback_source not in ("deterministic_oracle", "externally_verified",
                                     "user_approved"):
            skipped["wrong_feedback_source"] += 1
            continue
        prompts.append(f"用户：{e.observation.get('question', '')}\n助手：")
        answers.append(str(e.feedback.get("oracle", "")))
    return prompts, answers, {k: v for k, v in skipped.items() if v}


def verified_only(experiences: list[VerifiedExperience]) -> list[VerifiedExperience]:
    return [e for e in experiences if e.verification_status == "verified"]


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")
