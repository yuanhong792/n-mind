"""经验桥：练习结果 → VerifiedExperience → 学习流（M5 任务书 experience_bridge）。

* 只有 verification=verified 的经验能进入神经学习流（rejected/uncertain 隔离）；
* 来源（source_id/content_sha256/model_checkpoint_sha256）全程保留；
* 神经尝试失败但工具成功的经验同样 verified——但 outcome_source=tool，
  永不归属神经核心。
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


def build_experience(task: dict, *, neural_attempt: str, tool_result: str | None,
                     verification: str, validator_id: str,
                     model_sha: str, exp_id: str) -> VerifiedExperience:
    blob = json.dumps({"q": task["question"], "a": task["answer"]},
                       ensure_ascii=False, sort_keys=True)
    return VerifiedExperience(
        experience_id=exp_id, task_family=task["family"],
        observation={"question": task["question"]},
        action={"neural_attempt": neural_attempt},
        feedback={"tool_result": tool_result, "oracle": task["answer"]},
        verification_status=verification, validator_id=validator_id,
        source_id=f"practice:{task['family']}",
        content_sha256=hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16],
        model_checkpoint_sha256=model_sha)


def to_learning_rows(experiences: list[VerifiedExperience]) -> tuple[list[str], list[str]]:
    """verified 经验 → 语言训练行（prompt, answer）；rejected/uncertain 永不进入。"""
    prompts, answers = [], []
    for e in experiences:
        if e.verification_status != "verified":
            continue
        prompts.append(f"用户：{e.observation.get('question', '')}\n助手：")
        answers.append(str(e.feedback.get("oracle", "")))
    return prompts, answers


def verified_only(experiences: list[VerifiedExperience]) -> list[VerifiedExperience]:
    return [e for e in experiences if e.verification_status == "verified"]


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")
