from __future__ import annotations
from dataclasses import dataclass
from .world import LanguageWorld

@dataclass(frozen=True)
class CurriculumTask:
    task_id: str; family: str; difficulty: int; prompt: str; expected: str

def starter_tasks() -> list[CurriculumTask]:
    return [CurriculumTask("name-1", "object_name", 1, "桌上有什么颜色的苹果？", "红色"), CurriculumTask("loc-1", "location_query", 1, "书在哪里？", "盒子"), CurriculumTask("move-1", "move_update", 2, "把苹果放到盒子。然后苹果在哪里？", "盒子")]

def run_oracle_task(task: CurriculumTask) -> dict:
    world = LanguageWorld()
    if task.family == "move_update": world.act("move", "苹果", "盒子")
    result = world.oracle(task.prompt.split("然后")[-1])
    return {"task_id": task.task_id, "family": task.family, "expected": task.expected, "oracle": result, "verified": result.get("answer") == task.expected}
