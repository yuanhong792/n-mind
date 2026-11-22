from __future__ import annotations
from typing import Any
def record_experience(memory, task_id: str, request: str, run: dict[str, Any]) -> int:
    success = run.get("status") == "SUCCEEDED"
    return memory.observe(session_id=f"planning:{task_id}", task=request, observation=("成功方法：" if success else "失败原因：") + str(run), action="bounded_plan_executor", outcome=run.get("status"), feedback=1.0 if success else 0.0, source="planning_reflection", confidence=.8 if success else .4, memory_kind="procedural", verification_state="verified" if success else "unverified", task_id=task_id, tags=["reflection_source=rule"])
