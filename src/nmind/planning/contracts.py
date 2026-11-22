from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any
import uuid

def now() -> str: return datetime.now(timezone.utc).isoformat(timespec="seconds")

@dataclass(frozen=True)
class GoalSpec:
    goal_id: str
    user_request: str
    origin: str = "user"
    created_at: str = field(default_factory=now)
    success_criteria: tuple[str, ...] = ()
    scope: dict[str, Any] = field(default_factory=dict)
    budget: dict[str, Any] = field(default_factory=lambda: {"max_steps": 8, "max_tool_calls": 12, "max_replans": 2})
    permission_snapshot_hash: str = "offline"
    status: str = "CREATED"

    @classmethod
    def create(cls, request: str, **kwargs: Any) -> "GoalSpec":
        return cls(goal_id=uuid.uuid4().hex, user_request=request, **kwargs)

@dataclass(frozen=True)
class PlanStep:
    step_id: str
    task_id: str
    kind: str
    objective: str
    dependencies: tuple[str, ...] = ()
    tool_id: str | None = None
    input_refs: tuple[str, ...] = ()
    verifier_id: str | None = None
    idempotency_key: str = ""
    max_attempts: int = 1
    timeout_s: float = 30.0
    requires_approval: bool = False

@dataclass(frozen=True)
class Plan:
    task_id: str
    goal_id: str
    version: int
    source: str
    steps: tuple[PlanStep, ...]
    created_at: str = field(default_factory=now)
    validation_errors: tuple[str, ...] = ()

@dataclass
class StepResult:
    task_id: str
    step_id: str
    status: str
    response_source: str
    selection_source: str
    feedback_source: str
    output: dict[str, Any] = field(default_factory=dict)
    evidence_refs: list[str] = field(default_factory=list)
    started_at: str = field(default_factory=now)
    finished_at: str = ""
    failure_code: str | None = None

def as_json(obj: Any) -> dict[str, Any]:
    d = asdict(obj)
    if isinstance(obj, Plan): d["steps"] = [asdict(s) for s in obj.steps]
    return d
