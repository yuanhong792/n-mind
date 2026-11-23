from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
import uuid


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class EvidenceRef:
    kind: str
    title: str = ""
    uri: str = ""
    excerpt: str = ""
    trust: str = "unverified"


@dataclass
class V3Event:
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    event_type: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TaskRun:
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    title: str = ""
    status: str = "queued"
    steps: list[dict[str, Any]] = field(default_factory=list)
    current_step: int = 0
    error: str = ""
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
