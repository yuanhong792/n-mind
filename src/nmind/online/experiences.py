"""Experience 对象与持久化（字段与 M4 任务书第 3 节一致）。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

VERIFICATION_STATES = ("verified", "rejected", "uncertain")
TRAIN_SPLITS = ("stream_train", "stream_val", "sealed_eval")


@dataclass
class Experience:
    id: str
    observed_at: str                       # ISO-8601 UTC
    source_url: str | None
    source_sha256: str
    raw_excerpt: str
    structured_observation: dict
    action: dict | None
    expected_outcome: dict | None
    actual_outcome: dict | None
    verifier_id: str | None
    verification: str                      # verified | rejected | uncertain
    reward: float | None
    task_family: str
    train_split: str                       # stream_train | stream_val | sealed_eval

    def __post_init__(self) -> None:
        if self.verification not in VERIFICATION_STATES:
            raise ValueError(f"verification 非法：{self.verification!r}")
        if self.train_split not in TRAIN_SPLITS:
            raise ValueError(f"train_split 非法：{self.train_split!r}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Experience":
        return cls(**d)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save_jsonl(path: Path, items: list[Experience]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it.to_dict(), ensure_ascii=False) + "\n")
    return len(items)


def load_jsonl(path: Path) -> list[Experience]:
    out = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(Experience.from_dict(json.loads(line)))
    return out


def build_learning_stream(experiences: list[Experience], *,
                          allow_splits: tuple[str, ...] = ("stream_train",),
                          allow_verification: tuple[str, ...] = ("verified",)) -> list[Experience]:
    """学习流的唯一入口：只有指定 split + 通过验证的经验才可进入。

    sealed_eval 在任何情况下都进不了学习流——请求包含它时直接抛错（硬约束，
    test_sealed_eval_never_enters_replay 钉住）；rejected/uncertain 经验同样被滤除。
    """
    if "sealed_eval" in allow_splits:
        raise ValueError("sealed_eval 永远不能进入学习流/回放（硬约束）")
    return [e for e in experiences
            if e.verification in allow_verification and e.train_split in allow_splits]
