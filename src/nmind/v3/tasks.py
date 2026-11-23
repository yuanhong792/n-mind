from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .contracts import TaskRun, now_iso
from .events import EventLog


_ALLOWED = {"queued": {"running", "cancelled"}, "running": {"paused", "completed", "failed", "cancelled"}, "paused": {"running", "cancelled"}, "failed": set(), "completed": set(), "cancelled": set()}


class TaskService:
    def __init__(self, path: str | Path, events: EventLog):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.events = events
        self.db.execute("CREATE TABLE IF NOT EXISTS v3_tasks(task_id TEXT PRIMARY KEY,payload TEXT NOT NULL,updated_at TEXT NOT NULL)")
        self.db.commit()

    def _save(self, task: TaskRun) -> None:
        task.updated_at = now_iso()
        self.db.execute("INSERT OR REPLACE INTO v3_tasks VALUES(?,?,?)", (task.task_id, json.dumps(task.to_dict(), ensure_ascii=False), task.updated_at))
        self.db.commit()

    def create(self, title: str, steps: list[dict[str, Any]]) -> TaskRun:
        task = TaskRun(title=title[:200], steps=steps[:32])
        self._save(task)
        self.events.append("task.created", task.to_dict())
        return task

    def get(self, task_id: str) -> TaskRun | None:
        row = self.db.execute("SELECT payload FROM v3_tasks WHERE task_id=?", (task_id,)).fetchone()
        return TaskRun(**json.loads(row[0])) if row else None

    def transition(self, task_id: str, status: str, error: str = "") -> TaskRun:
        task = self.get(task_id)
        if task is None:
            raise KeyError(task_id)
        if status not in _ALLOWED.get(task.status, set()):
            if task.status == status: return task
            raise ValueError(f"invalid transition {task.status}->{status}")
        task.status, task.error = status, error[:500]
        self._save(task)
        self.events.append("task.state_changed", task.to_dict())
        return task

    def list(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.db.execute("SELECT payload FROM v3_tasks ORDER BY updated_at DESC LIMIT ?", (max(1, min(500, limit)),)).fetchall()
        return [json.loads(r[0]) for r in rows]

    def close(self) -> None: self.db.close()
