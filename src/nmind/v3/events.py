from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .contracts import V3Event


class EventLog:
    """Append-only local audit log. Payloads are data, never commands."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("CREATE TABLE IF NOT EXISTS v3_events(event_id TEXT PRIMARY KEY,event_type TEXT NOT NULL,payload TEXT NOT NULL,created_at TEXT NOT NULL)")
        self.db.commit()

    def append(self, event_type: str, payload: dict[str, Any] | None = None) -> V3Event:
        event = V3Event(event_type=event_type, payload=payload or {})
        self.db.execute("INSERT INTO v3_events VALUES(?,?,?,?)", (event.event_id, event.event_type, json.dumps(event.payload, ensure_ascii=False), event.created_at))
        self.db.commit()
        return event

    def list(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.db.execute("SELECT * FROM v3_events ORDER BY rowid DESC LIMIT ?", (max(1, min(1000, limit)),)).fetchall()
        return [{"event_id": r["event_id"], "event_type": r["event_type"], "payload": json.loads(r["payload"]), "created_at": r["created_at"]} for r in rows]

    def close(self) -> None:
        self.db.close()
