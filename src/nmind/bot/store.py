"""Bot SQLite 持久化：任务、能力、事件、模型世系（WAL + 事务，断电安全）。"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    run_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS capabilities (
    capability_id TEXT PRIMARY KEY,
    task_family TEXT NOT NULL,
    attempts INTEGER DEFAULT 0,
    verified_successes INTEGER DEFAULT 0,
    failures INTEGER DEFAULT 0,
    heldout_score REAL,
    outcome_source TEXT DEFAULT 'none',
    tool_invocations INTEGER DEFAULT 0,
    network_invocations INTEGER DEFAULT 0,
    weight_update_steps INTEGER DEFAULT 0,
    model_sha TEXT DEFAULT '',
    evidence_path TEXT DEFAULT '',
    status TEXT DEFAULT 'UNTESTED',
    updated_at TEXT
);
CREATE TABLE IF NOT EXISTS completed_tasks (
    task_id TEXT PRIMARY KEY,
    ts TEXT NOT NULL,
    task_family TEXT,
    outcome TEXT
);
CREATE TABLE IF NOT EXISTS kv (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT
);
"""


class BotStore:
    def __init__(self, db_path: str | Path) -> None:
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), isolation_level=None, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA synchronous = FULL")
        self.conn.executescript(_SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        """老库补列：completed_tasks.fingerprint（内容指纹去重）；R2 归因五列。"""
        cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(completed_tasks)")}
        if "fingerprint" not in cols:
            self.conn.execute("ALTER TABLE completed_tasks ADD COLUMN fingerprint TEXT")
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_completed_fingerprint ON completed_tasks(fingerprint)")
        # R2：一条经验"答对了"不等于"神经学会了"，必须把来源与用途写进同一行，
        # 否则台账又会把工具/检索/规则的成功混成神经能力（R1 的账目就是这么糊的）。
        cap_cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(capabilities)")}
        for col, decl in (("skill_id", "TEXT DEFAULT ''"),
                          ("response_source", "TEXT DEFAULT 'unknown'"),
                          ("feedback_source", "TEXT DEFAULT 'unverified'"),
                          ("learning_target", "TEXT DEFAULT 'none'"),
                          ("capability_state", "TEXT DEFAULT 'UNTESTED'")):
            if col not in cap_cols:
                self.conn.execute(f"ALTER TABLE capabilities ADD COLUMN {col} {decl}")

    def event(self, run_id: str, kind: str, payload: dict[str, Any]) -> None:
        self.conn.execute(
            "INSERT INTO events (ts, run_id, kind, payload) VALUES (?,?,?,?)",
            (time.strftime("%Y-%m-%dT%H:%M:%S%z"), run_id, kind,
             json.dumps(payload, ensure_ascii=False)))

    def upsert_capability(self, rec: dict[str, Any]) -> None:
        """字段补全后写库：调用方给不全也不允许 sqlite 报缺参（历史 bug：
        tracker.summary() 只有 5 个字段，直接绑定时 ProgrammingError 打断空闲循环）。"""
        merged: dict[str, Any] = {
            "capability_id": rec["capability_id"], "task_family": rec.get("task_family", ""),
            "attempts": rec.get("attempts", 0),
            "verified_successes": rec.get("verified_successes", 0),
            "failures": rec.get("failures", 0), "heldout_score": rec.get("heldout_score"),
            "outcome_source": rec.get("outcome_source", "none"),
            "tool_invocations": rec.get("tool_invocations", 0),
            "network_invocations": rec.get("network_invocations", 0),
            "weight_update_steps": rec.get("weight_update_steps", 0),
            "model_sha": rec.get("model_sha", ""), "status": rec.get("status", "UNTESTED"),
            "evidence_path": rec.get("evidence_path", ""),
            "skill_id": rec.get("skill_id", ""),
            "response_source": rec.get("response_source", "unknown"),
            "feedback_source": rec.get("feedback_source", "unverified"),
            "learning_target": rec.get("learning_target", "none"),
            "capability_state": rec.get("capability_state", rec.get("status", "UNTESTED")),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        merged.update(rec)
        merged["capability_id"] = rec["capability_id"]
        self.conn.execute(
            """INSERT INTO capabilities (capability_id, task_family, attempts,
                verified_successes, failures, heldout_score, outcome_source,
                tool_invocations, network_invocations, weight_update_steps,
                model_sha, evidence_path, status, updated_at,
                skill_id, response_source, feedback_source, learning_target,
                capability_state)
                VALUES (:capability_id,:task_family,:attempts,:verified_successes,
                :failures,:heldout_score,:outcome_source,:tool_invocations,
                :network_invocations,:weight_update_steps,:model_sha,:evidence_path,
                :status,:updated_at,:skill_id,:response_source,:feedback_source,
                :learning_target,:capability_state)
                ON CONFLICT(capability_id) DO UPDATE SET
                attempts=excluded.attempts, verified_successes=excluded.verified_successes,
                failures=excluded.failures, heldout_score=excluded.heldout_score,
                outcome_source=excluded.outcome_source,
                tool_invocations=excluded.tool_invocations,
                network_invocations=excluded.network_invocations,
                weight_update_steps=excluded.weight_update_steps,
                model_sha=excluded.model_sha, evidence_path=excluded.evidence_path,
                status=excluded.status, updated_at=excluded.updated_at,
                skill_id=excluded.skill_id, response_source=excluded.response_source,
                feedback_source=excluded.feedback_source,
                learning_target=excluded.learning_target,
                capability_state=excluded.capability_state""", merged)

    def capabilities(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM capabilities ORDER BY task_family, capability_id")]

    def set_state(self, key: str, value: Any) -> None:
        self.conn.execute(
            "INSERT INTO kv (key, value, updated_at) VALUES (?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
            (key, json.dumps(value, ensure_ascii=False),
             time.strftime("%Y-%m-%dT%H:%M:%S%z")))

    def get_state(self, key: str, default: Any = None) -> Any:
        row = self.conn.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def list_events(self, *, limit: int = 20, kind: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT ts, run_id, kind, payload FROM events"
        args: tuple = ()
        if kind:
            sql += " WHERE kind=?"
            args = (kind,)
        sql += " ORDER BY id DESC LIMIT ?"
        rows = self.conn.execute(sql, args + (int(limit),)).fetchall()
        out = []
        for r in rows:
            rec = dict(r)
            rec["payload"] = json.loads(rec["payload"])
            out.append(rec)
        return out

    def prune_events(self, *, keep: int = 5000) -> int:
        """日志容量上限：只保留最近 keep 条事件（任务书 §6 防无限增长）。"""
        cur = self.conn.execute("SELECT COUNT(*) AS c FROM events").fetchone()
        total = int(cur["c"])
        if total <= keep:
            return 0
        self.conn.execute(
            "DELETE FROM events WHERE id NOT IN "
            "(SELECT id FROM events ORDER BY id DESC LIMIT ?)", (int(keep),))
        return total - keep

    def capability(self, capability_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM capabilities WHERE capability_id=?",
                                 (capability_id,)).fetchone()
        return dict(row) if row else None

    def mark_task_done(self, task_id: str, task_family: str, outcome: str,
                      fingerprint: str = "") -> None:
        fp = fingerprint or task_id
        self.conn.execute(
            "INSERT OR IGNORE INTO completed_tasks (task_id, ts, task_family, outcome,"
            " fingerprint) VALUES (?,?,?,?,?)",
            (task_id, time.strftime("%Y-%m-%dT%H:%M:%S%z"), task_family, outcome, fp))

    def task_done(self, task_id: str) -> bool:
        row = self.conn.execute("SELECT 1 FROM completed_tasks WHERE task_id=?",
                                 (task_id,)).fetchone()
        return row is not None

    def task_seen(self, fingerprint: str) -> bool:
        """按内容指纹判断该任务实例是否已练过（重启/重复运行的幂等依据）。"""
        row = self.conn.execute(
            "SELECT 1 FROM completed_tasks WHERE fingerprint=? LIMIT 1", (fingerprint,)).fetchone()
        return row is not None

    def completed_count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) AS c FROM completed_tasks").fetchone()["c"])

    def close(self) -> None:
        self.conn.close()
