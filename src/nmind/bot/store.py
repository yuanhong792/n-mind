"""Bot SQLite 持久化：任务、能力、事件、模型世系（WAL + 事务，断电安全）。"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any


def _shared_connection_lock(fn):
    """Serialize every access to BotStore's shared sqlite connection.

    RLock is intentional: public methods call other public methods during normal
    operation (for example enqueue_message -> create_session).
    """
    def wrapped(self, *args, **kwargs):
        with self._conn_lock:
            return fn(self, *args, **kwargs)
    wrapped.__name__ = fn.__name__
    wrapped.__doc__ = fn.__doc__
    return wrapped

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


_CHAT_SCHEMA = """
-- Bot V1 聊天持久化（独立命名表，绝不改动 R1 的 events/capabilities/completed_tasks/kv）
CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chat_messages (
    message_id TEXT NOT NULL,          -- 用户消息 UUID；assistant 回复行复用同一 message_id（role 区分）
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,                -- user | assistant
    text TEXT NOT NULL,
    response_source TEXT DEFAULT '',   -- neural | retrieval | tool | rule | none
    citations_json TEXT,
    status TEXT DEFAULT 'QUEUED',      -- QUEUED | RUNNING | DONE | FAILED | CANCELLED | RETRYABLE
    created_at TEXT NOT NULL,
    updated_at TEXT,
    model_sha TEXT DEFAULT '',
    protocol_id TEXT DEFAULT '',
    latency_ms REAL,
    error TEXT,
    seq INTEGER,                       -- 会话内序号（回放顺序可复核）
    PRIMARY KEY (message_id, role)
);
CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages(session_id, seq);
CREATE INDEX IF NOT EXISTS idx_chat_messages_status ON chat_messages(status);
"""


class BotStore:
    def __init__(self, db_path: str | Path) -> None:
        self._conn_lock = threading.RLock()
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), isolation_level=None, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA synchronous = FULL")
        self.conn.execute("PRAGMA busy_timeout = 5000")
        self.conn.executescript(_SCHEMA)
        self._migrate()
        self.conn.executescript(_CHAT_SCHEMA)
        self._recover_interrupted()

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

    @_shared_connection_lock
    def event(self, run_id: str, kind: str, payload: dict[str, Any]) -> None:
        self.conn.execute(
            "INSERT INTO events (ts, run_id, kind, payload) VALUES (?,?,?,?)",
            (time.strftime("%Y-%m-%dT%H:%M:%S%z"), run_id, kind,
             json.dumps(payload, ensure_ascii=False)))

    @_shared_connection_lock
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

    @_shared_connection_lock
    def capabilities(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM capabilities ORDER BY task_family, capability_id")]

    @_shared_connection_lock
    def set_state(self, key: str, value: Any) -> None:
        self.conn.execute(
            "INSERT INTO kv (key, value, updated_at) VALUES (?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
            (key, json.dumps(value, ensure_ascii=False),
             time.strftime("%Y-%m-%dT%H:%M:%S%z")))

    @_shared_connection_lock
    def get_state(self, key: str, default: Any = None) -> Any:
        row = self.conn.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    @_shared_connection_lock
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

    @_shared_connection_lock
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

    @_shared_connection_lock
    def capability(self, capability_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM capabilities WHERE capability_id=?",
                                 (capability_id,)).fetchone()
        return dict(row) if row else None

    @_shared_connection_lock
    def mark_task_done(self, task_id: str, task_family: str, outcome: str,
                      fingerprint: str = "") -> None:
        fp = fingerprint or task_id
        self.conn.execute(
            "INSERT OR IGNORE INTO completed_tasks (task_id, ts, task_family, outcome,"
            " fingerprint) VALUES (?,?,?,?,?)",
            (task_id, time.strftime("%Y-%m-%dT%H:%M:%S%z"), task_family, outcome, fp))

    @_shared_connection_lock
    def task_done(self, task_id: str) -> bool:
        row = self.conn.execute("SELECT 1 FROM completed_tasks WHERE task_id=?",
                                 (task_id,)).fetchone()
        return row is not None

    @_shared_connection_lock
    def task_seen(self, fingerprint: str) -> bool:
        """按内容指纹判断该任务实例是否已练过（重启/重复运行的幂等依据）。"""
        row = self.conn.execute(
            "SELECT 1 FROM completed_tasks WHERE fingerprint=? LIMIT 1", (fingerprint,)).fetchone()
        return row is not None

    @_shared_connection_lock
    def completed_count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) AS c FROM completed_tasks").fetchone()["c"])

    @_shared_connection_lock
    def close(self) -> None:
        self.conn.close()

    # ------------------------------------------------------------ Bot V1 聊天持久化
    # 设计要点：message_id(UUID) 领取对应回复，不是读最新日志冒充；
    # 领取走 BEGIN IMMEDIATE 事务，跨进程并发发消息不丢不重；
    # 启动时 QUEUED/RUNNING → RETRYABLE（断电重启不静默丢队列）。
    def _recover_interrupted(self) -> None:
        self.conn.execute(
            "UPDATE chat_messages SET status='RETRYABLE', updated_at=?, error='进程中断，重启后标记为可重试' "
            "WHERE role='user' AND status IN ('QUEUED','RUNNING')",
            (time.strftime("%Y-%m-%dT%H:%M:%S%z"),))

    @_shared_connection_lock
    def create_session(self, session_id: str, title: str) -> None:
        now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.conn.execute(
            "INSERT OR IGNORE INTO chat_sessions (session_id, title, created_at, updated_at) "
            "VALUES (?,?,?,?)", (session_id, title, now, now))

    @_shared_connection_lock
    def list_sessions(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM chat_sessions ORDER BY updated_at DESC")]

    @_shared_connection_lock
    def delete_session(self, session_id: str) -> int:
        cur = self.conn.execute("DELETE FROM chat_messages WHERE session_id=?", (session_id,))
        self.conn.execute("DELETE FROM chat_sessions WHERE session_id=?", (session_id,))
        return cur.rowcount

    @_shared_connection_lock
    def rename_session(self, session_id: str, title: str) -> None:
        self.conn.execute(
            "UPDATE chat_sessions SET title=?, updated_at=? WHERE session_id=?",
            (title, time.strftime("%Y-%m-%dT%H:%M:%S%z"), session_id))

    @_shared_connection_lock
    def session_title(self, session_id: str) -> str | None:
        row = self.conn.execute("SELECT title FROM chat_sessions WHERE session_id=?",
                                (session_id,)).fetchone()
        return row["title"] if row else None

    @_shared_connection_lock
    def enqueue_message(self, message_id: str, session_id: str, text: str) -> dict[str, Any]:
        """入队用户消息（原子追加；同会话自动建目录并刷新 updated_at）。"""
        now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            self.conn.execute(
                "INSERT INTO chat_sessions (session_id,title,created_at,updated_at) "
                "VALUES (?,?,?,?) ON CONFLICT(session_id) DO NOTHING",
                (session_id, text[:20] or "新会话", now, now))
            seq = self.conn.execute(
                "SELECT COALESCE(MAX(seq),0)+1 AS s FROM chat_messages WHERE session_id=?",
                (session_id,)).fetchone()["s"]
            self.conn.execute(
                "INSERT INTO chat_messages (message_id, session_id, role, text, status, "
                "created_at, seq) VALUES (?,?,?,?,?,?,?)",
                (message_id, session_id, "user", text, "QUEUED", now, int(seq)))
            self.conn.execute("UPDATE chat_sessions SET updated_at=? WHERE session_id=?",
                              (now, session_id))
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise
        return {"message_id": message_id, "session_id": session_id, "status": "QUEUED",
                "seq": int(seq)}

    @_shared_connection_lock
    def claim_next_message(self) -> dict[str, Any] | None:
        """领取下一条待处理消息（QUEUED/RETRYABLE 中最早一条）→ RUNNING。
        BEGIN IMMEDIATE 保证两个进程/线程不会领到同一条。"""
        now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        try:
            self.conn.execute("BEGIN IMMEDIATE")
            row = self.conn.execute(
                "SELECT * FROM chat_messages WHERE role='user' AND status IN ('QUEUED','RETRYABLE') "
                "ORDER BY created_at LIMIT 1").fetchone()
            if row is None:
                self.conn.execute("COMMIT")
                return None
            self.conn.execute(
                "UPDATE chat_messages SET status='RUNNING', updated_at=? "
                "WHERE message_id=? AND role='user'", (now, row["message_id"]))
            self.conn.execute("COMMIT")
        except sqlite3.OperationalError:
            return None
        rec = dict(row)
        rec["status"] = "RUNNING"
        return rec

    @_shared_connection_lock
    def finish_message(self, message_id: str, *, reply_text: str, response_source: str,
                       citations: list[dict[str, Any]] | None = None, model_sha: str = "",
                       protocol_id: str = "", latency_ms: float = 0.0,
                       note: str = "", extra: dict[str, Any] | None = None) -> None:
        """写回复行（role=assistant，同一 message_id）并把用户行标 DONE。

        extra：附加协议字段（unk_rate/finish_reason 等），随 note 一起以 JSON
        落库，领取时还原——契约字段不因持久化丢字段。
        """
        now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        row = self.conn.execute(
            "SELECT session_id FROM chat_messages WHERE message_id=? AND role='user'",
            (message_id,)).fetchone()
        seq = None
        if row:
            seq = self.conn.execute(
                "SELECT COALESCE(MAX(seq),0)+1 AS s FROM chat_messages WHERE session_id=? AND role='assistant'",
                (row["session_id"],)).fetchone()["s"]
        meta: dict[str, Any] = {"note": note}
        if extra:
            meta.update(extra)
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO chat_messages (message_id, session_id, role, text, "
                "response_source, citations_json, status, created_at, updated_at, model_sha, "
                "protocol_id, latency_ms, error, seq) VALUES "
                "(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (message_id, row["session_id"] if row else "", "assistant", reply_text,
                 response_source, json.dumps(citations or [], ensure_ascii=False), "DONE",
                 now, now, model_sha, protocol_id, float(latency_ms),
                 json.dumps(meta, ensure_ascii=False) if len(meta) > 1 or note else None,
                 seq))
            self.conn.execute(
                "UPDATE chat_messages SET status='DONE', updated_at=? "
                "WHERE message_id=? AND role='user'", (now, message_id))

    @_shared_connection_lock
    def fail_message(self, message_id: str, error: str) -> None:
        now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        with self.conn:
            self.conn.execute(
                "UPDATE chat_messages SET status='FAILED', updated_at=?, error=? "
                "WHERE message_id=? AND role='user'", (now, error[:2000], message_id))

    @_shared_connection_lock
    def get_reply(self, message_id: str) -> dict[str, Any] | None:
        """用 message_id 领取对应回复（没有就 None——调用方继续等，不拿别的消息冒充）。"""
        row = self.conn.execute(
            "SELECT * FROM chat_messages WHERE message_id=? AND role='assistant'",
            (message_id,)).fetchone()
        if row is None:
            user = self.conn.execute(
                "SELECT status FROM chat_messages WHERE message_id=? AND role='user'",
                (message_id,)).fetchone()
            if user and user["status"] == "FAILED":
                return {"message_id": message_id, "status": "FAILED", "role": "assistant",
                        "text": "", "response_source": "none"}
            return None
        rec = dict(row)
        rec["citations"] = json.loads(rec["citations_json"] or "[]")
        rec["note"] = ""
        rec.pop("unk_rate", None)
        rec.pop("finish_reason", None)
        if rec.get("error"):
            try:
                meta = json.loads(rec["error"])
                if isinstance(meta, dict):
                    rec["note"] = meta.pop("note", "")
                    for k in ("unk_rate", "finish_reason", "memory_hits"):
                        if k in meta:
                            rec[k] = meta[k]
            except (ValueError, AttributeError):
                rec["note"] = ""
        return rec

    @_shared_connection_lock
    def message_status(self, message_id: str) -> str | None:
        row = self.conn.execute(
            "SELECT status FROM chat_messages WHERE message_id=? AND role='user'",
            (message_id,)).fetchone()
        return row["status"] if row else None

    @_shared_connection_lock
    def cancel_message(self, message_id: str) -> bool:
        cur = self.conn.execute(
            "UPDATE chat_messages SET status='CANCELLED', updated_at=? "
            "WHERE message_id=? AND role='user' AND status IN ('QUEUED','RETRYABLE')",
            (time.strftime("%Y-%m-%dT%H:%M:%S%z"), message_id))
        self.conn.commit()
        return cur.rowcount > 0

    @_shared_connection_lock
    def list_messages(self, session_id: str, *, limit: int = 200) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM chat_messages WHERE session_id=? ORDER BY seq LIMIT ?",
            (session_id, int(limit))).fetchall()
        out = []
        for r in rows:
            rec = dict(r)
            rec["citations"] = json.loads(rec["citations_json"] or "[]")
            if rec.get("error"):
                try:
                    meta = json.loads(rec["error"])
                    if isinstance(meta, dict):
                        rec["note"] = meta.pop("note", "")
                        for k in ("unk_rate", "finish_reason", "memory_hits"):
                            if k in meta:
                                rec[k] = meta[k]
                except (ValueError, AttributeError):
                    rec["note"] = ""
            else:
                rec["note"] = ""
            out.append(rec)
        return out

    @_shared_connection_lock
    def queue_len(self) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) AS c FROM chat_messages WHERE role='user' "
            "AND status IN ('QUEUED','RUNNING','RETRYABLE')").fetchone()
        return int(row["c"])
