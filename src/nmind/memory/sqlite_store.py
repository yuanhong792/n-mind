"""SQLite 记忆存储：情景记忆 + 语义记忆 + 训练记录。

设计上把三件事分开，避免"把数据库记住了说成神经网络学会了"：
  * 这里存的是**事实与经历**（可查询、可更正、可删除）；
  * 神经网络**参数**里的能力由 learning/ 负责，两者用不同指标衡量。

并发与断电：WAL + busy_timeout + 显式事务；写失败回滚不留半条记录。
去重：content_hash 唯一索引，同一事件重复写入只更新时间戳。
更正：不覆盖原文，而是新记录带 correction_of 指回旧记录，旧记录标 superseded，
      这样"谁在什么时候改口了"本身可审计。
删除：硬删 + 指纹墓碑（memory_deletions）。删除后检索与重放都取不到，
      巩固也不会把删掉的东西学回来。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .schema import ensure_schema


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def content_hash(*parts: Any) -> str:
    """对写入内容做稳定指纹（用于 UNIQUE 与墓碑），顺序无关的字段各自成段。"""
    blob = "\x1f".join("" if p is None else str(p) for p in parts)
    return hashlib.sha256(blob.encode("utf-8", "surrogatepass")).hexdigest()[:32]


@dataclass
class Episode:
    id: int
    session_id: str
    created_at: str
    task: str
    observation: str
    action: str | None
    outcome: str | None
    feedback: float | None
    source: str
    confidence: float
    privacy_level: str
    correction_of: int | None
    observation_json: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Episode":
        raw = row["observation_json"]
        return cls(
            id=row["id"], session_id=row["session_id"], created_at=row["created_at"],
            task=row["task"], observation=row["observation"], action=row["action"],
            outcome=row["outcome"], feedback=row["feedback"], source=row["source"],
            confidence=row["confidence"], privacy_level=row["privacy_level"],
            correction_of=row["correction_of"],
            observation_json=json.loads(raw) if raw else {},
        )


@dataclass
class Fact:
    id: int
    subject: str
    predicate: str
    object: str
    status: str
    confidence: float
    evidence: tuple[int, ...]
    created_at: str
    updated_at: str
    privacy_level: str = "normal"
    expires_at: str | None = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Fact":
        ev = row["evidence_episode_ids"] or ""
        return cls(
            id=row["id"], subject=row["subject"], predicate=row["predicate"],
            object=row["object"], status=row["status"], confidence=row["confidence"],
            evidence=tuple(int(v) for v in ev.split(",") if v.strip().isdigit()),
            created_at=row["created_at"], updated_at=row["updated_at"],
            privacy_level=row["privacy_level"], expires_at=row["expires_at"],
        )


class MemoryStore:
    """单文件 SQLite 记忆库。所有公开方法都不抛"库不存在"——自动建。"""

    def __init__(self, db_path: str | Path, *, busy_timeout_ms: int = 5000) -> None:
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), isolation_level=None, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute(f"PRAGMA busy_timeout = {int(busy_timeout_ms)}")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA synchronous = FULL")   # 断电可恢复优先于吞吐
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.capabilities = ensure_schema(self.conn)

    # ------------------------------------------------------------------ 写

    def add_episode(
        self,
        *,
        session_id: str,
        observation: str,
        task: str = "",
        action: str | None = None,
        outcome: str | None = None,
        feedback: float | None = None,
        source: str = "local",
        confidence: float = 0.0,
        privacy_level: str = "normal",
        correction_of: int | None = None,
        observation_json: dict[str, Any] | None = None,
    ) -> int:
        """写入一条经历；同内容重复写入返回既有 id（去重），不产生第二条。

        已删除过的内容**拒绝再写回来**，返回 -1 —— 用户删掉的东西不该被
        重放或巩固悄悄复活。
        """
        h = content_hash(session_id, task, observation, action, outcome)
        if self.is_deleted("episode", h):
            return -1
        payload = json.dumps(observation_json, ensure_ascii=False, sort_keys=True) if observation_json else None
        cur = self.conn.execute(
            """
            INSERT INTO episodes (session_id, created_at, task, observation, observation_json,
                                  action, outcome, feedback, source, confidence,
                                  privacy_level, correction_of, content_hash)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(content_hash) DO UPDATE SET
                feedback   = COALESCE(excluded.feedback, episodes.feedback),
                outcome    = COALESCE(excluded.outcome, episodes.outcome),
                confidence = MAX(excluded.confidence, episodes.confidence),
                created_at = episodes.created_at
            RETURNING id
            """,
            (session_id, _now(), task, observation, payload, action, outcome, feedback,
             source, float(confidence), privacy_level, correction_of, h),
        )
        row = cur.fetchone()
        new_id = int(row[0])
        if correction_of is not None:
            self.conn.execute("UPDATE episodes SET source='corrected' WHERE id=?", (int(correction_of),))
        return new_id

    def correct_episode(self, episode_id: int, *, observation: str, reason: str = "",
                        session_id: str | None = None) -> int:
        """更正规律：新建一条指向旧条，旧条作废但仍可查（审计需要）。"""
        old = self.conn.execute("SELECT * FROM episodes WHERE id=?", (int(episode_id),)).fetchone()
        if old is None:
            raise KeyError(f"episode {episode_id} 不存在")
        new_id = self.add_episode(
            session_id=session_id or old["session_id"],
            observation=observation,
            task=old["task"],
            action=old["action"],
            outcome=old["outcome"],
            feedback=old["feedback"],
            source=f"correction:{reason}" if reason else "correction",
            confidence=max(float(old["confidence"] or 0.0), 0.6),
            privacy_level=old["privacy_level"],
            correction_of=int(episode_id),
        )
        self.conn.execute("UPDATE episodes SET source='superseded' WHERE id=?", (int(episode_id),))
        return new_id

    def add_fact(self, *, subject: str, predicate: str, object: str,   # noqa: A002 - 与 SQL 语义对齐
                 evidence: Iterable[int] = (), confidence: float = 0.5,
                 privacy_level: str = "normal", expires_at: str | None = None) -> int:
        """写入一条主张。同一 (subject, predicate) 已有不同 object 时视为冲突：
        置信度更高者保留为 active，其余置 superseded —— 冲突是显式状态，不是静默覆盖。
        """
        sid, pid, obj = subject.strip(), predicate.strip(), str(object).strip()
        key = content_hash("fact", sid, pid, obj)
        if self.is_deleted("fact", key):
            return -1
        ev = ",".join(str(int(v)) for v in evidence)
        now = _now()
        with self.conn:      # 事务：冲突判定与写入必须原子，否则断电会留下两条 active
            existing = self.conn.execute(
                "SELECT id, object, confidence FROM facts WHERE subject=? AND predicate=? AND status='active'",
                (sid, pid)).fetchall()
            clash = [r for r in existing if r["object"] != obj]
            if clash:
                old_conf = max(float(r["confidence"] or 0.0) for r in clash)
                if confidence < old_conf:          # 新证据更弱：旧的留着，新的记为 candidate
                    status = "candidate"
                else:
                    status = "active"
                    for r in clash:
                        self.conn.execute(
                            "UPDATE facts SET status='superseded', updated_at=? WHERE id=?", (now, r["id"]))
            else:
                status = "active"
            cur = self.conn.execute(
                """
                INSERT INTO facts (subject, predicate, object, evidence_episode_ids, status,
                                   confidence, privacy_level, created_at, updated_at, expires_at, content_key)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(subject, predicate, object, status) DO UPDATE SET
                    confidence = MAX(excluded.confidence, facts.confidence),
                    evidence_episode_ids = CASE
                        WHEN excluded.evidence_episode_ids <> '' THEN excluded.evidence_episode_ids
                        ELSE facts.evidence_episode_ids END,
                    updated_at = excluded.updated_at
                RETURNING id
                """,
                (sid, pid, obj, ev, status, float(confidence), privacy_level,
                 now, now, expires_at, key),
            )
            return int(cur.fetchone()[0])

    def retire_fact(self, fact_id: int) -> None:
        self.conn.execute("UPDATE facts SET status='retired', updated_at=? WHERE id=?", (_now(), int(fact_id)))

    def delete_episode(self, episode_id: int, *, reason: str = "user") -> bool:
        """硬删除 + 墓碑。FTS 触发器会在 DELETE 时同步删索引，不留可搜副本。

        删除是**不可绕过的**：墓碑记下内容指纹，检索、巩固、重放都取不到，
        相同内容再写入也会被拒绝（见 add_episode）。原始正文另存一份在
        deleted_content，只用于向用户证明"确实删的是这条"，不参与任何召回。
        """
        row = self.conn.execute("SELECT * FROM episodes WHERE id=?", (int(episode_id),)).fetchone()
        if row is None:
            return False
        with self.conn:
            self.conn.execute("DELETE FROM episodes WHERE id=?", (int(episode_id),))
            self.conn.execute(
                "INSERT INTO memory_deletions (deleted_at, kind, content_hash, reason) VALUES (?,?,?,?)",
                (_now(), "episode", row["content_hash"], reason))
            self.conn.execute(
                """INSERT OR IGNORE INTO deleted_content
                   (content_hash, kind, observation, observation_json, task, action, outcome,
                    session_id, deleted_at, reason)
                   VALUES (?, 'episode',?,?,?,?,?,?,?,?)""",
                (row["content_hash"], row["observation"], row["observation_json"], row["task"],
                 row["action"], row["outcome"], row["session_id"], _now(), reason))
        return True

    def delete_fact(self, fact_id: int, *, reason: str = "user") -> bool:
        row = self.conn.execute("SELECT * FROM facts WHERE id=?", (int(fact_id),)).fetchone()
        if row is None:
            return False
        text = f"{row['subject']} {row['predicate']} {row['object']}"
        with self.conn:
            self.conn.execute("DELETE FROM facts WHERE id=?", (int(fact_id),))
            self.conn.execute(
                "INSERT INTO memory_deletions (deleted_at, kind, content_hash, reason) VALUES (?,?,?,?)",
                (_now(), "fact", row["content_key"], reason))
            self.conn.execute(
                """INSERT OR IGNORE INTO deleted_content
                   (content_hash, kind, observation, observation_json, task, action, outcome,
                    session_id, deleted_at, reason)
                   VALUES (?, 'fact', ?, NULL, 'fact', NULL, NULL, ?, ?, ?)""",
                (row["content_key"], text, row["status"], _now(), reason))
        return True

    def is_deleted(self, kind: str, h: str) -> bool:
        """指纹查墓碑。kind 只用于日志可读性：不同 kind 的哈希前缀本身不同（见 content_hash 调用点）。"""
        row = self.conn.execute(
            "SELECT 1 FROM deleted_content WHERE content_hash=? AND kind=?", (h, kind)).fetchone()
        if row is not None:
            return True
        return self.conn.execute(
            "SELECT 1 FROM memory_deletions WHERE kind=? AND content_hash=?", (kind, h)).fetchone() is not None

    def expire_facts(self) -> int:
        cur = self.conn.execute(
            "UPDATE facts SET status='expired', updated_at=? WHERE status='active' AND expires_at IS NOT NULL AND expires_at < ?",
            (_now(), _now()))
        return int(cur.rowcount or 0)

    # ------------------------------------------------------------------ 读

    def get_fact(self, subject: str, predicate: str) -> Fact | None:
        row = self.conn.execute(
            "SELECT * FROM facts WHERE subject=? AND predicate=? AND status='active' "
            "ORDER BY confidence DESC, updated_at DESC LIMIT 1", (subject.strip(), predicate.strip())).fetchone()
        return Fact.from_row(row) if row else None

    def latest_episode(self, session_id: str | None = None, task: str | None = None) -> Episode | None:
        q, args = "SELECT * FROM episodes WHERE 1=1", []
        if session_id:
            q += " AND session_id=?"
            args.append(session_id)
        if task:
            q += " AND task=?"
            args.append(task)
        q += " AND source <> 'superseded' ORDER BY id DESC LIMIT 1"
        row = self.conn.execute(q, args).fetchone()
        return Episode.from_row(row) if row else None

    def recent_episodes(self, limit: int = 20, task: str | None = None,
                        session_id: str | None = None) -> list[Episode]:
        q, args = "SELECT * FROM episodes WHERE source <> 'superseded'", []
        if task:
            q += " AND task=?"
            args.append(task)
        if session_id:
            q += " AND session_id=?"
            args.append(session_id)
        q += " ORDER BY id DESC LIMIT ?"
        args.append(int(limit))
        return [Episode.from_row(r) for r in self.conn.execute(q, args).fetchall()]

    def active_facts(self, limit: int = 200) -> list[Fact]:
        rows = self.conn.execute(
            "SELECT * FROM facts WHERE status IN ('active','candidate') ORDER BY updated_at DESC LIMIT ?",
            (int(limit),)).fetchall()
        return [Fact.from_row(r) for r in rows]

    def record_training_run(self, *, config_hash: str, dataset_hash: str,
                            metrics: dict[str, Any], checkpoint: str | None = None,
                            git_revision: str = "", notes: str = "") -> int:
        cur = self.conn.execute(
            "INSERT INTO training_runs (created_at, config_hash, git_revision, dataset_hash, metrics_json, checkpoint_path, notes) "
            "VALUES (?,?,?,?,?,?,?)",
            (_now(), config_hash, git_revision, dataset_hash,
             json.dumps(metrics, ensure_ascii=False, sort_keys=True), checkpoint, notes))
        return int(cur.lastrowid)

    def training_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM training_runs ORDER BY id DESC LIMIT ?", (int(limit),)).fetchall()
        return [{k: r[k] for k in r.keys()} for r in rows]

    def stats(self) -> dict[str, Any]:
        e = self.conn.execute("SELECT COUNT(*) c FROM episodes").fetchone()["c"]
        f = self.conn.execute("SELECT COUNT(*) c FROM facts WHERE status='active'").fetchone()["c"]
        d = self.conn.execute("SELECT COUNT(*) c FROM memory_deletions").fetchone()["c"]
        t = self.conn.execute("SELECT COUNT(*) c FROM training_runs").fetchone()["c"]
        by_status = {r["status"]: r["c"] for r in self.conn.execute(
            "SELECT status, COUNT(*) c FROM facts GROUP BY status")}
        return {"episodes": int(e), "active_facts": int(f), "deleted_tombstones": int(d),
                "training_runs": int(t), "facts_by_status": by_status,
                "schema": self.capabilities,
                "db_bytes": int(self.path.stat().st_size) if self.path.exists() else 0}

    def export_json(self) -> str:
        """用户数据导出（规则 7：用户可读走自己的记忆）。不含密钥。"""
        return json.dumps({
            "exported_at": _now(),
            "episodes": [{k: r[k] for k in r.keys()} for r in
                         self.conn.execute("SELECT * FROM episodes ORDER BY id")],
            "facts": [{k: r[k] for k in r.keys()} for r in
                      self.conn.execute("SELECT * FROM facts ORDER BY id")],
            "training_runs": self.training_runs(1000),
            "deleted_count": self.conn.execute("SELECT COUNT(*) c FROM memory_deletions").fetchone()["c"],
        }, ensure_ascii=False, indent=2)

    def close(self) -> None:
        self.conn.close()
