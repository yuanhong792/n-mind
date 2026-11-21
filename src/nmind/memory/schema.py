"""N-Memory 的表结构与迁移。

三张表各自职责不同，**不能混成一个 JSON 列表**（指令第 5 节）：
  episodes      情景记忆：发生过什么（观察/动作/结果/反馈/来源/更正链/隐私级别）
  facts         语义记忆：认为什么为真（主-谓-宾 + 证据 + 状态），可废弃可冲突
  training_runs 训练记录：哪次训练、什么配置、什么指标、权重在哪

版本用 schema_version 表管理。迁移是"按版本号依次执行"的显式列表，
新库直接建到最新版本，老库逐版本升级 —— 不依赖删除重建，避免用户记忆丢失。

隐私与删除：
  privacy_level 标记内容敏感程度，云端策略据此决定能不能带出去；
  memory_deletions 记录被用户删除的条目指纹，既防止 FTS 残留，
  也防止重放/巩固把已删除的内容又"学回来"。
"""

from __future__ import annotations

import sqlite3
from typing import Callable

SCHEMA_VERSION = 1

DDL_V1 = """
CREATE TABLE IF NOT EXISTS schema_version (
    version    INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS episodes (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id     TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    task           TEXT NOT NULL DEFAULT '',
    observation    TEXT NOT NULL,
    observation_json TEXT,
    action         TEXT,
    outcome        TEXT,
    feedback       REAL,
    source         TEXT NOT NULL DEFAULT 'local',
    confidence     REAL NOT NULL DEFAULT 0.0,
    privacy_level  TEXT NOT NULL DEFAULT 'normal',
    correction_of  INTEGER,
    content_hash   TEXT NOT NULL UNIQUE,
    embedding      TEXT,
    FOREIGN KEY (correction_of) REFERENCES episodes (id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_ep_session  ON episodes (session_id);
CREATE INDEX IF NOT EXISTS idx_ep_task     ON episodes (task, created_at);
CREATE INDEX IF NOT EXISTS idx_ep_feedback ON episodes (feedback);

CREATE TABLE IF NOT EXISTS facts (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    subject             TEXT NOT NULL,
    predicate           TEXT NOT NULL,
    object              TEXT NOT NULL,
    evidence_episode_ids TEXT NOT NULL DEFAULT '',
    status              TEXT NOT NULL DEFAULT 'active',
    confidence          REAL NOT NULL DEFAULT 0.0,
    privacy_level       TEXT NOT NULL DEFAULT 'normal',
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL,
    expires_at          TEXT,
    content_key         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_fact_key    ON facts (subject, predicate, status);
CREATE INDEX IF NOT EXISTS idx_fact_status ON facts (status);
CREATE UNIQUE INDEX IF NOT EXISTS uq_fact_triple ON facts (subject, predicate, object, status);

CREATE TABLE IF NOT EXISTS training_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT NOT NULL,
    config_hash     TEXT NOT NULL,
    git_revision    TEXT NOT NULL DEFAULT '',
    dataset_hash    TEXT NOT NULL,
    metrics_json    TEXT NOT NULL,
    checkpoint_path TEXT,
    notes           TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS memory_deletions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    deleted_at   TEXT NOT NULL,
    kind         TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    reason       TEXT NOT NULL DEFAULT 'user'
);

CREATE TABLE IF NOT EXISTS deleted_content (
    content_hash      TEXT PRIMARY KEY,
    kind              TEXT NOT NULL DEFAULT 'episode',
    observation       TEXT NOT NULL,
    observation_json  TEXT,
    task              TEXT NOT NULL DEFAULT '',
    action            TEXT,
    outcome           TEXT,
    session_id        TEXT NOT NULL,
    deleted_at        TEXT NOT NULL,
    reason            TEXT NOT NULL DEFAULT 'user'
);

CREATE INDEX IF NOT EXISTS idx_ep_content_hash ON episodes (content_hash);
CREATE INDEX IF NOT EXISTS idx_fact_content_key ON facts (content_key);
"""

# FTS5 是可选的：编译时没带就退回 LIKE，功能不断崖（指令禁止沉重外部依赖）
DDL_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
    observation, outcome, task, content='episodes', content_rowid='id',
    tokenize='unicode61'
);
CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
    subject, predicate, object, content='facts', content_rowid='id',
    tokenize='unicode61'
);
"""

MIGRATIONS: dict[int, Callable[[sqlite3.Connection], None]] = {}


def _install_ep_triggers(conn: sqlite3.Connection) -> None:
    """把 episodes/facts 的变更同步进 FTS 索引（外部内容表要求手工触发器）。"""
    conn.executescript(
        """
        CREATE TRIGGER IF NOT EXISTS ep_ai AFTER INSERT ON episodes BEGIN
          INSERT INTO episodes_fts (rowid, observation, outcome, task)
          VALUES (new.id, new.observation, coalesce(new.outcome,''), new.task);
        END;
        CREATE TRIGGER IF NOT EXISTS ep_ad AFTER DELETE ON episodes BEGIN
          INSERT INTO episodes_fts (episodes_fts, rowid, observation, outcome, task)
          VALUES ('delete', old.id, old.observation, coalesce(old.outcome,''), old.task);
        END;
        CREATE TRIGGER IF NOT EXISTS ep_au AFTER UPDATE ON episodes BEGIN
          INSERT INTO episodes_fts (episodes_fts, rowid, observation, outcome, task)
          VALUES ('delete', old.id, old.observation, coalesce(old.outcome,''), old.task);
          INSERT INTO episodes_fts (rowid, observation, outcome, task)
          VALUES (new.id, new.observation, coalesce(new.outcome,''), new.task);
        END;
        CREATE TRIGGER IF NOT EXISTS fa_ai AFTER INSERT ON facts BEGIN
          INSERT INTO facts_fts (rowid, subject, predicate, object)
          VALUES (new.id, new.subject, new.predicate, new.object);
        END;
        CREATE TRIGGER IF NOT EXISTS fa_ad AFTER DELETE ON facts BEGIN
          INSERT INTO facts_fts (facts_fts, rowid, subject, predicate, object)
          VALUES ('delete', old.id, old.subject, old.predicate, old.object);
        END;
        """
    )


def fts_available(conn: sqlite3.Connection) -> bool:
    """探测 SQLite 是否编译了 FTS5；缺了就退回 LIKE，不报错、不下载任何东西。"""
    try:
        conn.execute("CREATE VIRTUAL TABLE TEMP._fts_probe USING fts5(a)")
        conn.execute("DROP TABLE TEMP._fts_probe")
        return True
    except sqlite3.OperationalError:
        return False


def ensure_schema(conn: sqlite3.Connection) -> dict[str, object]:
    """建库或升级。新库一次建到最新版本；老库按 MIGRATIONS 逐级升，绝不删除重建。

    用户记忆是最高优先级资产：迁移失败必须抛错让调用方停机，
    而不是"重建一个空库"糊过去。
    """
    fresh = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='episodes'").fetchone() is None
    if fresh:
        conn.executescript(DDL_V1)
        conn.execute(
            "INSERT OR IGNORE INTO schema_version (version, applied_at) VALUES (1, datetime('now'))")
    else:
        current = int(conn.execute("SELECT COALESCE(MAX(version),0) FROM schema_version").fetchone()[0])
        for v in sorted(MIGRATIONS):
            if v > current:
                MIGRATIONS[v](conn)
                conn.execute(
                    "INSERT OR IGNORE INTO schema_version (version, applied_at) VALUES (?, datetime('now'))", (v,))
                conn.commit()          # 每级迁移单独提交，失败可定位到具体版本

    has_fts = fts_available(conn)
    if has_fts:
        conn.executescript(DDL_FTS)
        _install_ep_triggers(conn)
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    conn.commit()
    version = int(conn.execute("SELECT COALESCE(MAX(version),0) FROM schema_version").fetchone()[0])
    return {"schema_version": version, "fts5": has_fts}
