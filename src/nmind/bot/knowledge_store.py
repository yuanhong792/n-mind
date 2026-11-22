"""中文知识库持久层：SQLite WAL + FTS5 索引 + 来源指纹 + 版本去重。

与 BotStore（bot_state.db）分离为独立 knowledge.db：职责是网页/本地资料的
知识链（内容+指纹+时间+片段索引），不掺权限（权限链在 SourceRegistry）。
迁移策略：schema_version 表记录版本，升版只加表/列，绝不丢弃旧数据。

FTS5 中文策略（任务书 §5.1：不能用英文默认切词糊弄中文）：
  * 入索引前对正文做汉字双字 bigram 切分（ASCII 词保留原样）；
  * 查询词同样切分后进 FTS MATCH；BM25 rank 排序；
  * FTS5 不可用或零命中时退回 LIKE 关键词扫描（离线兜底，可运行实现）。
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

_SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY, value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,             -- sha256(canonical_url + content_sha256)：内容版本级唯一
    canonical_url TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    fetched_at_utc TEXT NOT NULL,
    extracted_at_utc TEXT,
    content_sha256 TEXT NOT NULL,
    mime TEXT DEFAULT 'text/html',
    http_status INTEGER,
    trust_state TEXT NOT NULL,           -- fixture | public-read-only
    status TEXT NOT NULL,                -- READY | EXTRACT_FAILED
    extractor_version TEXT NOT NULL,
    raw_len INTEGER DEFAULT 0,
    latest INTEGER DEFAULT 1,            -- 同 URL 多版本时只有最新版为 1
    source_host TEXT DEFAULT '',
    instruction_detected INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_docs_url ON documents(canonical_url, latest);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL REFERENCES documents(doc_id),
    ordinal INTEGER NOT NULL,
    text TEXT NOT NULL,
    char_start INTEGER NOT NULL,
    char_end INTEGER NOT NULL,
    text_sha256 TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id, ordinal);
CREATE TABLE IF NOT EXISTS fetch_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    requested_url TEXT NOT NULL,
    final_url TEXT,
    status TEXT NOT NULL,
    reason TEXT,
    bytes INTEGER DEFAULT 0,
    elapsed_ms REAL DEFAULT 0,
    policy_hash TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS explore_jobs (
    job_id TEXT PRIMARY KEY,
    source_host TEXT NOT NULL,
    url TEXT NOT NULL,
    depth INTEGER DEFAULT 0,
    state TEXT DEFAULT 'PENDING',        -- PENDING | DONE | FAILED | COOLDOWN
    next_eligible_at REAL DEFAULT 0,
    attempts INTEGER DEFAULT 0,
    reason TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_state ON explore_jobs(state, next_eligible_at);
"""


class KnowledgeStore:
    def __init__(self, db_path: str | Path) -> None:
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), isolation_level=None,
                                    check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA synchronous = FULL")
        self.conn.execute("PRAGMA busy_timeout = 5000")
        self.conn.executescript(_SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        row = self.conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        ver = int(row["value"]) if row else 0
        if ver > _SCHEMA_VERSION:
            raise RuntimeError(
                f"knowledge.db schema 版本 {ver} 高于本代码支持的 {_SCHEMA_VERSION}，"
                "拒绝打开（防止降级写入破坏新结构）")
        if ver < _SCHEMA_VERSION:
            self.conn.execute(
                "INSERT INTO meta (key, value) VALUES ('schema_version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(_SCHEMA_VERSION),))
        # FTS5 建表失败不阻塞打开（检索层会退回 LIKE 兜底）
        try:
            self.conn.executescript(
                "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5"
                "(chunk_id UNINDEXED, tokens);")
            self._fts_enabled = True
        except sqlite3.OperationalError:
            self._fts_enabled = False

    @property
    def fts_enabled(self) -> bool:
        return bool(getattr(self, "_fts_enabled", False))

    def close(self) -> None:
        self.conn.close()

    # ---------------------------------------------------------------- 记录抓取
    def record_fetch_event(self, requested_url: str, final_url: str | None,
                           status: str, reason: str | None = None, *,
                           bytes_read: int = 0, elapsed_ms: float = 0.0,
                           policy_hash: str = "") -> int:
        cur = self.conn.execute(
            "INSERT INTO fetch_events (ts, requested_url, final_url, status, reason, "
            "bytes, elapsed_ms, policy_hash) VALUES (?,?,?,?,?,?,?,?)",
            (time.strftime("%Y-%m-%dT%H:%M:%S%z"), requested_url, final_url, status,
             reason, bytes_read, elapsed_ms, policy_hash))
        return int(cur.lastrowid)

    def fetch_events(self, *, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM fetch_events ORDER BY event_id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    # ---------------------------------------------------------------- 文档入库
    def has_content(self, content_sha256: str) -> bool:
        return self.conn.execute(
            "SELECT 1 FROM documents WHERE content_sha256=? LIMIT 1",
            (content_sha256,)).fetchone() is not None

    def find_document(self, canonical_url: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM documents WHERE canonical_url=? AND latest=1",
            (canonical_url,)).fetchone()
        return dict(row) if row else None

    def insert_document(self, *, canonical_url: str, title: str, fetched_at_utc: str,
                        content_sha256: str, trust_state: str, extractor_version: str,
                        http_status: int | None = None, mime: str = "text/html",
                        raw_len: int = 0, source_host: str = "",
                        instruction_detected: bool = False,
                        status: str = "READY") -> str:
        """内容级去重：同内容 sha 已存在 → 返回现有 doc_id 并计 deduplicated。"""
        existing = self.conn.execute(
            "SELECT doc_id FROM documents WHERE content_sha256=? LIMIT 1",
            (content_sha256,)).fetchone()
        if existing:
            return existing["doc_id"]
        doc_id = hashlib.sha256(
            f"{canonical_url}|{content_sha256}".encode()).hexdigest()[:24]
        # 同 URL 新内容版本：旧版 latest=0（保留获取时间与原文，不覆盖历史）
        self.conn.execute(
            "UPDATE documents SET latest=0 WHERE canonical_url=? AND latest=1",
            (canonical_url,))
        self.conn.execute(
            "INSERT INTO documents (doc_id, canonical_url, title, fetched_at_utc, "
            "extracted_at_utc, content_sha256, mime, http_status, trust_state, status, "
            "extractor_version, raw_len, latest, source_host, instruction_detected) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,1,?,?)",
            (doc_id, canonical_url, title, fetched_at_utc,
             time.strftime("%Y-%m-%dT%H:%M:%S%z"), content_sha256, mime, http_status,
             trust_state, status, extractor_version, raw_len, source_host,
             int(instruction_detected)))
        return doc_id

    def insert_chunks(self, doc_id: str, chunks: list[dict[str, Any]]) -> int:
        n = 0
        for c in chunks:
            chunk_id = hashlib.sha256(
                f"{doc_id}|{c['ordinal']}|{c['text_sha256']}".encode()).hexdigest()[:24]
            self.conn.execute(
                "INSERT OR IGNORE INTO chunks (chunk_id, doc_id, ordinal, text, "
                "char_start, char_end, text_sha256) VALUES (?,?,?,?,?,?,?)",
                (chunk_id, doc_id, c["ordinal"], c["text"], c["char_start"],
                 c["char_end"], c["text_sha256"]))
            if self.fts_enabled:
                self.conn.execute(
                    "INSERT OR IGNORE INTO chunks_fts (chunk_id, tokens) VALUES (?,?)",
                    (chunk_id, bigram_tokens(c["text"])))
            n += 1
        return n

    def chunk(self, chunk_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT c.*, d.canonical_url AS url, d.title, d.fetched_at_utc, "
            "d.trust_state, d.status AS doc_status FROM chunks c "
            "JOIN documents d ON d.doc_id=c.doc_id WHERE c.chunk_id=?",
            (chunk_id,)).fetchone()
        return dict(row) if row else None

    def chunks_of(self, doc_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT c.*, d.canonical_url AS url, d.title, d.fetched_at_utc, d.trust_state "
            "FROM chunks c JOIN documents d ON d.doc_id=c.doc_id "
            "WHERE c.doc_id=? ORDER BY c.ordinal", (doc_id,)).fetchall()
        return [dict(r) for r in rows]

    # ---------------------------------------------------------------- 探索任务
    def upsert_job(self, url: str, source_host: str, *, depth: int = 0,
                   state: str = "PENDING", next_eligible_at: float = 0.0,
                   reason: str = "") -> str:
        job_id = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
        self.conn.execute(
            "INSERT INTO explore_jobs (job_id, source_host, url, depth, state, "
            "next_eligible_at, attempts, reason) VALUES (?,?,?,?,?,?,0,?) "
            "ON CONFLICT(job_id) DO UPDATE SET state=excluded.state, "
            "next_eligible_at=excluded.next_eligible_at, reason=excluded.reason",
            (job_id, source_host, url, depth, state, next_eligible_at, reason))
        return job_id

    def bump_job(self, job_id: str, *, state: str, reason: str = "",
                 next_eligible_at: float | None = None) -> None:
        self.conn.execute(
            "UPDATE explore_jobs SET attempts=attempts+1, state=?, reason=?, "
            "next_eligible_at=COALESCE(?, next_eligible_at) WHERE job_id=?",
            (state, reason, next_eligible_at, job_id))

    def next_jobs(self, *, limit: int = 5, now: float | None = None) -> list[dict[str, Any]]:
        now = time.time() if now is None else now
        rows = self.conn.execute(
            "SELECT * FROM explore_jobs WHERE state IN ('PENDING','COOLDOWN') AND "
            "next_eligible_at<=? ORDER BY depth, next_eligible_at LIMIT ?",
            (now, limit)).fetchall()
        return [dict(r) for r in rows]

    def job(self, job_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM explore_jobs WHERE job_id=?",
                                (job_id,)).fetchone()
        return dict(row) if row else None


# ---------------------------------------------------------------- 中文 bigram
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")


def bigram_tokens(text: str) -> str:
    """汉字双字切分 + ASCII 词保留：给 FTS5 提供可检索的中文 token 流。"""
    out: list[str] = []
    buf_ascii = ""
    chars = list(text)

    def flush_ascii():
        nonlocal buf_ascii
        if buf_ascii:
            out.append(buf_ascii.lower())
            buf_ascii = ""

    i = 0
    n = len(chars)
    while i < n:
        ch = chars[i]
        if _CJK_RE.match(ch):
            flush_ascii()
            if i + 1 < n and _CJK_RE.match(chars[i + 1]):
                out.append(ch + chars[i + 1])
            else:
                out.append(ch)
            i += 1
        elif ch.isalnum() and ch.isascii():
            buf_ascii += ch
            i += 1
        else:
            flush_ascii()
            i += 1
    flush_ascii()
    return " ".join(out[:4000])                  # 上限防超长文本撑爆索引
