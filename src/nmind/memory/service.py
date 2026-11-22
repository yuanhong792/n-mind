"""Memory V2 facade: durable recall/observe/feedback without neural training."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .consolidation import Consolidator
from .retriever import Hit, MemoryRetriever
from .sqlite_store import MemoryStore, content_hash

_SENSITIVE = re.compile(r"(?:password|token|secret|api[_ -]?key|密码|口令|令牌|私钥)", re.I)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class MemoryQuery:
    text: str
    session_id: str | None = None
    task_id: str | None = None
    limit: int = 6
    memory_types: tuple[str, ...] = ()


@dataclass
class MemoryContext:
    query: MemoryQuery
    hits: list[dict[str, Any]] = field(default_factory=list)
    used: bool = False
    explanation: str = ""

    @property
    def text(self) -> str:
        return "\n".join(h["text"] for h in self.hits)


class MemoryService:
    """Stable public API over the existing store/retriever/consolidator."""

    def __init__(self, store: MemoryStore, *, retriever: MemoryRetriever | None = None) -> None:
        self.store = store
        self.retriever = retriever or MemoryRetriever(store, top_k=8)
        self.consolidator = Consolidator(store)

    def _meta(self, kind: str, memory_id: int) -> dict[str, Any]:
        row = self.store.conn.execute("SELECT * FROM memory_meta WHERE kind=? AND memory_id=?", (kind, memory_id)).fetchone()
        return dict(row) if row else {"memory_kind": "episodic", "importance": 0.5, "trust": 0.5, "verification_state": "unverified"}

    def _put_meta(self, kind: str, memory_id: int, *, memory_kind: str, importance: float,
                  retention: str, verification_state: str, source_uri: str, trust: float,
                  expires_at: str | None, tags: list[str], explanation: str) -> None:
        self.store.conn.execute(
            """INSERT INTO memory_meta(kind,memory_id,memory_kind,importance,retention,verification_state,
               source_uri,trust,observed_at,expires_at,tags_json,explanation)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(kind,memory_id) DO UPDATE SET importance=excluded.importance,
               retention=excluded.retention,verification_state=excluded.verification_state,
               source_uri=excluded.source_uri,trust=excluded.trust,expires_at=excluded.expires_at,
               tags_json=excluded.tags_json,explanation=excluded.explanation""",
            (kind, memory_id, memory_kind, max(0.0, min(1.0, importance)), retention,
             verification_state, source_uri, max(0.0, min(1.0, trust)), _now(), expires_at,
             json.dumps(tags, ensure_ascii=False), explanation))

    def observe(self, *, session_id: str, observation: str, task: str = "", action: str | None = None,
                outcome: str | None = None, feedback: float | None = None, source: str = "local",
                confidence: float = 0.5, memory_kind: str = "episodic", importance: float | None = None,
                source_uri: str = "", trust: float = 0.5, verification_state: str = "unverified",
                privacy_level: str = "normal", task_id: str | None = None, tags: list[str] | None = None,
                claims: list[list[str]] | None = None) -> int:
        if _SENSITIVE.search(observation) or _SENSITIVE.search(action or ""):
            return -1
        payload = {"memory_kind": memory_kind, "task_id": task_id or "", "claims": claims or [], "tags": tags or []}
        eid = self.store.add_episode(session_id=session_id, observation=observation, task=task,
                                     action=action, outcome=outcome, feedback=feedback, source=source,
                                     confidence=confidence, privacy_level=privacy_level,
                                     observation_json=payload)
        if eid < 0:
            return eid
        score = importance if importance is not None else min(1.0, 0.35 + 0.25 * confidence + (0.2 if feedback and feedback > 0 else 0.0))
        self._put_meta("episode", eid, memory_kind=memory_kind, importance=score, retention="standard",
                       verification_state=verification_state, source_uri=source_uri, trust=trust,
                       expires_at=None, tags=tags or [], explanation="按来源、反馈、可信度和复用价值评分")
        return eid

    def recall(self, query: MemoryQuery | str, *, explain: bool = True) -> MemoryContext:
        q = query if isinstance(query, MemoryQuery) else MemoryQuery(query)
        raw = self.retriever.search(q.text, kind="both", limit=max(12, q.limit * 3), session_id=q.session_id)
        # FTS unicode61 is intentionally conservative for CJK.  A bounded
        # character-overlap fallback keeps Chinese recall useful without an
        # embedding service or an unbounded table scan.
        if not raw:
            qchars = {c for c in q.text if not c.isspace() and len(c) > 0}
            for ep in self.store.recent_episodes(limit=80):
                overlap = len(qchars & set(ep.observation))
                if overlap >= 2:
                    raw.append(Hit("episode", ep.id, f"{ep.observation} {ep.outcome or ''}".strip(),
                                   min(1.0, overlap / max(4, len(qchars))), ep.confidence,
                                   ep.privacy_level, {"task": ep.task, "source": ep.source}))
        rows: list[dict[str, Any]] = []
        for hit in raw:
            meta = self._meta(hit.kind, hit.id)
            if meta.get("deleted_at") or (q.memory_types and meta.get("memory_kind") not in q.memory_types):
                continue
            score = float(hit.score) * (0.55 + 0.45 * float(meta.get("importance", 0.5))) * (0.65 + 0.35 * float(meta.get("trust", 0.5)))
            rows.append({"kind": hit.kind, "id": hit.id, "text": hit.text, "score": round(score, 6),
                         "confidence": hit.confidence, "memory_kind": meta.get("memory_kind", "episodic"),
                         "source_uri": meta.get("source_uri", ""), "verification_state": meta.get("verification_state", "unverified"),
                         "importance": meta.get("importance", 0.5), "why": meta.get("explanation", "")})
        rows.sort(key=lambda x: x["score"], reverse=True)
        rows = rows[: max(0, q.limit)]
        return MemoryContext(q, rows, bool(rows), "命中后按相关性、重要性、可信度重排" if rows else "未找到可用长期记忆")

    def feedback(self, *, kind: str, memory_id: int, value: float, task_id: str = "", source: str = "user", note: str = "") -> None:
        self.store.conn.execute("INSERT INTO memory_feedback(kind,memory_id,task_id,value,source,created_at,note) VALUES(?,?,?,?,?,?,?)",
                                (kind, memory_id, task_id, float(value), source, _now(), note))
        row = self._meta(kind, memory_id)
        self._put_meta(kind, memory_id, memory_kind=row.get("memory_kind", "episodic"),
                       importance=min(1.0, max(0.0, float(row.get("importance", .5)) + .1 * float(value))),
                       retention=row.get("retention", "standard"), verification_state="verified" if value > 0 else row.get("verification_state", "unverified"),
                       source_uri=row.get("source_uri", ""), trust=row.get("trust", .5), expires_at=row.get("expires_at"), tags=[], explanation=row.get("explanation", ""))

    def consolidate(self, **kwargs: Any) -> dict[str, Any]:
        return self.consolidator.run(**kwargs)

    def forget(self, *, kind: str, memory_id: int, reason: str = "user") -> bool:
        ok = self.store.delete_episode(memory_id, reason=reason) if kind == "episode" else self.store.delete_fact(memory_id, reason=reason)
        if ok:
            self.store.conn.execute("UPDATE memory_meta SET deleted_at=? WHERE kind=? AND memory_id=?", (_now(), kind, memory_id))
        return ok

    def stats(self) -> dict[str, Any]:
        return {**self.store.stats(), "memory_meta": int(self.store.conn.execute("SELECT COUNT(*) FROM memory_meta WHERE deleted_at IS NULL").fetchone()[0])}
