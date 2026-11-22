"""中文知识检索：FTS5 BM25（bigram 切分）+ LIKE 关键词兜底。

返回结构保证"答案引用能定位到库里的 chunk"：每条结果带 chunk_id/doc_id/url/
title/fetched_at/原文摘录——GUI 与聊天服务据此渲染引用，绝不编造出处。
"""

from __future__ import annotations

import time
from typing import Any

from .knowledge_store import KnowledgeStore, bigram_tokens


class KnowledgeSearch:
    def __init__(self, store: KnowledgeStore) -> None:
        self.store = store

    def search(self, query: str, *, limit: int = 5) -> list[dict[str, Any]]:
        q = (query or "").strip()
        if not q:
            return []
        results = self._search_fts(q, limit) if self.store.fts_enabled else []
        if not results:
            results = self._search_like(q, limit)
        return results

    def _search_fts(self, query: str, limit: int) -> list[dict[str, Any]]:
        tokens = bigram_tokens(query)
        if not tokens:
            return []
        try:
            match_expr = " OR ".join(f'"{t}"' for t in tokens.split())
            rows = self.store.conn.execute(
                "SELECT f.chunk_id, bm25(chunks_fts) AS rank FROM chunks_fts f "
                "WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?",
                (match_expr, max(limit * 8, 24))).fetchall()
        except Exception:                            # noqa: BLE001 - MATCH 语法问题退回 LIKE
            return []
        scored = []
        for r in rows:
            chunk = self.store.chunk(r["chunk_id"])
            if chunk and chunk.get("doc_status") == "READY":
                text = str(chunk.get("text") or "").lower()
                coverage = sum(text.count(tok.lower()) for tok in tokens.split())
                scored.append((coverage, _bm25_to_relevance(r["rank"]),
                               {"citation": self._citation(chunk),
                                "score": _bm25_to_relevance(r["rank"]) + coverage}))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [item[2] for item in scored[:limit]]

    def _search_like(self, query: str, limit: int) -> list[dict[str, Any]]:
        """离线兜底：取查询里的关键词（汉字串/ASCII 词）做 LIKE 扫描。"""
        kws = _keywords(query)
        if not kws:
            return []
        scored: dict[str, int] = {}
        for ck in self.store.conn.execute(
                "SELECT c.*, d.canonical_url AS url, d.title, d.fetched_at_utc, "
                "d.trust_state, d.status AS doc_status FROM chunks c "
                "JOIN documents d ON d.doc_id=c.doc_id WHERE d.latest=1").fetchall():
            text = ck["text"]
            hits = sum(1 for k in kws if k in text)
            if hits:
                scored[ck["chunk_id"]] = hits
        ranked = sorted(scored.items(), key=lambda kv: -kv[1])[:limit]
        out = []
        for chunk_id, hits in ranked:
            chunk = self.store.chunk(chunk_id)
            if chunk and chunk.get("doc_status") == "READY":
                out.append({"citation": self._citation(chunk), "score": float(hits)})
        return out

    @staticmethod
    def _citation(chunk: dict[str, Any]) -> dict[str, Any]:
        stale = chunk.get("doc_status") != "READY"
        return {"doc_id": chunk["doc_id"], "chunk_id": chunk["chunk_id"],
                "title": chunk.get("title") or "未命名", "url": chunk.get("url") or "",
                "fetched_at_utc": chunk.get("fetched_at_utc") or "",
                "excerpt": chunk["text"], "trust_state": chunk.get("trust_state") or "",
                "stale": stale}

    def stats(self) -> dict[str, int]:
        docs = int(self.store.conn.execute(
            "SELECT COUNT(*) AS c FROM documents WHERE latest=1").fetchone()["c"])
        all_docs = int(self.store.conn.execute(
            "SELECT COUNT(*) AS c FROM documents").fetchone()["c"])
        chunks = int(self.store.conn.execute(
            "SELECT COUNT(*) AS c FROM chunks").fetchone()["c"])
        hosts = int(self.store.conn.execute(
            "SELECT COUNT(DISTINCT source_host) AS c FROM documents").fetchone()["c"])
        return {"documents": docs, "all_versions": all_docs, "chunks": chunks,
                "hosts": hosts,
                "deduplicated": all_docs - docs}


def _bm25_to_relevance(rank: float) -> float:
    """bm25(chunks_fts) 返回负值（越小越相关）→ 转成正的相关度展示值。"""
    try:
        return round(max(0.0, -float(rank)), 3)
    except (TypeError, ValueError):
        return 0.0


def _keywords(query: str) -> list[str]:
    """查询关键词提取：连续汉字串与 ASCII 词；太短的串跳过（避免全表噪声）。"""
    import re
    out = list(re.findall(r"[一-鿿]{2,}|[A-Za-z0-9][A-Za-z0-9_-]*", query))
    return out[:8]



def build_chat_citations(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把检索结果裁剪成聊天引用（控制摘录长度，保完整溯源字段）。"""
    out = []
    for r in results:
        c = dict(r["citation"])
        c["excerpt"] = c["excerpt"][:160]
        out.append(c)
    return out


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")
