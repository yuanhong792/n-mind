"""记忆检索：优先 SQLite FTS5（BM25），编译缺失时退回 LIKE，功能不断崖。

这里刻意**不引入向量数据库/联网 embedding 服务**。指令第 2.C 节要求先用
SQLite FTS5/BM25 或简单索引；向量检索留作可选优化（本模块提供
`vector_context()`，把命中条目的数值字段编码成固定维张量喂给网络，
这是本地可算的，不依赖外部模型）。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from .sqlite_store import Episode, Fact, MemoryStore, content_hash


@dataclass
class Hit:
    kind: str            # 'episode' | 'fact'
    id: int
    text: str
    score: float
    confidence: float
    privacy_level: str
    meta: dict[str, Any]


def _fts_query(raw: str) -> str:
    """把用户输入变成 FTS5 查询串：只保留词，加前缀匹配；引号与运算符一律剥掉。

    用户文本是不可信数据（指令第 6 节），绝不能当成查询语法执行。
    """
    toks = re.findall(r"[\w一-鿿]{1,32}", raw or "")
    if not toks:
        return ""
    return " OR ".join(f'"{t}"*' for t in toks[:16])


class MemoryRetriever:
    def __init__(self, store: MemoryStore, *, top_k: int = 5) -> None:
        self.store = store
        self.top_k = int(top_k)
        self.mode = "fts5" if store.capabilities.get("fts5") else "like"

    # ------------------------------------------------------------------ 文本检索

    def search(self, query: str, *, kind: str = "both", limit: int | None = None,
               session_id: str | None = None, task: str | None = None) -> list[Hit]:
        limit = int(limit or self.top_k)
        q = (query or "").strip()
        if not q:
            return []
        hits: list[Hit] = []
        if kind in ("episode", "both"):
            hits += self._search_episodes(q, limit, session_id, task)
        if kind in ("fact", "both"):
            hits += self._search_facts(q, limit)
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[: limit * 2]

    def _search_episodes(self, q: str, limit: int, session_id: str | None, task: str | None) -> list[Hit]:
        fts = _fts_query(q)
        out: list[Hit] = []
        if fts and self.mode == "fts5":
            sql = (
                "SELECT e.*, bm25(episodes_fts) AS rank FROM episodes_fts "
                "JOIN episodes e ON e.id = episodes_fts.rowid "
                "WHERE episodes_fts MATCH ?"
            )
            args: list[Any] = [fts]
            if session_id:
                sql += " AND e.session_id=?"
                args.append(session_id)
            if task:
                sql += " AND e.task=?"
                args.append(task)
            sql += " ORDER BY rank LIMIT ?"     # bm25() 越小越相关
            args.append(limit)
            try:
                for row in self.store.conn.execute(sql, args).fetchall():
                    text = f"{row['observation']} {row['outcome'] or ''}".strip()
                    out.append(Hit("episode", row["id"], text, _rank_to_score(row["rank"]),
                                   float(row["confidence"] or 0.0), row["privacy_level"],
                                   {"task": row["task"], "action": row["action"], "source": row["source"]}))
            except Exception:                   # noqa: BLE001 - 查询语法异常一律降级，不能崩
                self.mode = "like"
        if not out:
            out = self._search_episodes_like(q, limit, session_id, task)
        return out

    def _search_episodes_like(self, q: str, limit: int, session_id: str | None, task: str | None) -> list[Hit]:
        toks = re.findall(r"[\w一-鿿]{1,32}", q)[:8]
        if not toks:
            return []
        where = " OR ".join(["observation LIKE ?"] * len(toks))
        args = [f"%{t}%" for t in toks]
        sql = f"SELECT * FROM episodes WHERE ({where}) AND source <> 'superseded'"
        if session_id:
            sql += " AND session_id=?"
            args.append(session_id)
        if task:
            sql += " AND task=?"
            args.append(task)
        sql += " ORDER BY id DESC LIMIT ?"
        args.append(limit)
        rows = self.store.conn.execute(sql, args).fetchall()
        return [Hit("episode", r["id"], f"{r['observation']} {r['outcome'] or ''}".strip(),
                    0.5, float(r["confidence"] or 0.0), r["privacy_level"],
                    {"task": r["task"], "matcher": "like"}) for r in rows]

    def _search_facts(self, q: str, limit: int) -> list[Hit]:
        toks = re.findall(r"[\w一-鿿]{1,32}", q)[:8]
        if not toks:
            return []
        args: list[Any] = []
        where = []
        for t in toks:
            where.append("(subject LIKE ? OR predicate LIKE ? OR object LIKE ?)")
            args += [f"%{t}%"] * 3
        rows = self.store.conn.execute(
            f"SELECT * FROM facts WHERE status IN ('active','candidate') AND ({' OR '.join(where)}) "
            "ORDER BY confidence DESC, updated_at DESC LIMIT ?", (*args, limit)).fetchall()
        return [Hit("fact", r["id"], f"{r['subject']} {r['predicate']} {r['object']}",
                    0.6 + 0.4 * float(r["confidence"] or 0.0), float(r["confidence"] or 0.0),
                    r["privacy_level"], {"status": r["status"]}) for r in rows]

    # ------------------------------------------------------------------ 给网络用的张量上下文

    def vector_context(self, query: str, *, dim: int = 16, k: int = 4,
                       session_id: str | None = None) -> np.ndarray:
        """把命中的记忆压成一个定长向量，作为 NModel.step(mem=...) 的输入。

        编码方式是**确定的**：命中条目的数值字段做槽位累加，外加命中数与置信度。
        不是训练出来的 embedding —— 别把它说成语义表征。
        """
        vec = np.zeros(dim, dtype=np.float32)
        hits = self.search(query, limit=k, session_id=session_id)
        for i, h in enumerate(hits[:k]):
            vec[i % max(1, dim - 2)] += h.score * (1.0 + h.confidence)
            nums = re.findall(r"-?\d+(?:\.\d+)?", h.text)[:4]
            for v in nums:
                vec[2 + (i % max(1, dim - 2))] += float(v) / 100.0
        if hits:
            vec[dim - 2] = float(len(hits))
            vec[dim - 1] = float(np.mean([h.confidence for h in hits]))
        return vec

    def context_text(self, hits: list[Hit], *, max_chars: int = 400) -> str:
        return " | ".join(h.text for h in hits)[:max_chars]

    # ------------------------------------------------------------------ 供策略/脱敏判断

    def privacy_snapshot(self, hits: list[Hit]) -> dict[str, Any]:
        """命中记忆里的隐私构成（云端策略要用它，但不能把正文带出去）。"""
        levels: dict[str, int] = {}
        for h in hits:
            levels[h.privacy_level] = levels.get(h.privacy_level, 0) + 1
        return {"levels": levels, "count": len(hits),
                "hashes": [content_hash(h.kind, h.id) for h in hits]}


def _rank_to_score(rank: float) -> float:
    """bm25 输出是负的相似度度量（越小越好）；映射到 (0,1] 便于横向比较。"""
    return float(1.0 / (1.0 + max(0.0, float(rank))))


def episode_to_dict(ep: Episode) -> dict[str, Any]:
    return json.loads(json.dumps(ep.__dict__, ensure_ascii=False, default=str))
