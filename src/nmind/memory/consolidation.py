"""记忆巩固：把"经历过"提炼成"认为为真"，并给经验回放供样本。

这一步是记忆系统与神经网络之间唯一的正向通道，因此边界要写死：

* 只有**重复出现且反馈为正**的主张才允许升格为事实（min_evidence / min_feedback），
  单次巧合不会变成知识；
* 被用户删除过的条目（墓碑）不参与巩固 —— 删除就是删除，不会绕道学回来；
* 巩固产出的是 SQLite 里的事实行，**不会自动改动任何神经参数**；
  参数更新只能在 learning/ 里显式发生，两者指标分开报告。
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Callable

import numpy as np

from .sqlite_store import MemoryStore, content_hash


class Consolidator:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def run(self, *, min_evidence: int = 2, min_feedback: float = 0.0,
            task: str | None = None, confidence_cap: float = 0.95) -> dict[str, Any]:
        """扫描 episodes，把满足证据门槛的主张写入 facts。返回可审计的统计。"""
        q, args = "SELECT * FROM episodes", []
        if task:
            q += " WHERE task=?"
            args.append(task)
        rows = self.store.conn.execute(q + " ORDER BY id", args).fetchall()

        evidence: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(
            lambda: {"eps": [], "fb": [], "conf": [], "privacy": "public"})
        stats = {"scanned": len(rows), "no_claims": 0, "tombstoned": 0, "low_evidence": 0,
                 "low_feedback": 0}

        for r in rows:
            blob = r["observation_json"]
            claims = []
            if blob:
                try:
                    claims = json.loads(blob).get("claims") or []
                except (TypeError, ValueError):
                    claims = []
            if not claims:
                stats["no_claims"] += 1
                continue
            h = content_hash(r["session_id"], r["task"], r["observation"], r["action"], r["outcome"])
            if self.store.is_deleted("episode", h):        # 巩固阶段再核一遍墓碑，双保险
                stats["tombstoned"] += 1
                continue
            for c in claims:
                if not (isinstance(c, (list, tuple)) and len(c) == 3):
                    continue
                key = (str(c[0]), str(c[1]), str(c[2]))
                ev = evidence[key]
                ev["eps"].append(int(r["id"]))
                ev["fb"].append(float(r["feedback"] if r["feedback"] is not None else 0.0))
                ev["conf"].append(float(r["confidence"] or 0.0))
                if r["privacy_level"] in ("private", "secret"):
                    ev["privacy"] = r["privacy_level"]

        promoted: list[dict[str, Any]] = []
        for (sid, pid, obj), ev in evidence.items():
            n = len(set(ev["eps"]))
            if n < min_evidence:
                stats["low_evidence"] += 1
                continue
            mean_fb = float(np.mean(ev["fb"])) if ev["fb"] else 0.0
            if mean_fb < min_feedback:
                stats["low_feedback"] += 1
                continue
            conf = min(confidence_cap, 0.5 + 0.1 * n + 0.25 * max(0.0, mean_fb))
            fid = self.store.add_fact(
                subject=sid, predicate=pid, object=obj,
                evidence=sorted(set(ev["eps"])), confidence=conf,
                privacy_level=ev["privacy"])
            if fid > 0:
                promoted.append({"fact_id": fid, "triple": [sid, pid, obj],
                                 "evidence": n, "mean_feedback": round(mean_fb, 4),
                                 "confidence": round(conf, 4)})

        expired = self.store.expire_facts()
        return {"scanned": stats["scanned"], "promoted": len(promoted),
                "promoted_detail": promoted[:20], "skipped": {k: v for k, v in stats.items() if k != "scanned"},
                "expired_facts": expired, "gate": {"min_evidence": min_evidence,
                                                   "min_feedback": min_feedback}}

    # ------------------------------------------------------------------ 经验回放取样

    def replay_samples(self, *, task: str | None = None, limit: int = 512,
                       decoder: Callable[[Any], Any] | None = None) -> list[dict[str, Any]]:
        """为 learning/replay.py 取历史样本；已删除条目一律排除。

        decoder 用来把 observation_json 还原成训练需要的结构（由调用方提供，
        本模块不认识任何具体任务，避免把任务逻辑写进记忆层）。
        """
        q, args = "SELECT * FROM episodes", []
        if task:
            q += " WHERE task=?"
            args.append(task)
        q += " ORDER BY id DESC LIMIT ?"
        args.append(int(limit * 3))
        out: list[dict[str, Any]] = []
        for r in self.store.conn.execute(q, args).fetchall():
            h = content_hash(r["session_id"], r["task"], r["observation"], r["action"], r["outcome"])
            if self.store.is_deleted("episode", h):
                continue
            payload: dict[str, Any] = {}
            if r["observation_json"]:
                try:
                    payload = json.loads(r["observation_json"])
                except (TypeError, ValueError):
                    payload = {}
            if decoder is not None:
                payload = decoder(payload) or {}
            out.append({"id": int(r["id"]), "task": r["task"], "payload": payload,
                        "feedback": float(r["feedback"] or 0.0),
                        "confidence": float(r["confidence"] or 0.0),
                        "source": r["source"]})
            if len(out) >= limit:
                break
        return out
