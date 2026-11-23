from __future__ import annotations

import json
import sqlite3
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


class V33LearningEngine:
    """有界、可解释、持久化的非神经学习决策层。"""
    FAMILIES = ("arithmetic", "json_extract", "grid_rules")

    def __init__(self, path: str | Path, facade) -> None:
        self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row; self.facade = facade
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS v33_state(k TEXT PRIMARY KEY,v TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS v33_decisions(id TEXT PRIMARY KEY, payload TEXT NOT NULL, created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS v33_reviews(id TEXT PRIMARY KEY, payload TEXT NOT NULL, due_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS v33_failures(id TEXT PRIMARY KEY, payload TEXT NOT NULL, created_at TEXT NOT NULL);
        """); self.db.commit()
        self._setdefault("enabled", "0"); self._setdefault("state", "idle"); self._setdefault("session", "")

    def _get(self, k, default=""):
        r = self.db.execute("SELECT v FROM v33_state WHERE k=?", (k,)).fetchone(); return r[0] if r else default
    def _set(self, k, v):
        self.db.execute("INSERT OR REPLACE INTO v33_state VALUES(?,?)", (k, str(v))); self.db.commit()
    def _setdefault(self, k, v):
        if self._get(k, "") == "": self._set(k, v)
    def _json_rows(self, table, limit=50):
        return [json.loads(r[0]) for r in self.db.execute(f"SELECT payload FROM {table} ORDER BY rowid DESC LIMIT ?", (limit,))]

    def policy(self):
        return {"enabled": self._get("enabled") == "1", "state": self._get("state"), "network_enabled": False,
                "training_enabled": False, "optimizer_steps": 0, "max_tasks_per_session": 3,
                "strategy": "V33_ADAPTIVE", "cooldown_seconds": 60, "user_task_priority": True}

    def set_enabled(self, enabled):
        self._set("enabled", "1" if enabled else "0")
        if not enabled: self._set("state", "idle")
        self.facade.events.append("learning.policy_changed", self.policy()); return self.policy()

    def candidates(self):
        tracker = getattr(self.facade.runtime, "tracker", None)
        summary = tracker.family_summary() if tracker else {}
        now = datetime.now(timezone.utc)
        out = []
        for fam in self.FAMILIES:
            rec = summary.get(fam, {}); attempts = int(rec.get("attempts", 0)); fails = int(rec.get("failures", 0))
            score = (2.0 if attempts == 0 else 0.0) + fails * 2.0 + (1.0 if fam == "arithmetic" else 0.0)
            phase = "INTRO" if attempts == 0 else ("REMEDIATION" if fails else "REVIEW")
            reasons = (["CAPABILITY_UNTESTED"] if attempts == 0 else []) + (["RECENT_FAILURE"] if fails else [])
            out.append({"goal_id": f"goal-{fam}", "skill_id": fam, "task_family": fam, "source": "capability_gap" if attempts == 0 else "failure_recovery", "verification_method": "deterministic_oracle", "candidate_reason_codes": reasons or ["REVIEW_DUE"], "evidence_event_ids": [], "estimated_cost": {"max_tasks": 1, "max_seconds": 10}, "constraints": {"offline": True, "authorized": True}, "phase": phase, "priority_score": score, "created_at": now.isoformat(), "status": "ELIGIBLE"})
        return sorted(out, key=lambda x: (-x["priority_score"], x["skill_id"]))

    def choose(self):
        c = self.candidates(); chosen = c[0] if c else None
        if chosen:
            self.db.execute("INSERT INTO v33_decisions VALUES(?,?,?)", (chosen["goal_id"] + "-" + uuid.uuid4().hex[:8], json.dumps({"chosen": chosen, "candidates": c, "strategy": "V33_ADAPTIVE"}, ensure_ascii=False), datetime.now(timezone.utc).isoformat())); self.db.commit()
            self.facade.events.append("learning.goal_selected", chosen)
        return chosen

    def run_once(self):
        if not self.policy()["enabled"]: return {"status": "DISABLED", "reason": "USER_PERMISSION_REQUIRED", "optimizer_steps": 0}
        goal = self.choose()
        if not goal: self._set("state", "idle"); return {"status": "IDLE_NO_VERIFIABLE_GOAL", "optimizer_steps": 0}
        self._set("state", "running"); started = time.perf_counter()
        from ..bot.practice import make_task, execute_tool, verify, feedback_state
        import random
        task = make_task(goal["task_family"], random.Random(goal["goal_id"] + str(time.time_ns())), index=1)
        attempt = execute_tool(task); verdict = verify(task, attempt); verified = verdict == "verified"
        evidence = self.facade.events.append("learning.oracle_verified", {"task_id": task["task_id"], "verdict": verdict, "source": "tool/oracle"})
        tracker = getattr(self.facade.runtime, "tracker", None); capability = None
        if tracker:
            capability = tracker.record_attempt(capability_id=goal["skill_id"] + ":tool", task_family=goal["task_family"], success=verified, path="tool", evidence_id=evidence.event_id, tool_calls=1, response_source="tool", feedback_source=feedback_state(task, verdict), learning_target="memory_only")
        result = {"status": "completed", "goal": goal, "task_id": task["task_id"], "answer_source": "tool", "verification_source": "deterministic_oracle", "oracle_result": verdict, "attempt": attempt, "failure_reason": None if verified else "wrong_value", "duration_ms": round((time.perf_counter()-started)*1000), "optimizer_steps": 0, "network_requests": 0, "capability": capability}
        if not verified: self.db.execute("INSERT INTO v33_failures VALUES(?,?,?)", (uuid.uuid4().hex, json.dumps(result, ensure_ascii=False), datetime.now(timezone.utc).isoformat()))
        due = datetime.now(timezone.utc) + timedelta(seconds=60 if verified else 10)
        self.db.execute("INSERT INTO v33_reviews VALUES(?,?,?)", (uuid.uuid4().hex, json.dumps({"goal": goal, "task_id": task["task_id"], "status": "DUE"}, ensure_ascii=False), due.isoformat())); self.db.commit()
        self._set("state", "idle"); self.facade.events.append("learning.cycle_completed", result)
        try: result["experience"] = self.facade.v34_experience_from_cycle(result)
        except Exception: result["experience"] = {"status": "NOT_RECORDED"}
        return result

    def pause(self): self._set("state", "paused"); self.facade.events.append("learning.session_paused", {}); return self.policy()
    def resume(self): self._set("state", "idle"); self.facade.events.append("learning.session_resumed", {}); return self.policy()
    def stop(self): self._set("state", "idle"); self._set("session", ""); self.facade.events.append("learning.session_stopped", {}); return self.policy()
    def status(self): return {**self.policy(), "goals": self.candidates(), "decision": self._json_rows("v33_decisions", 1), "reviews_due": len(self.db.execute("SELECT 1 FROM v33_reviews WHERE due_at<=?", (datetime.now(timezone.utc).isoformat(),)).fetchall())}
    def reviews(self): return self._json_rows("v33_reviews")
    def failures(self): return self._json_rows("v33_failures")
    def decisions(self): return self._json_rows("v33_decisions")
    def close(self): self.db.close()
