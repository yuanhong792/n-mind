from __future__ import annotations
import random, time, uuid
from pathlib import Path
from typing import Any
from .events import EventLog
from .tasks import TaskService

class V3Facade:
    """Single local-first orchestration boundary for V3.1."""
    def __init__(self, root: str | Path, *, chat=None, memory=None, knowledge=None, runtime=None):
        root = Path(root); root.mkdir(parents=True, exist_ok=True)
        self.root, self.chat, self.memory, self.knowledge, self.runtime = root, chat, memory, knowledge, runtime
        self.events = EventLog(root / "v3_events.db"); self.tasks = TaskService(root / "v3_tasks.db", self.events)
        self.learning_enabled = False; self.learning_state = "idle"; self.cycles: list[dict[str, Any]] = []
    def status(self) -> dict[str, Any]:
        return {"version": "V3.1", "network_enabled": False, "training_enabled": False, "optimizer_steps": 0, "runtime": self.runtime.status() if self.runtime is not None else {}, "memory": self.memory.stats() if self.memory else {"status": "DISABLED"}, "knowledge": self.knowledge.stats() if self.knowledge else {"status": "DISABLED"}, "active_tasks": self.tasks.list(20), "learning_enabled": self.learning_enabled, "learning_state": self.learning_state}
    def chat_reply(self, session_id: str, text: str) -> dict[str, Any]:
        if self.chat is None: return {"text": "没有可用的本地聊天服务。", "response_source": "none", "citations": [], "status": "NO_SERVICE"}
        from ..bot.chat_service import ChatRequest
        out = self.chat.reply(ChatRequest.new(session_id, text)).to_dict(); out["v3_attribution"] = {"source": out.get("response_source", "none"), "neural_claim": out.get("response_source") == "neural"}; self.events.append("chat.replied", {"session_id": session_id, "source": out.get("response_source"), "citation_count": len(out.get("citations", []))}); return out
    def learn_once(self) -> dict[str, Any]:
        started = time.perf_counter(); cid = uuid.uuid4().hex; self.learning_state = "running"
        from ..bot.practice import make_task, execute_tool, verify
        task = make_task("arithmetic", random.Random(cid), index=1); verdict = verify(task, execute_tool(task))
        evidence_id = self.events.append("learning.oracle_verified", {"task_id": task["task_id"], "verdict": verdict}).event_id
        tracker = getattr(self.runtime, "tracker", None)
        capability = tracker.record_attempt(capability_id="arithmetic:tool", task_family="arithmetic", success=verdict == "verified", path="tool", evidence_id=evidence_id, tool_calls=1, response_source="tool", feedback_source="deterministic_oracle", learning_target="memory_only") if tracker else None
        out = {"cycle_id": cid, "goal_id": "goal-arithmetic", "skill_id": "arithmetic", "state": "completed", "task_id": task["task_id"], "neural_attempt": None, "neural_verdict": None, "oracle_result": verdict, "verified_experience_id": evidence_id, "capability": capability, "optimizer_steps": 0, "network_requests": 0, "source": "tool/oracle", "duration_ms": round((time.perf_counter()-started)*1000)}
        self.cycles.append(out); self.learning_state = "idle"; self.events.append("learning.cycle", out); return out
    def set_learning(self, enabled: bool) -> dict[str, Any]:
        self.learning_enabled = bool(enabled); out = {"enabled": self.learning_enabled, "state": self.learning_state}; self.events.append("learning.permission_changed", out); return out
    def learning_cycles(self) -> list[dict[str, Any]]: return list(reversed(self.cycles[-50:]))
    def memory_create(self, body: dict[str, Any]) -> dict[str, Any]:
        if self.memory is None: return {"status": "DISABLED"}
        eid = self.memory.observe(session_id=str(body.get("session_id") or "v3.1"), observation=str(body.get("text") or "")[:2000], task="user_asserted", source="user", confidence=1.0, importance=float(body.get("importance", .7)), source_uri="user://chat", verification_state="user_asserted", memory_kind=str(body.get("memory_kind") or "fact")); out = {"memory_id": eid, "status": "SAVED" if eid >= 0 else "REJECTED"}; self.events.append("memory.created", out); return out
    def memory_delete(self, memory_id: int) -> dict[str, Any]:
        ok = bool(self.memory and self.memory.forget(kind="episode", memory_id=memory_id)); out = {"memory_id": memory_id, "deleted": ok}; self.events.append("memory.deleted", out); return out
    def memory_correct(self, memory_id: int, body: dict[str, Any]) -> dict[str, Any]:
        old = self.memory_delete(memory_id); new = self.memory_create(body); out = {"old": old, "new": new, "status": "CORRECTED" if new.get("memory_id", -1) >= 0 else "FAILED"}; self.events.append("memory.corrected", out); return out
    def growth(self) -> dict[str, Any]:
        tracker = getattr(self.runtime, "tracker", None); return {"capabilities": tracker.summary() if tracker else {}, "learning_cycles": len(self.cycles), "sources": {"neural": 0, "tool": sum(c.get("source") == "tool/oracle" for c in self.cycles), "memory": 0, "retrieval": 0}, "optimizer_steps": 0}
    def permissions(self) -> dict[str, Any]: return {"network_enabled": False, "training_enabled": False, "optimizer_steps": 0, "autonomous_learning": self.learning_enabled, "external_llm": False}
    def close(self) -> None: self.tasks.close(); self.events.close()
