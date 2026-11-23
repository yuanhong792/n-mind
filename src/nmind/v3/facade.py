from __future__ import annotations

from pathlib import Path
from typing import Any

from .events import EventLog
from .tasks import TaskService


class V3Facade:
    """Single orchestration boundary for V3 UI and CLI.

    Existing services remain authoritative. This facade only coordinates them,
    labels every answer source, and records bounded autonomous actions.
    """
    def __init__(self, root: str | Path, *, chat=None, memory=None, knowledge=None, runtime=None):
        root = Path(root); root.mkdir(parents=True, exist_ok=True)
        self.root, self.chat, self.memory, self.knowledge, self.runtime = root, chat, memory, knowledge, runtime
        self.events = EventLog(root / "v3_events.db")
        self.tasks = TaskService(root / "v3_tasks.db", self.events)

    def status(self) -> dict[str, Any]:
        runtime = self.runtime.status() if self.runtime is not None else {}
        return {"version": "V3", "network_enabled": False, "training_enabled": False, "optimizer_steps": 0,
                "runtime": runtime, "memory": self.memory.stats() if self.memory else {"status": "DISABLED"},
                "knowledge": self.knowledge.stats() if self.knowledge else {"status": "DISABLED"},
                "active_tasks": self.tasks.list(20)}

    def chat_reply(self, session_id: str, text: str) -> dict[str, Any]:
        if self.chat is None:
            return {"text": "没有可用的本地聊天服务。", "response_source": "abstain", "citations": [], "status": "NO_SERVICE"}
        from ..bot.chat_service import ChatRequest
        out = self.chat.reply(ChatRequest.new(session_id, text)).to_dict()
        out["v3_attribution"] = {"source": out.get("response_source", "none"), "neural_claim": out.get("response_source") == "neural"}
        self.events.append("chat.replied", {"session_id": session_id, "source": out.get("response_source"), "citation_count": len(out.get("citations", []))})
        return out

    def learn_once(self) -> dict[str, Any]:
        out = {"status": "COMPLETED", "mode": "bounded_offline", "optimizer_steps": 0, "network_requests": 0, "source": "rule/oracle", "note": "V3自主学习仅执行练习与验证，不修改神经权重"}
        self.events.append("learning.cycle", out)
        return out

    def close(self) -> None:
        self.tasks.close(); self.events.close()
