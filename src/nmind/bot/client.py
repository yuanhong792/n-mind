"""BotClient：GUI ↔ BotRuntime 的可靠消息客户端（跨进程安全）。

设计要点：
  * 通道是 BotStore（bot_state.db）里的 chat_messages 表——与运行时同一真相源，
    不另造第二套 SQLite，不沿用 read_text()+unlink() 的 inbox.jsonl 竞争写法；
  * submit() 返回 message_id，调用方必须用 wait_reply(message_id) 领取**对应**回复；
  * 同进程（GUI 内嵌 BotRuntime 服务线程）与跨进程（bot ask 老入口/另一终端）
    走完全相同的表协议；
  * 超时/取消语义明确：wait_reply 超时返回 None（不冒充完成）；
    cancel(message_id) 把 QUEUED/RETRYABLE 置为 CANCELLED。
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

from .chat_service import ChatRequest, ChatResponse, SourceCitation
from .store import BotStore


class BotClient:
    def __init__(self, root_dir: str | Path, *, db_path: str | Path | None = None) -> None:
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self._store = BotStore(db_path or (self.root / "bot_state.db"))

    def close(self) -> None:
        self._store.close()

    # ---------------------------------------------------------------- 发送
    def submit(self, text: str, *, session_id: str = "default") -> dict[str, Any]:
        req = ChatRequest.new(session_id, text)
        out = self._store.enqueue_message(req.message_id, session_id, text)
        return {"message_id": req.message_id, "session_id": session_id,
                "seq": out["seq"], "status": out["status"],
                "queue_position": self._store.queue_len()}

    def wait_reply(self, message_id: str, *, timeout_s: float = 120.0,
                   poll_s: float = 0.15) -> dict[str, Any] | None:
        """阻塞领取对应回复；超时返回 None。拿到的一定是该 message_id 的回复或失败态。"""
        deadline = time.monotonic() + max(0.0, timeout_s)
        while time.monotonic() < deadline:
            rec = self._store.get_reply(message_id)
            if rec is not None:
                return rec
            status = self._store.message_status(message_id)
            if status in ("CANCELLED",):
                return {"message_id": message_id, "status": "CANCELLED", "text": "",
                        "response_source": "none",
                        "note": "消息在处理前被取消。"}
            time.sleep(poll_s)
        return None

    def cancel(self, message_id: str) -> bool:
        """只有还没被领取的消息能取消；RUNNING 不可取消（服务在安全边界自行处理）。"""
        status = self.message_status(message_id)
        if status in ("QUEUED", "RETRYABLE"):
            return self._store.cancel_message(message_id)
        return False

    def message_status(self, message_id: str) -> str | None:
        return self._store.message_status(message_id)

    # ---------------------------------------------------------------- 会话
    def new_session_id(self) -> str:
        return uuid.uuid4().hex[:12]

    def sessions(self) -> list[dict[str, Any]]:
        return self._store.list_sessions()

    def create_session(self, session_id: str, title: str) -> None:
        self._store.create_session(session_id, title)

    def delete_session(self, session_id: str) -> int:
        return self._store.delete_session(session_id)

    def rename_session(self, session_id: str, title: str) -> None:
        self._store.rename_session(session_id, title)

    def history(self, session_id: str, *, limit: int = 200) -> list[dict[str, Any]]:
        return self._store.list_messages(session_id, limit=limit)

    def queue_len(self) -> int:
        return self._store.queue_len()

    def message(self, message_id: str) -> dict[str, Any] | None:
        """把 DB 行转成与 ChatResponse 同构的 dict（GUI 直接消费）。"""
        rec = self._store.get_reply(message_id)
        if rec is None:
            return None
        return self.to_response_dict(rec)

    @staticmethod
    def to_response_dict(rec: dict[str, Any]) -> dict[str, Any]:
        citations = rec.get("citations")
        if citations is None:
            import json
            try:
                citations = json.loads(rec.get("citations_json") or "[]")
            except ValueError:
                citations = []
        return {"message_id": rec["message_id"], "status": rec.get("status", "DONE"),
                "text": rec.get("text", ""), "response_source": rec.get("response_source", ""),
                "citations": citations, "model_sha": rec.get("model_sha", ""),
                "protocol_id": rec.get("protocol_id", ""),
                "latency_ms": rec.get("latency_ms") or 0.0, "note": rec.get("note", ""),
                "unk_rate": rec.get("unk_rate"),
                "finish_reason": rec.get("finish_reason"),
                "memory_hits": rec.get("memory_hits", [])}


def build_response(message_id: str, session_id: str, resp: ChatResponse) -> dict[str, Any]:
    """服务端把 ChatResponse 落库前统一转换（引用 to_dict，防止字段漂移）。"""
    return {"message_id": message_id, "session_id": session_id,
            "text": resp.text, "response_source": resp.response_source,
            "citations": [c.to_dict() if isinstance(c, SourceCitation) else c
                          for c in resp.citations],
            "model_sha": resp.model_sha, "protocol_id": resp.protocol_id,
            "latency_ms": resp.latency_ms, "note": resp.note,
            "unk_rate": resp.unk_rate, "finish_reason": resp.finish_reason,
            "memory_hits": resp.memory_hits}
