"""真实 DeepSeek 8B 聊天服务（AUTONOMOUS BOT 工作台的神经后端）。

设计目标：
  让原工作台 `web_server.py` 的聊天走真实本地 DeepSeek-R1-0528-Qwen3-8B，
  而不是旧的 ~45 万参数闭集 NMindLM。继承 ChatService 的全部非神经路由
  （算术工具 / 知识检索 / 记忆 / 云教师），只把「神经生成」一路替换为 8B。

诚实口径：
  * response_source = "neural" 表示真实本地 8B 神经生成（非检索拼接、非外部 API）；
  * note 明确标注模型 ID、4bit 量化、继承官方权重（非本系统训练）；
  * 模型未加载时如实报错，绝不静默回退到旧小模型或外部大模型兜底；
  * 本模块**只读推理**：不加载训练器、不创建优化器、不执行 optimizer.step。

复用资产（不重复造轮子）：
  * product/nmind8b.py 的 Neuro8B（4bit 推理 + 神经活动 hook + 预量化缓存）
  * 4bit 预量化缓存：首次 load 约 11.6s，而非 413s 冷量化
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterator

from .chat_service import (
    RESPONSE_SOURCES,
    ChatRequest,
    ChatResponse,
    ChatService,
    SourceCitation,
    _looks_explicitly_out_of_scope,
)

# product 目录（Neuro8B 所在）相对本文件的路径：src/nmind/bot -> n-mind/product
_PRODUCT_DIR = Path(__file__).resolve().parents[3] / "product"
REAL_MODEL_ID = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
REAL_PROTOCOL_ID = "deepseek-r1-qwen3-8b-4bit-local"


class DeepSeek8BChatService(ChatService):
    """把 Neuro8B 作为神经后端接入原工作台，其余路由完全继承 ChatService。"""

    def __init__(self, *, knowledge=None, memory_service=None, cloud_provider=None,
                 history_provider: Callable[[str], list[dict[str, Any]]] | None = None,
                 max_new_tokens: int = 400, lazy_load: bool = True,
                 auto_load: bool = True) -> None:
        # 先建一个不带模型的父类：保留 retrieval/tool/memory/cloud 路由
        super().__init__(knowledge=knowledge, memory_service=memory_service,
                         cloud_provider=cloud_provider, max_new_tokens=max_new_tokens)
        self._bot = None
        self._lock = threading.Lock()
        self._history_provider = history_provider
        self._auto_load = bool(auto_load)
        self.loading = False
        self.load_error: str | None = None
        self.model_sha = REAL_MODEL_ID          # 供状态栏展示真实模型身份
        self._vocab_size: int | None = None
        self._session_messages: dict[str, list[dict[str, str]]] = {}
        if not lazy_load:
            self.ensure_loaded()

    # ---------------------------------------------------------------- 加载（只读推理）
    @property
    def model_loaded(self) -> bool:
        return self._bot is not None and getattr(self._bot, "model", None) is not None

    def ensure_loaded(self) -> dict:
        """懒加载 8B（优先预量化缓存）。线程安全，失败如实记录，不回退。"""
        if self.model_loaded:
            return self._bot.load_info
        with self._lock:
            if self.model_loaded:
                return self._bot.load_info
            self.loading = True
            self.load_error = None
            try:
                import sys
                if str(_PRODUCT_DIR) not in sys.path:
                    sys.path.insert(0, str(_PRODUCT_DIR))
                from nmind8b import Neuro8B  # 延迟 import：启动不加载 torch/transformers
                bot = Neuro8B()
                info = bot.load()
                self._bot = bot
                self._vocab_size = info.get("vocab_size")
                self.model_sha = REAL_MODEL_ID
                return info
            except Exception as exc:  # noqa: BLE001 - 加载失败要让用户看到真实原因
                self.load_error = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                self.loading = False

    def load_info(self) -> dict:
        if not self.model_loaded:
            return {"status": "NOT_LOADED", "model_id": REAL_MODEL_ID,
                    "cache_ready": self._cache_ready(), "load_error": self.load_error,
                    "note": "模型未加载，首次对话时从 4bit 预量化缓存加载（约 12 秒）。"}
        info = dict(self._bot.load_info)
        info["cache_ready"] = True
        return info

    def _cache_ready(self) -> bool:
        try:
            cache = _PRODUCT_DIR / "models" / "DeepSeek-R1-0528-Qwen3-8B-4bit-cache"
            return (cache / "READY").exists() and (cache / "config.json").exists()
        except Exception:
            return False

    def resource_status(self) -> dict:
        if not self.model_loaded:
            return {"status": "NOT_LOADED", "vram_gb": 0.0}
        return self._bot.resource_status()

    def neural_snapshot(self) -> dict:
        if not self.model_loaded:
            return {"status": "NOT_LOADED", "note": "模型未加载，暂无真实神经活动。"}
        return self._bot._read_snapshot()

    # ---------------------------------------------------------------- 对话上下文
    def _messages_for(self, session_id: str, text: str,
                      supplied: list[dict[str, Any]] | None = None) -> list[dict[str, str]]:
        """优先使用请求携带的完整会话；没有时才从服务端历史恢复。"""
        if supplied:
            clean=[]
            for row in supplied:
                role=str(row.get('role') or '')
                body=str(row.get('content') or row.get('text') or '').strip()
                if role in ('user','assistant') and body:
                    clean.append({'role':role,'content':body})
            if clean:
                if clean[-1]['role'] != 'user' or clean[-1]['content'] != text.strip():
                    clean.append({'role':'user','content':text.strip()})
                return clean
        history: list[dict[str, str]] = []
        try:
            if self._history_provider:
                for row in (self._history_provider(session_id) or []):
                    role = str(row.get("role") or "")
                    body = str(row.get("text") or "").strip()
                    if role in ("user", "assistant") and body:
                        history.append({"role": role, "content": body})
        except Exception:
            history = []
        history = history[-16:]
        history.append({"role": "user", "content": text.strip()})
        return history

    # ---------------------------------------------------------------- 神经生成（替换父类）
    def reply(self, req: ChatRequest, *, prefer_cloud: bool = False) -> ChatResponse:
        t0 = time.perf_counter()
        # 云教师优先：仅在用户显式配置且调用方要求时使用，且标注来源
        if prefer_cloud and self.cloud_provider is not None:
            hit = self._try_cloud(req, (req.text or "").strip(), t0)
            if hit is not None:
                return hit

        text = (req.text or "").strip()
        if not text:
            return self._rule(req, "（空消息）请输入中文内容。", "没有可处理的内容。", t0)

        # 确定性算术仍走工具（比大模型可靠，且不算神经能力）——保持原有口径
        tool_hit = self._try_tool(req, text, t0)
        if tool_hit is not None:
            return tool_hit

        # 神经生成：真实本地 8B
        try:
            self.ensure_loaded()
            if not self.model_loaded:
                raise RuntimeError("8B 模型未加载")
            messages = self._messages_for(req.session_id, text)
            out = self._bot.chat(messages, max_new_tokens=self.max_new_tokens,
                                 temperature=0.6, top_p=0.9,
                                 capture_neural=True, keep_thinking=False)
            reply_text = (out.get("reply") or "").strip()
            if not reply_text:
                return ChatResponse(
                    message_id=req.message_id, session_id=req.session_id,
                    text="本地 8B 本次未产出最终答案（思考链可能被截断）。请重试或换一种问法。",
                    response_source="none", status="ERROR",
                    latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                    note="DeepSeek-R1 思考链未在 token 上限内闭合；未把未完成内容伪装成答案。")
            note = (f"本地神经生成：{REAL_MODEL_ID}（继承官方预训练权重，4bit NF4 量化，"
                    f"非本系统训练）。思考链已剥离，仅展示最终答案。")
            return ChatResponse(
                message_id=req.message_id, session_id=req.session_id,
                text=reply_text, response_source="neural",
                model_sha=self.model_sha, protocol_id=REAL_PROTOCOL_ID,
                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                finish_reason="stop", note=note,
                memory_hits=[{"neural": out.get("neural", {})}])
        except Exception as exc:  # noqa: BLE001
            return ChatResponse(
                message_id=req.message_id, session_id=req.session_id,
                text=f"本地 8B 推理失败：{type(exc).__name__}: {exc}",
                response_source="none", status="ERROR",
                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                note="如实报告推理错误；未回退到旧小模型或外部大模型。")

    # ---------------------------------------------------------------- 真实流式（SSE）
    def stream_reply(self, session_id: str, text: str, *, keep_thinking: bool = False,
                     max_new_tokens: int = 2048,
                     supplied_messages: list[dict[str, Any]] | None = None) -> Iterator[dict]:
        """逐 chunk 产出结构化事件 {delta, state}，供 /api/chat/stream 推送。"""
        text = (text or "").strip()
        if not text:
            yield {"delta": "（空消息）请输入中文内容。", "state": "done", "source": "rule"}
            return
        try:
            self.ensure_loaded()
            if not self.model_loaded:
                raise RuntimeError("8B 模型未加载")
            messages = self._messages_for(session_id, text, supplied_messages)
            for ev in self._bot.chat_stream(messages, max_new_tokens=max(1, int(max_new_tokens)),
                                            keep_thinking=keep_thinking):
                yield {**ev, "source": "neural", "model": REAL_MODEL_ID}
        except Exception as exc:  # noqa: BLE001
            # 错误不能伪装成正常 done，也不能回落到 none/资料不足。
            name = type(exc).__name__
            code = "MODEL_LOAD_FAILED" if "load" in name.lower() else "GENERATION_FAILED"
            yield {"delta": f"本地 N-MIND 推理失败（{code}）：{exc}",
                   "state": "error", "error_code": code, "source": "neural",
                   "model": REAL_MODEL_ID, "finish_reason": "error"}
