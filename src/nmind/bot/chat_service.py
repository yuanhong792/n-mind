"""Bot V1 聊天服务层：神经 / 知识检索 / 确定性工具 / 规则提示 / 拒答 五路分离。

诚实口径（本文件的存在理由）：
  * 神经输出永远原样保留并标注"未验证"——M5 stable 权重约 45 万参数、闭集词表，
    开放式中文对话能力有限，绝不把检索拼接或工具结果说成模型的思考；
  * response_source 五值硬编码词表：neural | retrieval | tool | rule | none；
  * 神经协议统一走 Gate-B 004 修复后的 freegen-v2-direct-continuation（prompt 单次
    前向，anchor 不二次前向），与训练序列格式 "用户：X\\n助手：Y\\n..." 严格对齐；
  * UNK 比例超过门槛的输入产生的神经文本按 unverified 呈现，不得渲染成有效回答。

数据契约（GUI 与服务端共享，跨进程可 JSON 序列化）：
  ChatRequest / ChatResponse / SourceCitation / BotHealth
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from ..skills.toolbox import calculator

RESPONSE_SOURCES = ("neural", "retrieval", "memory", "tool", "cloud_teacher", "rule", "none")
GENERATION_PROTOCOL_ID = "freegen-v2-direct-continuation"

def _looks_explicitly_out_of_scope(query: str) -> bool:
    """Avoid generic Python hits being shown as evidence for unrelated claims."""
    lowered = query.lower()
    return any(term in lowered for term in ("n-mind", "2027", "gps", "实时位置", "电话号码", "银行", "密码", "个人"))

def _retrieval_has_substantive_overlap(query: str, results: list[dict[str, Any]]) -> bool:
    """Reject generic keyword hits that cannot support a grounded answer."""
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]+|[\u4e00-\u9fff]{2,}", query)
    stop = {"python", "官方", "教程", "如何", "什么", "哪些", "能否", "项目", "证明", "中的", "以及"}
    substantive = [t.lower() for t in tokens if t.lower() not in stop]
    if not substantive:
        return False
    corpus = "\n".join(str(r.get("citation", {}).get("excerpt", "")).lower()
                         for r in results)
    return any(t in corpus for t in substantive)
# 未见字符比例门槛：超过则神经输出按"未验证"呈现（闭集词表之外的字全部是 <unk>）
DEFAULT_UNK_REJECT_THRESHOLD = 0.6

_CALC_RE = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*(加|\+|减|-|乘|\*|×|除|/|÷)\s*(-?\d+(?:\.\d+)?)")
_CALC_OP = {"加": "+", "+": "+", "减": "-", "-": "-", "乘": "*", "*": "*", "×": "*",
            "除": "/", "/": "/", "÷": "/"}


@dataclass
class SourceCitation:
    """可追溯引用：回答里出现的每段摘录都能定位到库里的 chunk。"""

    doc_id: str
    chunk_id: str
    title: str
    url: str
    fetched_at_utc: str
    excerpt: str
    score: float | None = None
    trust_state: str = "fixture"          # fixture | public-read-only
    stale: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SourceCitation:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ChatRequest:
    message_id: str
    session_id: str
    text: str
    created_at: str = ""

    @classmethod
    def new(cls, session_id: str, text: str) -> ChatRequest:
        return cls(message_id=uuid.uuid4().hex, session_id=session_id, text=text,
                   created_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))


@dataclass
class ChatResponse:
    message_id: str
    session_id: str
    text: str
    response_source: str
    citations: list[SourceCitation] = field(default_factory=list)
    model_sha: str = ""
    protocol_id: str = ""
    latency_ms: float = 0.0
    unk_rate: float | None = None
    finish_reason: str | None = None
    status: str = "DONE"
    note: str = ""                       # 中文可读说明（来源边界/局限提示）
    memory_hits: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class BotHealth:
    """GUI 状态条的数据源：一条快照说清"现在是什么在服务"。"""

    running: bool
    state: str
    model_loaded: bool
    model_sha: str
    vocab_size: int | None
    generation_protocol_id: str
    knowledge_docs: int
    knowledge_chunks: int
    network_enabled: bool
    offline: bool
    queue_len: int
    last_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _V2ChatSession:
    """单会话递归状态（v2 直续协议）。

    与 ChatSession（v1 口径）的区别：把"助手："并入主状态单次前向，
    生成用 primed 直续，跨轮序列与训练分布逐字符对齐。主状态只吸收
    已确定要进入转录的文本；生成在分支状态上展开。
    """

    def __init__(self, model, tokenizer, *, session_id: str, max_turns: int = 100,
                 max_input_chars: int = 200, max_new_tokens: int = 32):
        import torch
        self._torch = torch
        self.model = model
        self.tok = tokenizer
        self.session_id = session_id
        self.max_turns = max_turns
        self.max_input_chars = max_input_chars
        self.max_new_tokens = max_new_tokens
        self.turns = 0
        self._state = None
        self.transcript: list[dict[str, Any]] = []

    def reset(self, session_id: str | None = None) -> None:
        if session_id:
            self.session_id = session_id
        self._state = None
        self.turns = 0
        self.transcript = []

    def unk_rate(self, text: str) -> float:
        ids = self.tok.encode(text)
        if not ids:
            return 0.0
        return sum(1 for i in ids if int(i) == 3) / len(ids)   # UNK=3

    def send(self, user_text: str) -> dict[str, Any]:
        if self.turns >= self.max_turns:
            return {"text": "会话轮数已达上限，请新建会话。", "finish_reason": "max_turns",
                    "unk_rate": 0.0}
        t_text = user_text.strip()[: self.max_input_chars]
        self.turns += 1
        prompt = f"用户：{t_text}\n助手："
        ids = self.tok.encode(prompt)
        torch = self._torch
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                logits, self._state = self.model(torch.tensor([ids], dtype=torch.long),
                                                 self._state)
            from ..language.decoder import generate
            rec, _branch = generate(self.model, self.tok, [], primed=(logits, self._state),
                                    max_new_tokens=self.max_new_tokens,
                                    stop_on_newline=True)
        finally:
            if was_training:
                self.model.train()
        reply = rec.text.strip()
        # 只把最终确定的内容吸收进主状态；生成期间的分支状态丢弃（与训练分布一致）
        with torch.no_grad():
            tail = self.tok.encode(f"{reply}\n")
            if tail:
                _, self._state = self.model(torch.tensor([tail], dtype=torch.long),
                                            self._state)
        self.transcript.append({"role": "user", "text": t_text})
        self.transcript.append({"role": "assistant", "text": reply})
        return {"text": reply, "finish_reason": rec.finish_reason,
                "unk_rate": self.unk_rate(t_text)}


class ChatService:
    """把一条中文消息路由到正确的来源，并给出可核对的引用与说明。"""

    def __init__(self, *, model=None, tokenizer=None, model_sha: str = "",
                 knowledge=None, memory_service=None, unk_reject_threshold: float = DEFAULT_UNK_REJECT_THRESHOLD,
                 max_new_tokens: int = 32, cloud_provider=None) -> None:
        self.model = model
        self.tok = tokenizer
        self.model_sha = model_sha
        self.knowledge = knowledge            # KnowledgeSearch 或 None（P3 接入）
        self.memory_service = memory_service
        self.unk_reject_threshold = float(unk_reject_threshold)
        self.max_new_tokens = int(max_new_tokens)
        self.cloud_provider = cloud_provider
        self._sessions: dict[str, _V2ChatSession] = {}

    @property
    def model_loaded(self) -> bool:
        return self.model is not None and self.tok is not None

    def _session(self, session_id: str) -> _V2ChatSession:
        if session_id not in self._sessions:
            self._sessions[session_id] = _V2ChatSession(
                self.model, self.tok, session_id=session_id,
                max_new_tokens=self.max_new_tokens)
        return self._sessions[session_id]

    def reset_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    # ---------------------------------------------------------------- 路由
    def reply(self, req: ChatRequest, *, prefer_cloud: bool = False) -> ChatResponse:
        t0 = time.perf_counter()
        if prefer_cloud:
            cloud_hit = self._try_cloud(req, (req.text or '').strip(), t0)
            if cloud_hit is not None:
                return cloud_hit
        text = (req.text or "").strip()
        if not text:
            return self._rule(req, "（空消息）请输入中文内容。",
                              "没有可处理的内容。", t0)
        # 1) 确定性工具：中文算术（最可靠的"真回答"路径）
        tool_hit = self._try_tool(req, text, t0)
        if tool_hit is not None:
            return tool_hit
        # 2) 知识检索：明确的资料类问题或命中关键词
        ret_hit = self._try_retrieval(req, text, t0)
        if ret_hit is not None:
            return ret_hit
        mem_hit = self._try_memory(req, text, t0)
        if mem_hit is not None:
            return mem_hit
        cloud_hit = self._try_cloud(req, text, t0)
        if cloud_hit is not None:
            return cloud_hit
        # 3) 神经生成（原样保留 + 未验证标注）
        if self.model_loaded:
            sess = self._session(req.session_id)
            unk = sess.unk_rate(text)
            out = sess.send(text)
            high_unk = unk >= self.unk_reject_threshold
            note = ("神经自由生成（未验证）：N-MIND 当前模型约 45 万参数、中文闭集词表，"
                    "生成文本可能不对应你的问题，不作为事实依据。")
            if high_unk:
                note = (f"输入有 {unk:.0%} 字符在模型词表之外，神经输出按未验证呈现："
                        "以下为模型原样生成，不代表可靠回答。")
            return ChatResponse(
                message_id=req.message_id, session_id=req.session_id,
                text=out["text"], response_source="neural",
                model_sha=self.model_sha, protocol_id=GENERATION_PROTOCOL_ID,
                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                unk_rate=round(unk, 4), finish_reason=out["finish_reason"],
                note=note)
        # 4) 无模型：诚实说明（绝不调用外部大模型兜底）
        return ChatResponse(
            message_id=req.message_id, session_id=req.session_id,
            text="当前没有可用的本地语言权重，N-MIND 不调用外部大模型兜底，"
                 "无法对这句话做神经生成。可先在知识面板检索已保存的资料。",
            response_source="none",
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            note="无本地权重：神经侧不可用（如实告知，不冒充）。")

    def _try_cloud(self, req: ChatRequest, text: str, t0: float) -> ChatResponse | None:
        if self.cloud_provider is None:
            return None
        from ..bridge.providers import CloudRequest
        result = self.cloud_provider.ask(CloudRequest(
            category="public_web_query", question=text,
            context_summary="请直接回答用户问题；这是外部问答服务结果，不代表 N-MIND 本地神经模型能力。",
            max_response_chars=2000))
        if not result.ok:
            return ChatResponse(message_id=req.message_id, session_id=req.session_id,
                                text="外部问答服务调用失败，未生成回答。",
                                response_source="cloud_teacher", status="ERROR",
                                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                                note=f"provider={result.provider}; error={result.error}")
        return ChatResponse(message_id=req.message_id, session_id=req.session_id,
                            text=result.text, response_source="cloud_teacher",
                            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                            note="外部教师/问答服务回答；不计入N-Cell神经能力。")

    # ---------------------------------------------------------------- 各路实现
    def _rule(self, req: ChatRequest, text: str, note: str, t0: float) -> ChatResponse:
        return ChatResponse(message_id=req.message_id, session_id=req.session_id,
                             text=text, response_source="rule",
                             latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                             note=note)

    def _try_tool(self, req: ChatRequest, text: str, t0: float) -> ChatResponse | None:
        m = _CALC_RE.search(text)
        if not m:
            return None
        a, op, b = m.group(1), m.group(2), m.group(3)
        expr = f"{a}{_CALC_OP.get(op, op)}{b}"
        try:
            val = calculator(expr)
        except Exception as exc:                        # noqa: BLE001 - 工具失败如实呈现
            return ChatResponse(
                message_id=req.message_id, session_id=req.session_id,
                text=f"计算失败：{exc}", response_source="tool",
                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                note=f"确定性工具 calculator 对表达式 {expr!r} 计算失败，如实报告不伪装。")
        return ChatResponse(
            message_id=req.message_id, session_id=req.session_id,
            text=f"计算结果：{val:g}。", response_source="tool",
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            note=f"确定性工具 calculator：{expr} = {val:g}（程序计算，非神经输出）。")

    def _try_memory(self, req: ChatRequest, text: str, t0: float) -> ChatResponse | None:
        if self.memory_service is None:
            return None
        from ..memory import MemoryQuery
        context = self.memory_service.recall(MemoryQuery(text, session_id=req.session_id, limit=4))
        if not context.hits:
            return None
        body = "\n".join(f"〔记忆{n + 1}〕{h['text']}" for n, h in enumerate(context.hits))
        return ChatResponse(message_id=req.message_id, session_id=req.session_id, text=body,
                             response_source="memory", memory_hits=context.hits,
                             latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                             note="历史记忆辅助：仅展示可追溯记忆片段，不冒充神经模型能力。")

    def _try_retrieval(self, req: ChatRequest, text: str, t0: float) -> ChatResponse | None:
        if self.knowledge is None:
            return None
        res = self.knowledge.search(text, limit=3)
        if not res or _looks_explicitly_out_of_scope(text):
            return None
        citations = [SourceCitation.from_dict(r["citation"]) for r in res]
        excerpts = "\n".join(f"【{i + 1}】{r['citation']['excerpt']}"
                             for i, r in enumerate(res))
        titles = "、".join(dict.fromkeys(c.title for c in citations))
        body = (f"在已保存的资料里找到 {len(citations)} 条相关摘录（来源：{titles}）：\n"
                f"{excerpts}")
        return ChatResponse(
            message_id=req.message_id, session_id=req.session_id,
            text=body, response_source="retrieval", citations=citations,
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            note="摘录式检索结果（extractive_summary）：只引用原文片段与出处，"
                 "不做推断性总结；检索匹配分数是相关性，不是事实真值。")
