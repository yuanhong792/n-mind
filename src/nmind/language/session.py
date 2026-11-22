"""会话状态管理（M5-Language 任务书 P0/P3）：跨轮递归状态 + 工具路由 + reset/隔离。

状态一致性设计（与训练分布对齐）：
  * send() 先把本轮用户后缀吸收进主状态；生成在**分支状态**上自回归展开；
  * 生成结束后，主状态按"助手：{最终回复}+EOS"吸收——无论回复来自神经生成、
    工具结果还是检索结果，会话转录与训练序列结构保持一致；
  * 神经生成 / 工具 / 检索 通过 response_source 显式区分，绝不互相冒充；
  * 输入长度、轮数、生成长度均有硬预算；reset 清空全部状态；不同 session 互不共享。
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from ..skills.toolbox import calculator
from .decoder import generate
from .dialogue_data import FACTS
from .neural_dialogue import LMState, NMindLM
from .tokenizer import CharTokenizer


@dataclass
class ChatReply:
    text: str
    response_source: str            # neural | tool | retrieval
    generated_text: str             # 模型原始生成（工具路由前）
    tool: str | None
    tool_result: str | None
    finish_reason: str
    session_id: str
    turn_index: int
    latency_ms: float


class ChatSession:
    """单会话：递归状态跨轮携带；reset 清空；不同会话互不共享任何状态。"""

    def __init__(self, model: NMindLM, tokenizer: CharTokenizer, *, session_id: str = "s1",
                 max_turns: int = 100, max_input_chars: int = 200, max_new_tokens: int = 32,
                 greedy: bool = True, temperature: float = 1.0, sample_seed: int | None = None,
                 tools: dict | None = None, facts: dict | None = None):
        self.model = model
        self.tok = tokenizer
        self.max_turns = max_turns
        self.max_input_chars = max_input_chars
        self.max_new_tokens = max_new_tokens
        self.greedy = greedy
        self.temperature = temperature
        self.sample_seed = sample_seed
        self.tools = tools or {"calculator": calculator}
        self.facts = facts or dict(FACTS)
        self.transcript: list[dict] = []
        self.reset(session_id)

    def reset(self, session_id: str | None = None) -> None:
        self.session_id = session_id or f"{self.session_id}-r{int(time.time())}"
        self._state: LMState | None = None
        self.turns = 0

    def _absorb(self, text: str) -> None:
        """把文本吸收进主递归状态（no_grad、eval；仅更新状态，不产生输出）。"""
        ids = self.tok.encode(text, add_eos=True)
        if not ids:
            return
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            _, self._state = self.model(torch.tensor([ids], dtype=torch.long), self._state)
        if was_training:
            self.model.train()

    def send(self, user_text: str) -> ChatReply:
        if self.turns >= self.max_turns:
            return ChatReply(text="会话轮数已达上限，请 reset。", response_source="rule_baseline",
                              generated_text="", tool=None, tool_result=None,
                              finish_reason="max_turns", session_id=self.session_id,
                              turn_index=self.turns, latency_ms=0.0)
        t0 = time.perf_counter()
        user_text = user_text.strip()[: self.max_input_chars]
        self.turns += 1
        self._absorb(f"用户：{user_text}\n")
        seed = None if self.sample_seed is None else self.sample_seed + self.turns
        rec, _branch = generate(self.model, self.tok, self.tok.encode("助手："),
                                state=self._state, max_new_tokens=self.max_new_tokens,
                                greedy=self.greedy, temperature=self.temperature,
                                seed=seed, stop_on_newline=True)
        gen_text = rec.text.strip()
        reply_text, source, tool_name, tool_result = gen_text, "neural", None, None
        if gen_text.startswith("TOOL_CALCULATOR"):
            expr = gen_text[len("TOOL_CALCULATOR"):].strip()
            tool_name = "calculator"
            try:
                val = self.tools["calculator"](expr)
                tool_result = f"计算结果：{val:g}。"
            except Exception as e:                     # 工具失败如实呈现，不伪装
                tool_result = f"计算失败：{e}"
            reply_text = tool_result or ""
            source = "tool"
        elif gen_text.startswith("TOOL_LOOKUP"):
            key = gen_text[len("TOOL_LOOKUP"):].strip()
            tool_name = "lookup"
            val = self.facts.get(key)
            tool_result = f"查询结果：{val}。" if val else "查询结果：未找到该资料。"
            reply_text = tool_result
            source = "retrieval"
        elif not gen_text:
            reply_text = "（空生成）"
            source = "neural"
        self._absorb(f"助手：{reply_text}")
        self.transcript.append({"role": "user", "text": user_text})
        self.transcript.append({"role": "assistant", "text": reply_text,
                                 "source": source, "generated": gen_text})
        return ChatReply(text=reply_text, response_source=source, generated_text=gen_text,
                          tool=tool_name, tool_result=tool_result,
                          finish_reason=rec.finish_reason, session_id=self.session_id,
                          turn_index=self.turns,
                          latency_ms=round((time.perf_counter() - t0) * 1000, 2))
