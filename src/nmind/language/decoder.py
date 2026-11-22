"""自回归自由解码（M5-Language 任务书 P0）：绝不读取目标答案。

* greedy / 带种子 temperature 采样 / 可选重复惩罚；
* 停止于 EOS、换行（轮次边界）或 max_new_tokens；
* 逐步记录 token 与 logprob（生成记录可审计）；
* 生成期间 model.eval() + torch.no_grad()：无优化器、无权重副作用；
* 返回的 state 是**分支状态**（随生成 token 延伸），调用方的主会话状态由
  session 按"助手：{最终回复}+EOS"统一吸收——与训练分布保持一致。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .neural_dialogue import LMState, NMindLM
from .tokenizer import EOS, CharTokenizer


@dataclass
class GenerationRecord:
    token_ids: list[int] = field(default_factory=list)
    tokens: list[str] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    finish_reason: str = "max_tokens"      # eos | newline | max_tokens
    text: str = ""


def generate(model: NMindLM, tokenizer: CharTokenizer, prompt_ids: list[int], *,
             state: LMState | None = None, max_new_tokens: int = 32, greedy: bool = True,
             temperature: float = 1.0, repetition_penalty: float | None = None,
             seed: int | None = None, stop_on_newline: bool = True) -> tuple[GenerationRecord, LMState]:
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens 必须 > 0")
    was_training = model.training
    model.eval()
    gen = None
    if seed is not None and not greedy:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))
    produced: list[int] = []
    logps: list[float] = []
    finish = "max_tokens"
    newline_id = tokenizer.vocab.get("\n")
    rec = GenerationRecord()
    try:
        with torch.no_grad():
            x = torch.tensor([list(prompt_ids)], dtype=torch.long)
            logits, st = model(x, state)
            for _ in range(max_new_tokens):
                last = logits[0, -1].clone()
                if repetition_penalty and produced:
                    for tid in set(produced):
                        last[tid] -= (float(repetition_penalty) if last[tid] > 0
                                       else -float(repetition_penalty))
                logprobs = torch.log_softmax(last, dim=-1)
                if greedy:
                    tid = int(torch.argmax(last))
                else:
                    probs = torch.softmax(last / max(1e-6, float(temperature)), dim=-1)
                    tid = int(torch.multinomial(probs, 1, generator=gen))
                logp = float(logprobs[tid])
                if tid == EOS:
                    finish = "eos"
                    break
                if stop_on_newline and newline_id is not None and tid == newline_id:
                    finish = "newline"
                    break
                produced.append(tid)
                logps.append(logp)
                rec.tokens.append(tokenizer.id2tok.get(tid, "<unk>"))
                logits, st = model(torch.tensor([[tid]], dtype=torch.long), st)
    finally:
        if was_training:
            model.train()
    rec.token_ids = produced
    rec.logprobs = logps
    rec.finish_reason = finish
    rec.text = tokenizer.decode(produced, skip_special=True)
    return rec, st
