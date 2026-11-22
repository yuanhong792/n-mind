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
             seed: int | None = None, stop_on_newline: bool = True,
             primed: tuple[torch.Tensor, LMState] | None = None
             ) -> tuple[GenerationRecord, LMState]:
    """自回归生成。

    primed：调用方已完成 prompt 前向时，直接传入 (prompt 末位 logits, prompt 末位
    LMState)，生成从该 logits 采样首个新 token，prompt_ids 不再前向（Gate-B 004
    free_generation_v2 的无重复前缀口径）。None 时保持历史行为（对 prompt_ids 做一次
    前向再采样）。两条路径的采样/停止/记录逻辑完全共用。"""
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
            if primed is not None:
                logits, st = primed
            else:
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


# 自由生成协议 ID（Gate-B 004 起）：协议显式命名，历史结果与协议绑定，不许事后重解释。
FREEGEN_PROTOCOL_V1 = "freegen-v1-anchor-reforward"
FREEGEN_PROTOCOL_V2 = "freegen-v2-direct-continuation"


def free_generation(model: NMindLM, tokenizer: CharTokenizer, prompt: str, *,
                    anchor: str = "助手：", max_new_tokens: int = 24,
                    greedy: bool = True) -> GenerationRecord:
    """v1 口径（协议 FREEGEN_PROTOCOL_V1，001—003 全部结果与待复现工件绑定此协议）：
    整段 prompt 前向拿递归状态，再把 anchor 作为 generate 的初始 prompt 续写。

    已知问题（Gate-B 004 审计实锤）：prompt 已以 anchor 结尾，generate 内部又把
    anchor 单独前向一次 —— anchor 的 token 被递归更新两次，首个生成 token 的条件
    状态是"prompt + 二次 anchor"。001—003 的所有臂（含基线/负对照）共用此口径，
    臂间比较内部一致，历史判定不改写；新实验一律使用 free_generation_v2。

    同一段逻辑原来在 `bot.runtime.neural_attempt`、`bot.learning_loop.free_gen_em`、
    `experiments/m5_language/run_m52.eval_free` 各写了一遍，三处容易走偏（忘了先前向
    prompt、或把 anchor 也算进生成结果）。集中在这里，其他调用方统一复用。
    不读标签、不 teacher forcing；`model.eval()` + `no_grad` 由 `generate` 内部保证。
    """
    if not prompt.endswith(anchor):
        raise ValueError(f"prompt 必须以锚点 {anchor!r} 结尾：{prompt!r}")
    ids = tokenizer.encode(prompt)
    with torch.no_grad():
        _logits, state = model(torch.tensor([ids], dtype=torch.long))
    rec, _st = generate(model, tokenizer, tokenizer.encode(anchor), state=state,
                         max_new_tokens=max_new_tokens, greedy=greedy)
    return rec


def free_generation_v2(model: NMindLM, tokenizer: CharTokenizer, prompt: str, *,
                       anchor: str = "助手：", max_new_tokens: int = 24,
                       greedy: bool = True) -> GenerationRecord:
    """v2 口径（协议 FREEGEN_PROTOCOL_V2，Gate-B 004 起的新实验统一协议）：
    完整 prompt **单次前向**，直接从最后一个位置的 logits 采样首个新 token，
    anchor 不二次前向；后续自回归与 v1 完全共用 generate 的采样/停止/记录逻辑。

    与 v1 的差异仅在首个 token 的条件状态：v1 = prompt + 二次 anchor 的递归更新，
    v2 = prompt 末位。答案字符续写的语义（从 "助手：" 之后接答案）两者一致。"""
    if not prompt.endswith(anchor):
        raise ValueError(f"prompt 必须以锚点 {anchor!r} 结尾：{prompt!r}")
    ids = tokenizer.encode(prompt)
    with torch.no_grad():
        logits, state = model(torch.tensor([ids], dtype=torch.long))
    rec, _st = generate(model, tokenizer, [], primed=(logits, state),
                        max_new_tokens=max_new_tokens, greedy=greedy)
    return rec
