"""V1.7.2 共享协议：BPE 边界一致性、正确的判别 token 评分、模式/状态管理。

本模块是本轮工程修复的核心，所有新脚本必须走这里，禁止再各自
`tok.encode(prompt + answer, add_eos=True)` 或拿 prompt 末尾 logits 当
"答案第 k 个 token" 的条件概率。

设计原则：
1. 自回归条件协议唯一：prompt_ids + answer_ids + [EOS]，分段编码后拼接。
2. 判别 token 若不在答案首位，必须先把前缀 token 逐个送入模型，
   再用该位置的输出 logits 评分；不得复用 prompt 末尾 logits。
3. 任何 eval/训练态切换必须成对恢复；样本之间状态必须显式新建。
4. 所有指标名称固定，禁止用首 token 代理冒充完整答案能力。
"""
from __future__ import annotations

import contextlib
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from nmind.language.tokenizer_v2 import ByteBPETokenizer, EOS
from nmind.language.bpe_decoder import effective_token_ids

LEGACY_PAIRS = [
    ("用户：你好\n助手：", "你好！"),
    ("用户：你叫什么？\n助手：", "我叫N-MIND。"),
    ("用户：谢谢\n助手：", "不客气！"),
    ("用户：小明有红色苹果\n助手：", "小明的苹果是红色的。"),
    ("用户：小红有绿色苹果\n助手：", "小红的苹果是绿色的。"),
    ("用户：小刚有蓝色苹果\n助手：", "小刚的苹果是蓝色的。"),
    ("用户：小丽有黄色苹果\n助手：", "小丽的苹果是黄色的。"),
    ("用户：我叫小明\n助手：", "你好，小明！"),
    ("用户：我叫什么？\n助手：", "你叫小明。"),
]
EXPECTED_TOK_HASH = "26af116ebd13ff00"


def build_tokenizer() -> ByteBPETokenizer:
    """按 V1.5 冻结语料重建 BPE；hash 不符直接拒绝（不改词表）。"""
    tok = ByteBPETokenizer.train([p + a for p, a in LEGACY_PAIRS], vocab_size=4096)
    if tok.vocab_hash() != EXPECTED_TOK_HASH:
        raise RuntimeError(f"tokenizer hash 漂移: {tok.vocab_hash()} != {EXPECTED_TOK_HASH}")
    return tok


@dataclass(frozen=True)
class TeachingSequence:
    prompt_ids: tuple[int, ...]
    answer_ids: tuple[int, ...]
    sequence: tuple[int, ...]
    answer_start: int          # 答案首 token 在 sequence 中的下标
    target_start: int          # 第一个被监督 target 在 labels(x[:,1:]) 中的下标
    eos_index: int             # EOS 在 sequence 中的下标
    legacy_sequence: tuple[int, ...] | None = None  # 旧口径 encode(prompt+answer) 仅用于审计


def encode_autoregressive(tok: ByteBPETokenizer, prompt: str, answer: str,
                          *, audit_legacy: bool = False) -> TeachingSequence:
    """唯一的训练/评测拼接协议：分段编码再拼接，末尾追加 EOS。

    target t 对应 sequence[t+1]，所以 answer 首 token 的 target 下标为
    answer_start - 1（prompt 最后一个 token 的输出预测答案首 token）。
    """
    prompt_ids = list(tok.encode(prompt))
    answer_ids = list(tok.encode(answer))
    seq = prompt_ids + answer_ids + [EOS]
    legacy = list(tok.encode(prompt + answer, add_eos=True)) if audit_legacy else None
    return TeachingSequence(
        prompt_ids=tuple(prompt_ids),
        answer_ids=tuple(answer_ids),
        sequence=tuple(seq),
        answer_start=len(prompt_ids),
        target_start=len(prompt_ids) - 1,
        eos_index=len(seq) - 1,
        legacy_sequence=tuple(legacy) if legacy is not None else None,
    )


def discriminative_positions(tok: ByteBPETokenizer, candidates: list[str]) -> dict[str, int]:
    """每个候选答案相对自身 token 序列的首次可判别位置。"""
    enc = {c: tok.encode(c) for c in candidates}
    out: dict[str, int] = {}
    for c, ids in enc.items():
        max_lcp = 0
        for other, oids in enc.items():
            if other == c:
                continue
            lcp = 0
            for a, b in zip(ids, oids):
                if a == b:
                    lcp += 1
                else:
                    break
            max_lcp = max(max_lcp, lcp)
        out[c] = min(max_lcp, len(ids) - 1)
    return enc, out


def valid_vocab(tok: ByteBPETokenizer, device=None):
    valid = torch.tensor(sorted(effective_token_ids(tok)), device=device)
    lookup = torch.full((tok.vocab_size,), -1, dtype=torch.long, device=device)
    lookup[valid] = torch.arange(valid.numel(), device=device)
    return valid, lookup


def _lp_at(logits: torch.Tensor, token_id: int, valid: torch.Tensor) -> float:
    pos = int((valid == token_id).nonzero(as_tuple=True)[0][0].item())
    return float(torch.log_softmax(logits.index_select(0, valid), -1)[pos])


def conditional_token_lps(model, tok: ByteBPETokenizer, prompt: str, answer: str,
                          *, valid: torch.Tensor | None = None,
                          start: int = 0) -> list[float]:
    """逐个答案 token 返回 log P(token_j | prompt, answer_<j)。

    当 start>0 时先把前 start 个答案 token 作为条件送入模型；
    这是 P4 修复的关键：判别位不在首位时不能拿 prompt 末尾 logits 评分。
    """
    if valid is None:
        valid, _ = valid_vocab(tok)
    ts = encode_autoregressive(tok, prompt, answer)
    aids = list(ts.answer_ids)
    model.eval()
    with torch.inference_mode():
        logits, state = model(torch.tensor([list(ts.prompt_ids)], dtype=torch.long))
        for j in range(start):
            tid = aids[j]
            logits, state = model(torch.tensor([[tid]], dtype=torch.long), state)
        return [_lp_at(logits[0, -1], aids[j], valid) for j in range(start, len(aids))]


def full_answer_conditional(model, tok: ByteBPETokenizer, prompt: str, answer: str,
                            *, valid: torch.Tensor | None = None) -> float:
    """完整答案条件 logprob（含 EOS），自回归逐 token，不复用任何缓存分数。"""
    if valid is None:
        valid, _ = valid_vocab(tok)
    ts = encode_autoregressive(tok, prompt, answer)
    aids = list(ts.answer_ids) + [EOS]
    model.eval()
    with torch.inference_mode():
        logits, state = model(torch.tensor([list(ts.prompt_ids)], dtype=torch.long))
        total = 0.0
        for j, tid in enumerate(aids):
            total += _lp_at(logits[0, -1], tid, valid)
            if j + 1 < len(aids):
                logits, state = model(torch.tensor([[tid]], dtype=torch.long), state)
    return total


def discriminative_proxy(model, tok: ByteBPETokenizer, prompt: str,
                         candidates: list[str], *, valid: torch.Tensor | None = None,
                         positions: dict[str, int] | None = None,
                         cached_prompt_logits: torch.Tensor | None = None) -> dict[str, float]:
    """DISCRIMINATIVE_TOKEN_PROXY：只评分判别 token 的条件概率。

    - 判别位 == 0：允许复用 prompt 末尾 logits（cached_prompt_logits）。
    - 判别位 >  0：必须逐 token 前馈前缀，绝不复用 prompt logits。
    结果不得命名为 binding_accuracy / full_answer_conditional_accuracy。
    """
    if valid is None:
        valid, _ = valid_vocab(tok)
    if positions is None:
        _, positions = discriminative_positions(tok, candidates)
    out: dict[str, float] = {}
    model.eval()
    with torch.inference_mode():
        for cand in candidates:
            pos = positions.get(cand, 0)
            if pos == 0 and cached_prompt_logits is not None:
                ids = tok.encode(cand)
                out[cand] = _lp_at(cached_prompt_logits, ids[0], valid)
            else:
                ts = encode_autoregressive(tok, prompt, cand)
                aids = list(ts.answer_ids)
                logits, state = model(torch.tensor([list(ts.prompt_ids)], dtype=torch.long))
                for j in range(pos):
                    logits, state = model(torch.tensor([[aids[j]]], dtype=torch.long), state)
                out[cand] = _lp_at(logits[0, -1], aids[pos], valid)
    return out


def prompt_last_logits(model, tok: ByteBPETokenizer, prompt: str) -> torch.Tensor:
    model.eval()
    with torch.inference_mode():
        logits, _ = model(torch.tensor([tok.encode(prompt)], dtype=torch.long))
    return logits[0, -1].clone()


@contextlib.contextmanager
def preserved_mode(model):
    """P5 修复：进入时记录 training flag 与 router 统计开关，退出时逐一恢复。"""
    was_training = bool(model.training)
    was_load_update = bool(getattr(model, "use_router_load_update", False))
    was_grad_path = bool(getattr(model, "router_gradient_path_enabled", False))
    load_snapshot = None
    count_snapshot = None
    if hasattr(model, "router") and hasattr(model.router, "load_bias"):
        load_snapshot = model.router.load_bias.detach().clone()
    if hasattr(model, "block_load_count"):
        count_snapshot = model.block_load_count.detach().clone()
    try:
        yield model
    finally:
        model.train(was_training)
        model.use_router_load_update = was_load_update
        model.router_gradient_path_enabled = was_grad_path
        if load_snapshot is not None:
            with torch.no_grad():
                model.router.load_bias.copy_(load_snapshot)
        if count_snapshot is not None:
            with torch.no_grad():
                model.block_load_count.copy_(count_snapshot)


@contextlib.contextmanager
def fresh_state(model, batch: int, device=None):
    """P5 修复：每个样本/批次显式新建状态，杜绝跨样本泄漏。"""
    dev = device or torch.device("cpu")
    st = model.initial_state(batch, dev)
    try:
        yield st
    finally:
        del st


METRIC_NAMES = (
    "binding_accuracy",
    "counterfactual_pair_accuracy",
    "position_accuracy",
    "dialogue_accuracy",
    "context_accuracy",
    "free_generation_accuracy",
    "immediate_eos_rate",
    "full_answer_conditional_accuracy",
)

PROXY_METRIC_NAMES = (
    "DISCRIMINATIVE_TOKEN_PROXY",
    "FIRST_TOKEN_PROXY_LEGACY",
)


def file_sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_sha(model) -> str:
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode("utf-8"))
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def write_json(path: str | Path, payload: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
