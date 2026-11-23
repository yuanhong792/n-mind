from __future__ import annotations
from dataclasses import dataclass, field
import torch
from .tokenizer_v2 import ByteBPETokenizer, EOS

def effective_token_ids(tokenizer: ByteBPETokenizer) -> list[int]:
    """IDs that have an actual byte, merge, or protocol EOS meaning."""
    return sorted({*range(4, 260), EOS, *[260 + i for i in range(len(tokenizer.merges))]})

def constrained_causal_loss(logits: torch.Tensor, labels: torch.Tensor, tokenizer: ByteBPETokenizer) -> torch.Tensor:
    """Causal CE over the same effective vocabulary used by generation."""
    valid = set(effective_token_ids(tokenizer)); flat_labels = labels.reshape(-1)
    if any(int(x) not in valid for x in flat_labels.detach().cpu().tolist()):
        raise ValueError("training label contains an undefined BPE token")
    allowed = torch.tensor(effective_token_ids(tokenizer), device=logits.device)
    masked = logits.index_select(-1, allowed)
    lookup = torch.full((tokenizer.vocab_size,), -1, dtype=torch.long, device=logits.device)
    lookup[allowed] = torch.arange(allowed.numel(), device=logits.device)
    return torch.nn.functional.cross_entropy(masked.reshape(-1, allowed.numel()), lookup[flat_labels].reshape(-1))

# ---------------- V1.7 UTF-8 incremental decode ----------------
def utf8_prefix_status(buf: bytes) -> str:
    """严格 RFC 3629 判定字节缓冲的 UTF-8 状态：
    - complete:   所有字符完整合法
    - incomplete: 结尾是一个合法但尚未凑齐的多字节序列前缀（可继续生成）
    - invalid:    不可恢复错误（非法起始/续字节、过长编码、越界码点）
    仅依据字节本身；不猜测、不替换任何字节。
    """
    i, n = 0, len(buf)
    while i < n:
        b = buf[i]
        if b <= 0x7F:
            i += 1; continue
        if 0xC2 <= b <= 0xDF:
            need, lo, hi = 1, 0x80, 0xBF
        elif 0xE0 <= b <= 0xEF:
            need, lo, hi = 2, 0x80, 0xBF
        elif 0xF0 <= b <= 0xF4:
            need, lo, hi = 3, 0x80, 0xBF
        else:
            return "invalid"
        avail = n - i - 1
        upto = min(avail, need)
        for k in range(1, upto + 1):
            if not (lo <= buf[i + k] <= hi): return "invalid"
        if upto >= 1:
            if b == 0xE0 and buf[i + 1] < 0xA0: return "invalid"
            if b == 0xED and buf[i + 1] > 0x9F: return "invalid"
            if b == 0xF0 and buf[i + 1] < 0x90: return "invalid"
            if b == 0xF4 and buf[i + 1] > 0x8F: return "invalid"
        if avail < need: return "incomplete"
        i += need + 1
    return "complete"

@dataclass
class BPEGeneration:
    token_ids: list[int] = field(default_factory=list)
    text: str = ""
    finish_reason: str = "max_tokens"   # eos | max_tokens | utf8_incomplete | byte_decode_error | utf8_filtered
    invalid_candidate_ids: list[int] = field(default_factory=list)  # 未定义词表槽位(旧字段保留)
    # V1.7 可审计字段：原始 token 永不删除
    rejected_utf8_tokens: list[int] = field(default_factory=list)  # 因破坏 UTF-8 被拒的候选
    utf8_incomplete_steps: int = 0       # 出现合法但未完成字节前缀的步数
    eos_blocked_by_utf8: int = 0         # UTF-8 未完成时 EOS 被拦截次数
    raw_byte_hex: str = ""               # 最终字节缓冲(hex)

def _greedy_step(last: torch.Tensor, valid_ids: list[int], forbidden: set[int]) -> int:
    """在有效词表中按 logit 从高到低选第一个未被禁的 token。"""
    scores = last.index_select(0, torch.tensor(valid_ids, dtype=torch.long, device=last.device))
    for j in sorted(range(len(valid_ids)), key=lambda x: -float(scores[x])):
        tid = valid_ids[j]
        if tid not in forbidden:
            return tid
    return -1

def generate_bpe(model, tokenizer: ByteBPETokenizer, prompt: str, *, max_new_tokens=32,
                 utf8_safe: bool = True):
    """贪心 BPE 生成 + 显式有效词表掩码。

    V1.7 utf8_safe=True：增量 UTF-8 前缀检查 —— 逐 token 展开真实字节，
    拒绝会破坏 UTF-8 的候选；允许汉字字节跨 token 形成；UTF-8 未完成时禁止 EOS；
    原始 token / 被拒候选 / 字节缓冲全部保留，绝不静默替换为"正确答案"。
    utf8_safe=False 走 V1.5 旧路径，仅供冻结模型 A/B 对比、分离 decoder 收益。
    """
    model.eval(); ids = tokenizer.encode(prompt); x = torch.tensor([ids], dtype=torch.long)
    active = set(effective_token_ids(tokenizer)); valid_ids = sorted(active)
    rec = BPEGeneration(); blocked = []; buf = bytearray()
    with torch.inference_mode():
        logits, state = model(x)
        for _ in range(max_new_tokens):
            last = logits[0, -1].clone()
            blocked.extend(i for i in range(last.numel()) if i not in active)
            forbidden: set[int] = set()
            if utf8_safe:
                cur_status = utf8_prefix_status(bytes(buf))
                if cur_status == "incomplete":
                    rec.utf8_incomplete_steps += 1
                    forbidden.add(EOS)
                for tid in valid_ids:
                    if tid == EOS: continue
                    tb = tokenizer.token_bytes(tid)
                    if tb is None or utf8_prefix_status(bytes(buf) + tb) == "invalid":
                        forbidden.add(tid); rec.rejected_utf8_tokens.append(tid)
                tid = _greedy_step(last, valid_ids, forbidden)
                if tid == -1:
                    rec.finish_reason = "utf8_filtered"; break
                if tid == EOS:
                    if cur_status == "incomplete":
                        rec.eos_blocked_by_utf8 += 1; continue
                    rec.finish_reason = "eos"; break
                tb = tokenizer.token_bytes(tid)
                rec.token_ids.append(tid)
                buf.extend(tb or b"")
                logits, state = model(torch.tensor([[tid]]), state)
            else:
                mask = torch.full_like(last, float("-inf"))
                valid = torch.tensor(valid_ids, dtype=torch.long)
                mask[valid] = last[valid]; tid = int(torch.argmax(mask))
                if tid == EOS: rec.finish_reason = "eos"; break
                rec.token_ids.append(tid)
                tb = tokenizer.token_bytes(tid); buf.extend(tb or b"")
                logits, state = model(torch.tensor([[tid]]), state)
    rec.invalid_candidate_ids = sorted(set(blocked)); rec.raw_byte_hex = bytes(buf).hex()
    status = utf8_prefix_status(bytes(buf))
    if status == "complete":
        try: rec.text = bytes(buf).decode("utf-8")
        except UnicodeDecodeError: rec.finish_reason = "byte_decode_error"
    elif status == "incomplete":
        # 保留原始字节，不猜字；用 ignore 仅为展示，并在 finish_reason 如实标记
        rec.text = bytes(buf).decode("utf-8", "ignore")
        if rec.finish_reason == "max_tokens": rec.finish_reason = "utf8_incomplete"
    else:
        rec.finish_reason = "byte_decode_error"; rec.text = ""
    return rec

# 旧 decoder 显式别名，供冻结模型 A/B 对比（分离 decoder 修复收益）
def generate_bpe_legacy(model, tokenizer, prompt, *, max_new_tokens=32):
    return generate_bpe(model, tokenizer, prompt, max_new_tokens=max_new_tokens, utf8_safe=False)
