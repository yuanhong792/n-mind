"""M2.1 P0.1：SEP 训练/推理协议修复工具。

M2 缺陷：Legacy 模型用无 SEP 训练（序列=prompt+answer+EOS），但评测函数默认
用 SEP 输入（prompt+[SEP]），造成训练/评测协议不一致。本模块提供双协议
encode/generate，供交叉评测使用。

- legacy 序列：prompt_ids + answer_ids + [EOS]
- sep    序列：prompt_ids + [SEP] + answer_ids + [EOS]
tokenizer 字符 ID 与 M2 完全一致（从 math_m2.tokenizer_m2 加载，避免漂移）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
M2ROOT = HERE.parent.parent / "math_m2"
sys.path.insert(0, str(M2ROOT))

from tokenizer_m2 import M2Tokenizer, EOS, SEP, PAD  # noqa: E402

PROTOCOLS = ("legacy", "sep")


def encode_sequence(tok: M2Tokenizer, prompt: str, answer: str, protocol: str):
    """返回 (seq, meta)。meta 记录 answer 监督位置的起始下标。"""
    p = tok.encode(prompt)
    a = tok.encode(answer)
    if protocol == "sep":
        seq = p + [SEP] + a + [EOS]
        a_start = len(p) + 1
        return seq, {"prompt_len": len(p), "answer_start": a_start,
                     "answer_len": len(a), "eos_index": len(seq) - 1}
    if protocol == "legacy":
        seq = p + a + [EOS]
        a_start = len(p)
        return seq, {"prompt_len": len(p), "answer_start": a_start,
                     "answer_len": len(a), "eos_index": len(seq) - 1}
    raise ValueError(f"unknown protocol {protocol}")


def encode_inference(tok: M2Tokenizer, prompt: str, protocol: str):
    """推理输入。legacy=prompt_ids；sep=prompt_ids+[SEP]。"""
    p = tok.encode(prompt)
    return p + [SEP] if protocol == "sep" else p


def allowed_ids(tok: M2Tokenizer):
    return [EOS] + [i for i in range(4, 101)]


@torch.inference_mode()
def generate(model, tok: M2Tokenizer, prompt: str, protocol: str, max_new=12):
    """严格按指定推理协议生成；模型状态管理完整恢复。"""
    was = model.training
    old_load = getattr(model, "use_router_load_update", None)
    old_grad = getattr(model, "router_gradient_path_enabled", None)
    model.eval()
    if old_load is not None:
        model.use_router_load_update = False
    if old_grad is not None:
        model.router_gradient_path_enabled = False
    inp = encode_inference(tok, prompt, protocol)
    logits, state = model(torch.tensor([inp]))
    allowed = torch.tensor(allowed_ids(tok))
    out = []
    finish = "max_tokens"
    for _ in range(max_new):
        tail = logits[0, -1].index_select(0, allowed)
        tid = int(allowed[int(torch.argmax(tail))])
        if tid == EOS:
            finish = "eos"
            break
        out.append(tid)
        logits, state = model(torch.tensor([[tid]]), state)
    model.train(was)
    if old_load is not None:
        model.use_router_load_update = old_load
    if old_grad is not None:
        model.router_gradient_path_enabled = old_grad
    return {"text": tok.decode(out), "token_ids": out, "finish_reason": finish}
