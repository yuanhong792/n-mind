"""资源基准：参数、状态、CPU前向/生成/训练吞吐、峰值内存。不消耗训练预算。"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import psutil
import torch

from candidate_registry import build_model

HERE = Path(__file__).resolve().parent


def _batch(tok, rows, n=8):
    seqs = [tok.encode(r["prompt"]) + tok.encode(r["answer"]) + [2] for r in rows[:n]]
    T = max(len(s) for s in seqs)
    x = torch.zeros(len(seqs), T, dtype=torch.long)
    valid = torch.zeros(len(seqs), T, dtype=torch.bool)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s)
        valid[i, :len(s)] = True
    return x, valid


def benchmark_model(name, tok, rows, threads=4):
    torch.set_num_threads(threads)
    proc = psutil.Process()
    gc.collect()
    model = build_model(name, tok.vocab_size)
    model.eval()
    x, valid = _batch(tok, rows)

    # 前向延迟
    with torch.inference_mode():
        for _ in range(3):
            model(x[:, :-1], valid_mask=valid[:, :-1])
        t0 = time.perf_counter()
        for _ in range(20):
            model(x[:, :-1], valid_mask=valid[:, :-1])
        fwd_ms = (time.perf_counter() - t0) / 20 * 1000

    # 自回归生成延迟（单样本）
    ids = tok.encode(rows[0]["prompt"])
    with torch.inference_mode():
        logits, state = model(torch.tensor([ids]))
        t0 = time.perf_counter()
        for _ in range(8):
            tid = int(torch.argmax(logits[0, -1]))
            logits, state = model(torch.tensor([[tid]]), state)
        gen_s = time.perf_counter() - t0

    # 训练吞吐（一次 forward+backward，不做 optimizer.step）
    model.train()
    model.zero_grad(set_to_none=True)
    t0 = time.perf_counter()
    logits, _ = model(x[:, :-1], valid_mask=valid[:, :-1])
    logits.mean().backward()
    train_s = time.perf_counter() - t0

    return {
        "architecture": name,
        "trainable_params": int(model.n_params()),
        "state_elements": int(model.state_elements()),
        "model_config": model.config_dict(),
        "forward_batch_ms": round(fwd_ms, 3),
        "per_token_forward_ms": round(fwd_ms / max(1, x.size(1) - 1) / x.size(0), 5),
        "generation_tokens_per_second": round(8 / max(1e-9, gen_s), 2),
        "train_forward_backward_seconds": round(train_s, 4),
        "train_tokens_per_second": round(x.numel() / max(1e-9, train_s), 2),
        "peak_rss_mb": round(proc.memory_info().rss / 1024**2, 2),
        "peak_cuda_mb": "NOT_AVAILABLE",
    }


def benchmark_all(tok, rows, names=None):
    names = names or list(__import__("candidate_registry").candidate_names())
    return {n: benchmark_model(n, tok, rows) for n in names}
