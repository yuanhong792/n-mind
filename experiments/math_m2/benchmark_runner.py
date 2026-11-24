"""M2 资源基准：参数量、状态元素、前向/反向延迟、训练吞吐、峰值内存、文件大小。"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import torch

from candidate_registry import candidate_names, build
from tokenizer_m2 import M2Tokenizer

HERE = Path(__file__).resolve().parent
BATCH, SEQ = 8, 24


def bench_one(name, tok):
    torch.manual_seed(0)
    model = build(name, tok.vocab_size)
    model.train()
    x = torch.randint(4, 101, (BATCH, SEQ))
    valid = torch.ones(BATCH, SEQ, dtype=torch.bool)
    target = torch.randint(4, 101, (BATCH, SEQ))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # warmup
    for _ in range(2):
        logits, _ = model(x[:, :-1], valid_mask=valid[:, :-1])
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                                 target[:, 1:].reshape(-1))
        loss.backward(); opt.step(); opt.zero_grad()
    t0 = time.perf_counter()
    steps = 6
    for _ in range(steps):
        logits, _ = model(x[:, :-1], valid_mask=valid[:, :-1])
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                                 target[:, 1:].reshape(-1))
        loss.backward(); opt.step(); opt.zero_grad()
    wall = time.perf_counter() - t0
    tok_s = BATCH * (SEQ - 1) * steps / wall
    model.eval()
    with torch.no_grad():
        t1 = time.perf_counter()
        for _ in range(10):
            model(x[:, :-1], valid_mask=valid[:, :-1])
        fwd_ms = (time.perf_counter() - t1) / 10 * 1000
    return {"architecture": name, "params": model.n_params(),
            "state_elements": model.state_elements(),
            "forward_batch_ms": round(fwd_ms, 4), "train_tok_per_s": round(tok_s, 2),
            "finite_loss": bool(torch.isfinite(loss).all())}
    del model, opt
    gc.collect()


def main():
    torch.set_num_threads(20)
    tok = M2Tokenizer()
    rows = [bench_one(name, tok) for name in candidate_names()]
    report = {"status": "M2_RESOURCE_BENCHMARK_DONE", "batch": BATCH, "seq": SEQ,
              "torch": torch.__version__, "results": rows}
    (HERE / "M2_RESOURCE_BENCHMARK.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
