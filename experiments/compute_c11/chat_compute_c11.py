"""N-MIND C1.1 计算能力交互终端。

默认加载 candidate_c11_best.pt。输入空行退出；多行程序用字面量 \\n 分隔。
支持：COPY/ECHO/PRINT/SHOW、REVERSE、INDEX、变量赋值与 GET（含值复制语义）。
标准答案由独立规则函数计算，只用于显示验证，绝不回写或纠正神经网络输出。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
C1 = HERE.parent / "compute_c1"
sys.path.insert(0, str(C1))
sys.path.insert(0, str(HERE))

from tokenizer_compute import ComputeTokenizer, EOS
from model_compute import load_compute


def independent_answer(prompt: str) -> tuple[str | None, str]:
    """仅提供标准答案；与模型推理完全隔离。"""
    lines = [ln.strip() for ln in prompt.splitlines() if ln.strip()]
    if not lines:
        return None, "EMPTY_PROMPT"

    first = lines[0].split()
    if first and first[0].upper() in {"COPY", "ECHO", "PRINT", "SHOW"} and len(first) >= 2:
        return first[1], "copy_family"
    if first and first[0].upper() == "REVERSE" and len(first) >= 2:
        return first[1][::-1], "reverse"
    if first and first[0].upper() == "INDEX" and len(first) >= 3:
        seq, idx_s = first[1], first[2]
        if idx_s.isdigit() and 1 <= int(idx_s) <= len(seq):
            return seq[int(idx_s) - 1], "index"

    env: dict[str, str] = {}
    for ln in lines:
        if ln.startswith("GET "):
            var = ln.split()[1]
            return env.get(var), "variable_current_or_copied"
        if "=" not in ln:
            return None, "UNSUPPORTED_FORMAT"
        var, expr = [x.strip() for x in ln.split("=", 1)]
        if expr.isdigit():
            env[var] = expr
        elif expr in env:
            # C1.1 语义：b=a 是"值复制"，之后 a 改写不影响 b
            env[var] = env[expr]
        else:
            return None, "UNDEFINED_VARIABLE"
    return None, "NO_QUERY"


def generate_timed(model, tok, prompt: str, max_new: int = 12) -> dict:
    was_train = model.training
    old_load = getattr(model, "use_router_load_update", False)
    model.eval()
    model.use_router_load_update = False

    ids = tok.encode(prompt)
    t0 = time.perf_counter()
    with torch.inference_mode():
        logits, state = model(torch.tensor([ids]))
    prompt_seconds = time.perf_counter() - t0

    allowed = torch.tensor([EOS] + list(range(4, tok.vocab_size)))
    out: list[int] = []
    t1 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(max_new):
            tail = logits[0, -1].index_select(0, allowed)
            tid = int(allowed[int(torch.argmax(tail))])
            if tid == EOS:
                finish = "eos"
                break
            out.append(tid)
            logits, state = model(torch.tensor([[tid]]), state)
        else:
            finish = "max_tokens"
    gen_seconds = time.perf_counter() - t1

    model.train(was_train)
    model.use_router_load_update = old_load
    return {
        "raw_output": tok.decode(out),
        "token_ids": out,
        "generated_token_count": len(out),
        "prompt_token_count": len(ids),
        "prompt_seconds": round(prompt_seconds, 6),
        "generation_seconds": round(gen_seconds, 6),
        "finish_reason": finish,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="candidate_c11_best.pt")
    args = ap.parse_args()
    torch.set_num_threads(20)

    tok = ComputeTokenizer()
    ckpt = HERE / args.checkpoint
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {ckpt}")
    model, blob = load_compute(ckpt, tok)
    model.eval()
    print(f"N-MIND C1.1 交互终端｜checkpoint={ckpt.name}｜step={blob['optimizer_steps_executed']}")
    print("输入空行退出；多行程序用字面量 \\n，例如：a=3\\nb=a\\na=5\\nGET b")
    print("-" * 60)

    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if not line:
            break
        prompt = line.replace("\\n", "\n")
        neural = generate_timed(model, tok, prompt)
        expected, source = independent_answer(prompt)
        result = {
            "prompt": prompt,
            "neural": neural,
            "verification": {
                "expected": expected,
                "source": source,
                "matches": (expected is not None and neural["raw_output"].strip() == expected),
                "note": "标准答案仅显示，不参与也不纠正模型输出",
            },
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("-" * 60)


if __name__ == "__main__":
    main()
