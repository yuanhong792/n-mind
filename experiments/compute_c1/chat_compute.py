"""N-MIND 计算能力交互程序。

输入示例：
  COPY ABC
  a=5\nb=8\nGET a
  ADD 3 7
输出：N-MIND 原始预测（生成 token、耗时、结束原因）与解释器验证结果严格分栏；
解释器答案绝不覆盖错误预测。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from tokenizer_compute import ComputeTokenizer
from model_compute import ComputeConfig, ComputeLM, load_compute
from evaluate_compute import generate
from environment import ComputeEnvironment

HERE = Path(__file__).resolve().parent


def run_line(model, tok, env, prompt: str):
    t0 = time.perf_counter()
    g = generate(model, tok, prompt)
    row = {"prompt": prompt, "answer": ""}  # 交互无标准答案；只做语法/可执行验证
    fb = env.verify(row, g["text"])
    return {
        "input": prompt,
        "neural": {"raw_prediction": g["text"],
                   "generated_tokens": g["token_ids"],
                   "finish_reason": g["finish_reason"],
                   "inference_seconds": g["seconds"]},
        "interpreter": {"syntax_valid": fb.syntactically_valid,
                        "executed": fb.executed,
                        "error": fb.error,
                        "note": "解释器只验证，不覆盖模型输出"},
        "wall_seconds": round(time.perf_counter() - t0, 5),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="candidate_compute_best.pt")
    args = ap.parse_args()
    torch.set_num_threads(20)
    tok = ComputeTokenizer()
    ckpt = HERE / args.checkpoint
    if not ckpt.exists():
        print(f"checkpoint 不存在: {ckpt}")
        return
    model, blob = load_compute(ckpt, tok)
    model.eval()
    env = ComputeEnvironment()
    print(f"N-MIND 计算交互｜checkpoint step={blob['optimizer_steps_executed']}｜输入空行退出")
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if not line:
            break
        # 多行程序：以 \n 字面量换行
        prompt = line.replace("\\n", "\n")
        result = run_line(model, tok, env, prompt)
        print("--- N-MIND 预测 ---")
        print(json.dumps(result["neural"], ensure_ascii=False))
        print("--- 解释器验证（独立分栏）---")
        print(json.dumps(result["interpreter"], ensure_ascii=False))
        print(f"耗时 {result['wall_seconds']}s\n")


if __name__ == "__main__":
    main()
