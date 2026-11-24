"""M2 交互终端：加载最佳 checkpoint，SEP 协议自由生成，展示逐字符输出与动力学。

用法：
  python chat_m2.py                      # 交互模式
  python chat_m2.py "COPY ABCD"          # 单次
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from tokenizer_m2 import M2Tokenizer
from candidate_registry import build
from evaluate_m2 import generate

HERE = Path(__file__).resolve().parent
CHECK = HERE / "checkpoints"


def find_best_checkpoint():
    """优先用复验/初筛中 macro 最高的 best checkpoint。"""
    results_path = HERE / "M2_TRAINING_RESULTS.json"
    default_arch = "M2-E"
    if results_path.exists():
        try:
            res = json.loads(results_path.read_text(encoding="utf-8"))
            cands = []
            for key in ("stage3_recheck", "stage2_screen", "stage1_sep_ab"):
                for r in res.get(key, []):
                    if r.get("protocol") == "legacy":
                        continue
                    cands.append((r.get("best_score", 0), r["run_id"], r["architecture"]))
            if cands:
                cands.sort(reverse=True)
                return cands[0][1], cands[0][2]
        except (KeyError, json.JSONDecodeError):
            pass
    return "S2_M2-E", default_arch


def load(run_id, arch):
    tok = M2Tokenizer()
    model = build(arch, tok.vocab_size)
    p = CHECK / f"{run_id}_best.pt"
    source = "trained"
    if p.exists():
        blob = torch.load(p, map_location="cpu", weights_only=False)
        model.load_state_dict(blob["state_dict"])
    else:
        source = "random_init(checkpoint missing)"
    model.eval()
    return tok, model, source


def main():
    torch.set_num_threads(4)
    if len(sys.argv) > 1:
        prompts = [sys.argv[1]]
    else:
        print("M2 交互终端（输入空行退出）。SEP 协议：prompt + [SEP] -> 答案。")
        prompts = None
    run_id, arch = find_best_checkpoint()
    tok, model, source = load(run_id, arch)
    print(f"[模型] {arch}  checkpoint={run_id}_best.pt  来源={source}")
    while True:
        if prompts is None:
            try:
                prompt = input("\n> ").strip()
            except EOFError:
                break
            if not prompt:
                break
        else:
            prompt = prompts[0]
        g = generate(model, tok, prompt, max_new=14, collect_states=True)
        print(f"  输出: {g['text']!r}")
        print(f"  tokens={g['token_ids']}  finish={g['finish_reason']}  "
              f"unique={len(set(g['token_ids']))}/{max(1,len(g['token_ids']))}")
        if prompts is not None:
            break
    print("再见。")


if __name__ == "__main__":
    main()
