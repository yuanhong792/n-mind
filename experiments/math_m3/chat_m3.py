"""M3 交互终端：加载已训练 checkpoint，对任意输入做 SEP 协议自回归生成。

用法：
    python chat_m3.py "COPY ABCD"
    python chat_m3.py --arch M3-A --seed 1 --role best "REVERSE X4A"
不带参数进入交互模式。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "models"))

from tokenizer_m3 import M3Tokenizer  # noqa: E402
from evaluate_m3 import generate  # noqa: E402

MODELS = {
    "R1": ("R1_gru_reference", "gru_reference", "GRUReferenceM3"),
    "R2": ("R2_tiny_transformer", "transformer_reference", "TinyTransformerM3"),
    "M3-A": ("M3-A_low_rank_neural", "low_rank_neural", "LowRankNeuralM3"),
    "M3-B": ("M3-B_adaptive_computation", "adaptive_computation", "AdaptiveComputationM3"),
}


def load_model(arch_key: str, seed: int, role: str):
    import importlib
    key, mod_name, cls_name = MODELS[arch_key]
    mod = importlib.import_module(f"models.{mod_name}")
    cls = getattr(mod, cls_name)
    model = cls(102)
    ckpt = HERE / "checkpoints" / f"{key}_s{seed}_{role}.pt"
    if ckpt.exists():
        blob = torch.load(ckpt, weights_only=False)
        model.load_state_dict(blob["model"])
        model.eval()
        return model, f"{key}_s{seed}_{role} (loaded)"
    model.eval()
    return model, f"{key}_s{seed}_{role} (random, 未训练)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", nargs="?", help="输入 prompt")
    ap.add_argument("--arch", default="R2", choices=list(MODELS.keys()))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--role", default="best", choices=["best", "final"])
    args = ap.parse_args()

    torch.set_num_threads(20)
    tok = M3Tokenizer()
    model, source = load_model(args.arch, args.seed, args.role)
    print(f"[chat_m3] 模型: {source}  tokenizer_sha={tok.vocab_sha}")

    if args.input:
        gen = generate(model, tok, args.input)
        print(f"输入: {args.input!r}")
        print(f"输出: {gen['text']!r}  ({gen['finish_reason']})")
        return

    print("进入交互模式，输入 prompt 回车生成，空行退出。")
    while True:
        try:
            line = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            break
        gen = generate(model, tok, line)
        print(f"  -> {gen['text']!r}  ({gen['finish_reason']})")


if __name__ == "__main__":
    main()
