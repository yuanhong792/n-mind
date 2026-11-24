"""M4 本地推理入口：只加载真实 checkpoint，拒绝随机初始化冒充训练结果。

无梯度、无解释器、无硬编码答案；答案完全由 checkpoint 中的神经模型生成。
当前未授权训练，checkpoints/ 不存在，因此直接运行会明确报 NOT_AVAILABLE。

示例：
    python chat_m4.py --checkpoint checkpoints/main_R2_tiny_transformer_s1_best.pt \
        --prompt "COPY ab"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "models"))

from tokenizer_m4 import M4Tokenizer  # noqa: E402
from evaluate_m4 import generate  # noqa: E402
from gru_reference import GRUReferenceM3  # noqa: E402
from transformer_reference import TinyTransformerM3  # noqa: E402
from low_rank_interaction import LowRankInteractionM4  # noqa: E402
from selective_interaction import SelectiveInteractionM4  # noqa: E402

MODELS = {
    "R1_gru_reference": GRUReferenceM3,
    "R2_tiny_transformer": TinyTransformerM3,
    "M4-A_low_rank_interaction": LowRankInteractionM4,
    "M4-B_selective_interaction": SelectiveInteractionM4,
}
CFG = {
    "R1_gru_reference": {},
    "R2_tiny_transformer": {"d_model": 64, "nhead": 4, "layers": 2},
    "M4-A_low_rank_interaction": {"d_model": 64, "nhead": 4, "layers": 2, "r": 16},
    "M4-B_selective_interaction": {"d_model": 64, "nhead": 4, "layers": 2,
                                   "group_size": 4, "top_k": 2},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--arch", default=None)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--gen-limit", type=int, default=24)
    ap.add_argument("--soft-gate", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(20)
    ckpt = Path(args.checkpoint)
    if not ckpt.is_absolute():
        ckpt = HERE / ckpt
    if not ckpt.exists():
        raise SystemExit(f"NOT_AVAILABLE: 缺少真实 checkpoint：{ckpt}")

    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    meta = blob.get("meta", {})
    arch = args.arch or meta.get("arch")
    if arch not in MODELS:
        raise SystemExit(f"UNKNOWN_ARCH: {arch!r}; 可选 {sorted(MODELS)}")

    tok = M4Tokenizer()
    model = MODELS[arch](tok.vocab_size, **CFG[arch])
    model.load_state_dict(blob["model"], strict=True)
    model.eval()

    for key, tensor in model.state_dict().items():
        if not torch.equal(tensor, blob["model"][key]):
            raise RuntimeError(f"checkpoint 逐张量核验失败：{key}")

    with torch.inference_mode():
        out = generate(model, tok, args.prompt, gen_limit=args.gen_limit,
                       hard_gate=not args.soft_gate)

    print(json.dumps({"checkpoint": str(ckpt), "arch": arch, "role": meta.get("role"),
                      "step": meta.get("step"), "prompt": args.prompt,
                      "generated": out}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
