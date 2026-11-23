"""Offline Language V1 preflight; never calls optimizer.step."""
from __future__ import annotations
import json, platform, sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from nmind.language.tokenizer_v2 import ByteBPETokenizer
from nmind.language.neural_language_v2 import LanguageV2Config, LanguageV2Model

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "language_v1"

def main():
    texts = ["你好，世界。", "苹果在桌子上。", "Python可以处理中文。"]
    tok = ByteBPETokenizer.train(texts, vocab_size=4096)
    model = LanguageV2Model(LanguageV2Config(vocab_size=tok.vocab_size, d_model=32, n_blocks=2, block_hidden=32, workspace_dim=16, workspace_slots=2))
    model.eval(); ids = torch.tensor([tok.encode("你好，世界。", add_eos=True)], dtype=torch.long)
    t0 = time.perf_counter()
    with torch.inference_mode(): logits, state = model(ids)
    elapsed = time.perf_counter() - t0
    result = {"status":"PASS_ZERO_STEP_PREFLIGHT", "optimizer_steps":0, "training_executed":False,
      "python":sys.version, "platform":platform.platform(), "torch":torch.__version__,
      "cuda_available":torch.cuda.is_available(), "cuda_device":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
      "tokenizer":{"vocab_size":tok.vocab_size,"hash":tok.vocab_hash(),"roundtrip":tok.decode(tok.encode(texts[0]))==texts[0]},
      "model_parameters":sum(p.numel() for p in model.parameters()), "forward_seconds":elapsed,
      "stable_checkpoint_modified":False, "network_requests":0, "formal_training":"PENDING_USER_APPROVAL"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"LANGUAGE_V1_PREFLIGHT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))

if __name__ == "__main__": main()
