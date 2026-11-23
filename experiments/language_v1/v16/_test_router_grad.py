"""V1.6 失败测试：Router 必须在语言 CE 下收到非零梯度。

预期：修复前 FAIL（router.score_mlp.* 无梯度），修复后 PASS。
只检查梯度，不执行 optimizer.step，不修改权重。
"""
from __future__ import annotations
import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
import torch
from nmind.language.neural_dialogue import NMindLM, LMConfig
from nmind.language.tokenizer_v2 import ByteBPETokenizer

def build_model():
    cfg = LMConfig(vocab_size=260, d_model=64, n_blocks=4, block_hidden=64,
                   workspace_dim=32, workspace_slots=4, top_k=2, query_readout=False)
    return NMindLM(cfg)

def main():
    torch.set_num_threads(1)
    m = build_model(); m.train(); m.use_router_load_update = False
    x = torch.randint(4, 260, (2, 12))
    logits, _ = m(x)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), x.reshape(-1))
    m.zero_grad(); loss.backward()
    rg = {n: (None if p.grad is None else float(p.grad.abs().sum()))
          for n, p in m.named_parameters() if "router" in n}
    nonzero = all(v is not None and v > 0 for v in rg.values())
    print("router grads:", {k: (round(v,6) if v is not None else None) for k,v in rg.items()})
    if nonzero:
        print("RESULT: PASS - Router 收到非零梯度")
    else:
        print("RESULT: FAIL - Router score_mlp 无梯度（STE 软权重未接入 forward）")
        sys.exit(1)

if __name__ == "__main__":
    main()
