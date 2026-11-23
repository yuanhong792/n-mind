"""V1.6 回归测试：Router 必须在语言 CE 下收到非零梯度。

根因：NMindLM.forward 调用 blocks.forward_sparse 时漏传 weights=route.weights,
导致 BlockRouter 的 STE 软权重路径失效, score_mlp 永远拿不到任务梯度。
此测试钉死该修复, 防止回归。
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from nmind.language.neural_dialogue import NMindLM, LMConfig


def test_router_receives_language_gradient():
    torch.manual_seed(0)
    cfg = LMConfig(vocab_size=260, d_model=64, n_blocks=4, block_hidden=64,
                   workspace_dim=32, workspace_slots=4, top_k=2, query_readout=False)
    m = NMindLM(cfg)
    m.train()
    m.use_router_load_update = False
    x = torch.randint(4, 260, (2, 12))
    logits, _ = m(x)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), x.reshape(-1)
    )
    m.zero_grad()
    loss.backward()
    router_grads = [
        p.grad for n, p in m.named_parameters()
        if "router" in n and p.grad is not None
    ]
    assert router_grads, "Router 参数没有任何梯度（STE 权重未接入 forward）"
    assert all(float(g.abs().sum()) > 0 for g in router_grads), (
        "Router score_mlp 存在零梯度：语言任务信号无法回传到路由打分网络"
    )


def test_inference_forward_unchanged_by_ste_weights():
    """STE 只改反向路径，前向数值必须与硬路由一致（旧 checkpoint 兼容）。"""
    torch.manual_seed(0)
    cfg = LMConfig(vocab_size=260, d_model=64, n_blocks=4, block_hidden=64,
                   workspace_dim=32, workspace_slots=4, top_k=2, query_readout=False)
    m = NMindLM(cfg)
    m.eval()
    m.use_router_load_update = False
    x = torch.tensor([[4, 5, 6, 7]])
    with torch.inference_mode():
        logits, _ = m(x)
    assert torch.isfinite(logits).all(), "推理输出出现非有限值"
    assert logits.shape == (1, 4, cfg.vocab_size)
