"""BlockRouter：Top-K 合法性、误差是否真的进入打分、反塌缩偏置的符号与边界。"""

from __future__ import annotations

import torch

from nmind.core.router import BlockRouter

B, Z, DH, DW, N = 4, 8, 6, 5, 8


def _router(seed: int = 0) -> BlockRouter:
    torch.manual_seed(seed)
    return BlockRouter(Z, DH, DW, N, hidden=16)


def _x(seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(B, Z, generator=g)
    h = torch.randn(B, N, DH, generator=g)
    w = torch.randn(B, DW, generator=g)
    return z, h, w


def test_topk_indices_are_valid_and_unique():
    r = _router()
    z, h, w = _x()
    res = r.route(z, h, w, torch.rand(B, 1), torch.rand(B, 1), k=3)
    assert res.indices.shape == (B, 3)
    for row in res.indices:
        assert len(set(row.tolist())) == 3          # 同一行不重复选块
        assert bool(((row >= 0) & (row < N)).all())
    assert torch.allclose(res.weights.sum(-1), torch.ones(B), atol=1e-6)


def test_k_is_clamped_into_range():
    r = _router()
    z, h, w = _x()
    assert r.route(z, h, w, torch.rand(B, 1), torch.rand(B, 1), k=0).k == 1
    assert r.route(z, h, w, torch.rand(B, 1), torch.rand(B, 1), k=N + 5).k == N


def test_stats_report_activation_ratio():
    r = _router()
    z, h, w = _x()
    res = r.route(z, h, w, torch.rand(B, 1), torch.rand(B, 1), k=2)
    assert res.stats["k"] == 2
    assert abs(res.stats["activation_ratio"] - 2 / N) < 1e-9
    assert len(res.stats["load"]) == N
    # load 的量纲是"每样本平均选中次数"（one-hot 掩码对 batch 求均值），
    # 所以它求和等于 k 而不是 1。写死 1/N 当目标会把所有块判成过载，
    # 反塌缩机制会静默失效——update_load_bias 的默认 target 必须跟着这个量纲走。
    assert abs(sum(res.stats["load"]) - 2.0) < 1e-6
    bal = r.update_load_bias(torch.tensor(res.stats["load"]))
    assert abs(bal["target"] - 2.0 / N) < 1e-6


def test_prediction_error_reaches_the_scores():
    """"误差高 ⇒ 多算"这条假说的**前提**是误差真的进了打分；先钉住前提，
    假说本身成不成立交给 ablations 的 no_error_gating 对照。"""
    r = _router().eval()
    z, h, w = _x()
    with torch.no_grad():
        low, _ = r.scores(z, h, w, torch.zeros(B, 1), torch.zeros(B, 1))
        high, _ = r.scores(z, h, w, torch.ones(B, 1) * 2.0, torch.ones(B, 1) * 2.0)
    assert not torch.allclose(low, high)
    # 同一个入口还得能反向传播到打分网络
    r2 = _router().train()
    z, h, w = _x(seed=2)
    res = r2.route(z, h, w, torch.rand(B, 1), torch.rand(B, 1), k=2)
    res.scores.sum().backward()
    grad = r2.score_mlp[0].weight.grad
    assert grad is not None and float(grad.abs().sum()) > 0.0


def test_load_bias_moves_against_the_load_and_stays_bounded():
    r = _router()
    load = torch.tensor([0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    first = r.update_load_bias(load)
    assert first["collapsed"] is True               # 过半块零负载 ⇒ 判定塌缩
    b = r.load_bias
    assert float(b[0]) < 0.0                        # 过载块被压
    assert float(b[2]) > 0.0                        # 空载块被抬
    for _ in range(200):
        r.update_load_bias(load)
    assert float(b.abs().max()) <= r.balance_max + 1e-9   # 硬边界，绝不无限漂移


def test_load_bias_is_not_a_trainable_parameter():
    r = _router()
    names = [n for n, _ in r.named_parameters()]
    assert "load_bias" not in names
    assert "load_bias" in dict(r.named_buffers())


def test_hard_selection_does_not_claim_fake_gradients():
    """Top-K 的离散索引本身没有梯度；只有 probs/scores 承载学习信号。"""
    r = _router()
    z, h, w = _x(seed=5)
    res = r.route(z, h, w, torch.rand(B, 1), torch.rand(B, 1), k=2)
    assert not res.indices.requires_grad
    assert res.probs.requires_grad and res.weights.requires_grad
