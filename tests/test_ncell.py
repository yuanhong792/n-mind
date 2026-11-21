"""N-Cell：稠密与稀疏执行等价、直通估计、形状校验。

这里钉死的是"稀疏执行只是省算力的等价改写"这一条：
如果两者数值不等价，后面所有"块稀疏"的结论都作废，所以单独测。
"""

from __future__ import annotations

import pytest
import torch

from nmind.core.ncell import NCell

Z, DH, DW, N, B, K = 8, 6, 5, 4, 3, 2


def _cell(seed: int = 0) -> NCell:
    torch.manual_seed(seed)
    return NCell(Z, DH, DW, N)


def _inputs(seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(B, Z, generator=g)
    h = torch.randn(B, N, DH, generator=g)
    w = torch.randn(B, DW, generator=g)
    idx = torch.stack([torch.randperm(N, generator=g)[:K] for _ in range(B)])
    return z, h, w, idx


def test_initial_state_is_zeros():
    h0 = NCell.initial_state(B, N, DH)
    assert h0.shape == (B, N, DH)
    assert torch.count_nonzero(h0) == 0


def test_dense_and_sparse_are_equivalent():
    """同一批输入：硬掩码稠密路径与只算选中块的稀疏路径必须逐位一致。"""
    cell = _cell()
    z, h, w, idx = _inputs()
    mask = torch.zeros(B, N)
    mask.scatter_(-1, idx, 1.0)
    h_dense, g_dense, m_dense, _p = cell(z, h, w, mask)
    h_sparse, g_sel, m_sparse, _ps, n_pairs = cell.forward_sparse(z, h, w, idx)
    assert torch.allclose(h_dense, h_sparse, atol=1e-6), (h_dense - h_sparse).abs().max()
    assert torch.equal(m_dense, m_sparse)
    assert n_pairs == B * K                     # 只计算 (样本, 块) 对，不多算
    assert g_sel.shape == (B * K, DH)


def test_unselected_blocks_are_untouched():
    cell = _cell()
    z, h, w, idx = _inputs(seed=7)
    mask = torch.zeros(B, N)
    mask.scatter_(-1, idx, 1.0)
    h_new, *_ = cell(z, h, w, mask)
    untouched = (mask == 0)
    assert torch.equal(h_new[untouched], h[untouched])


def test_gate_zero_keeps_old_state():
    """g=0 时 h' 必须完全等于旧状态：这是"门控循环"的定义，不是近似。"""
    cell = _cell()
    z, h, w, idx = _inputs(seed=3)
    mask = torch.ones(B, N)
    with torch.no_grad():                        # 让 gate 输出 0：把权重清零 + 大负偏置
        cell.W_g.zero_()
        cell.b_g.fill_(-30.0)
    h_new, g, *_ = cell(z, h, w, mask)
    assert float(g.detach().max()) < 1e-6
    assert torch.allclose(h_new, h, atol=1e-5)


def test_straight_through_gives_router_gradient():
    """训练态 + 软掩码：离散 Top-K 的梯度不该为 0（直通估计的意义所在）。"""
    cell = _cell().train()
    z, h, w, idx = _inputs(seed=5)
    soft = torch.zeros(B, N, requires_grad=True)
    mask = torch.zeros(B, N)
    mask.scatter_(-1, idx, 1.0)
    soft = soft.detach().scatter_(-1, idx, 0.5).requires_grad_(True)
    h_new, *_ = cell(z, h, w, mask, soft=soft)
    h_new.sum().backward()
    assert soft.grad is not None
    assert float(soft.grad.abs().sum()) > 0.0    # 梯度确实回传到软选择上
    # 直通估计让梯度落到 softm 的**每一格**（未选中格拿到 (h_prop-h_prev) 的梯度），
    # 这正是负载均衡损失能训到路由打分网络的原因；不要误以为梯度只走选中块。
    assert float(soft.grad[torch.arange(B)[:, None], idx].abs().sum()) > 0.0


def test_sparse_soft_weights_are_ste():
    """weights 只存在于反向：前向恒为硬采纳（与 weights=None 一致，容差 1e-6），
    梯度按 (1-w)·旧 + w·全更新 的插值路径回传（M2 收尾 F10 的语义，2026-11-21 起）。"""
    cell = _cell()
    z, h, w, idx = _inputs(seed=9)
    rows = torch.arange(B)[:, None].expand(-1, K).reshape(-1)
    cols = idx.reshape(-1)
    h_hard = cell.forward_sparse(z, h, w, idx)[0][rows, cols]
    for wv in (0.0, 0.5, 1.0):
        h_w = cell.forward_sparse(z, h, w, idx, weights=torch.full((B, K), wv))[0][rows, cols]
        assert torch.allclose(h_w, h_hard, atol=1e-6), f"weights={wv} 不应改变前向数值"
    # 反向通路存在：对输出加权和反传后，至少一个参数拿到非零梯度
    # （router 级别的任务梯度断言在 test_router_ste.py 整模型验证）。
    h_out, *_ = cell.forward_sparse(z, h, w, idx, weights=torch.full((B, K), 0.5))
    (h_out * torch.randn_like(h_out)).sum().backward()
    assert any(p.grad is not None and float(p.grad.pow(2).sum()) > 0
               for p in cell.parameters() if p.requires_grad)


def test_shape_errors_are_loud():
    """形状不对必须立刻报错，而不是靠广播静默算出错东西。"""
    cell = _cell()
    z, h, w, idx = _inputs()
    with pytest.raises(ValueError):
        cell(torch.randn(B, Z + 1), h, w, torch.ones(B, N))
    with pytest.raises(ValueError):
        cell(z, torch.randn(B, N + 1, DH), w, torch.ones(B, N + 1))
    with pytest.raises(ValueError):
        cell(z, h, torch.randn(B, DW + 3), torch.ones(B, N))
    with pytest.raises(ValueError):
        cell(z, h, w, torch.ones(B, N + 2))


def test_probs_are_normalized_over_selected_blocks():
    """probs 是被选中块之间均分的概率质量：每行和为 1，未选中为 0。"""
    cell = _cell()
    z, h, w, idx = _inputs(seed=11)
    mask = torch.zeros(B, N)
    mask.scatter_(-1, idx, 1.0)
    *_rest, probs = cell(z, h, w, mask)
    assert torch.allclose(probs.sum(-1), torch.ones(B), atol=1e-6)
    assert torch.equal(probs[mask == 0], torch.zeros(int((mask == 0).sum())))
    *_rest2, probs2, _n = cell.forward_sparse(z, h, w, idx)
    assert probs2.shape == (B, N)
    assert torch.allclose(probs2.sum(-1), torch.ones(B), atol=1e-6)
