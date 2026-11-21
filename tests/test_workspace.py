"""GlobalWorkspace：固定容量、竞争写入、溢出驱逐与反向图安全。

这里最容易写出"看起来对"的 bug 有三类，都有对应测试：
1. 融合时把多条写入直接相加（等于求平均后淹没）；
2. 在带计算图的张量上做就地索引赋值（破坏反向图）；
3. 声称"容量是硬上限"却只腾出一个槽 —— 本会话真实踩过，见
   test_capacity_is_a_hard_limit_with_overflow_records。
"""

from __future__ import annotations

import torch

from nmind.core.workspace import GlobalWorkspace

B, Z, DW, S = 2, 8, 6, 4


def _ws(seed: int = 0, **kw) -> GlobalWorkspace:
    torch.manual_seed(seed)
    return GlobalWorkspace(z_dim=Z, d_w=DW, slots=S, **kw)


def _flat_state(occupied: float) -> dict:
    return {"content": torch.zeros(B, S, DW), "occupied": torch.full((B, S), occupied),
            "utility": torch.zeros(B, S)}


def test_initial_state_is_empty_and_read_is_uniform():
    ws = _ws()
    st = ws.initial_state(B)
    assert st["content"].shape == (B, S, DW)
    assert float(st["occupied"].abs().sum()) == 0.0
    r = ws.read(st, torch.randn(B, Z))
    assert torch.allclose(r.attn, torch.full((B, S), 1.0 / S), atol=1e-6)  # 全空时保持均匀
    assert torch.allclose(r.value, torch.zeros(B, DW), atol=1e-7)


def test_read_masks_empty_slots_when_some_are_occupied():
    """空槽必须被明显压低注意力。

    注意：一次 write 会把 1 单位质量按 softmax 摊到**所有**槽，没有哪个槽会正好是空的，
    所以这里手工构造"只有 1 号槽有内容"的状态，而不是指望写一次自然留有空槽。
    """
    ws = _ws()
    st = ws.initial_state(B)
    st["content"][:, 1] = torch.randn(B, DW)
    st["occupied"][:, 1] = 1.0
    r = ws.read(st, torch.randn(B, Z))
    empty = [s for s in range(S) if s != 1]
    assert bool((r.attn[:, 1] > r.attn[:, empty].max(dim=-1).values).all())
    assert float(r.attn[:, empty].max()) < 0.25
    assert torch.allclose(r.attn.sum(-1), torch.ones(B), atol=1e-6)


def test_assignment_mass_is_normalized_and_competition_needs_heterogeneous_occupancy():
    ws = _ws()
    st = ws.initial_state(B)
    a = torch.randn(1, 2, DW)
    wr1 = ws.write(st, a.expand(B, 2, DW).contiguous())
    assert torch.allclose(wr1.assignments.sum(-1), torch.ones(B, 2), atol=1e-5)

    # softmax 对整体平移不变：所有槽同样拥挤时，降权项被完全抵消，
    # 分配结果与空工作空间逐位相同。旧断言"拥挤时最大分配质量下降"其实是假的。
    wr_flat = ws.write(_flat_state(3.0), a.expand(B, 2, DW).contiguous())
    assert torch.allclose(wr_flat.assignments, wr1.assignments, atol=1e-6)

    # 只有**异质**占用才产生竞争：把空场时最吃香的槽单独灌满，
    # 它拿到的质量必须下降，且下降的那部分必须转到别的槽（总量仍是每写入 1）。
    mass_empty = wr1.assignments.sum(dim=1)                       # [B,S] 每个槽收到多少写入
    crowded_slot = int(torch.argmax(mass_empty[0]).item())
    occ = torch.zeros(B, S)
    occ[:, crowded_slot] = 3.0
    wr2 = ws.write({"content": torch.zeros(B, S, DW), "occupied": occ,
                    "utility": torch.zeros(B, S)}, a.expand(B, 2, DW).contiguous())
    mass_crowded = wr2.assignments.sum(dim=1)
    assert float(mass_crowded[:, crowded_slot].max()) < float(mass_empty[:, crowded_slot].min())
    others = [s for s in range(S) if s != crowded_slot]
    assert float(mass_crowded[:, others].min()) > float(mass_empty[:, others].max())
    assert torch.allclose(mass_crowded.sum(-1), mass_empty.sum(-1), atol=1e-5)


def test_capacity_is_a_hard_limit_with_overflow_records():
    """硬上限 = 腾完之后**每个槽**都 <= capacity，不是"这步至少腾了一个"。

    旧实现每个拥挤样本只腾一个槽，多个槽同时超容量时会留下 occupied≈2.0 > capacity=1.0，
    模块文档里"容量是硬上限"那句话就成了谎话。这条测试就是用来钉死它的。
    """
    ws = _ws(capacity=1.0)
    st = {"content": torch.zeros(B, S, DW), "occupied": torch.full((B, S), 0.9),
          "utility": torch.zeros(B, S)}
    wr = ws.write(st, torch.randn(B, 3, DW) * 3.0)
    assert float(wr.occupied.max()) <= 1.0 + 1e-6
    assert len(wr.overflow) >= 1
    mass = 0.0
    for item in wr.overflow:
        assert set(item) == {"batch", "slot", "content", "occupied", "evicted_mass"}
        assert len(item["content"]) == DW
        assert item["occupied"] <= 1.0 + 1e-6
        assert item["evicted_mass"] > 0.0
        mass += item["evicted_mass"]
        # 腾出后仍保留内容（不是清零）：清零会让下一步读数凭空变成"从没写过"
        assert float(wr.state["content"][item["batch"], item["slot"]].abs().sum()) > 0.0
    # 质量守恒：腾走的量 = 融合后未腾出的总占用 - 腾完的总占用。
    # 未腾出的总占用 = ws.decay*旧占用 + 写入条数（默认 decay=0.9，别按 1.0 算）。
    unevicted = 0.9 * ws.decay * S * B + 3 * B
    assert abs(mass - (unevicted - float(wr.occupied.sum()))) < 1e-4


def test_occupancy_is_counted_in_write_items_not_content_magnitude():
    """占用度单位是"条目"：总占用 = decay*旧占用 + 写入条数，与内容幅度无关。

    这是容量语义的地基。如果占用随幅度漂移，那么换个 d_w 或换个初始化尺度，
    配置里写的 capacity=1.5 含义就悄悄变了，所有容量结论作废。
    """
    ws = _ws(decay=0.5, capacity=8.0)          # 容量放大，单独看守恒不掺腾出
    st = ws.initial_state(B)
    wr1 = ws.write(st, torch.randn(B, 1, DW))
    assert abs(float(wr1.occupied.sum(dim=-1).mean()) - 1.0) < 1e-5
    wr2 = ws.write({"content": wr1.state["content"].detach(),
                    "occupied": wr1.occupied.detach(),
                    "utility": wr1.utility.detach()}, torch.zeros(B, 1, DW))
    assert abs(float(wr2.occupied.sum(dim=-1).mean()) - 1.5) < 1e-5      # 0.5*1 + 1
    big = ws.write(st, torch.randn(B, 1, DW) * 100.0)
    assert abs(float(big.occupied.sum(dim=-1).mean()) - 1.0) < 1e-5      # 幅度不改占用

    # 融合是"按占用度加权平均"，所以写一条**空内容但强度为 1**的写入会按已知比例稀释旧内容：
    # 方向完全保持、幅度乘上 旧质量/(旧质量+新质量)。这不是失忆，但也不是无损，
    # 文档不许把"槽里还有东西"说成"旧内容没受影响"。
    old_mass = wr1.occupied.detach() * ws.decay                            # [B,S]
    new_mass = wr2.assignments.detach().sum(dim=1)                          # [B,S]，每条强度 1
    denom = (old_mass + new_mass).clamp_min(1e-6)
    expected = wr1.state["content"].detach() * old_mass.unsqueeze(-1) / denom.unsqueeze(-1)
    assert torch.allclose(wr2.state["content"].detach(), expected, atol=1e-6)
    # 方向不变要**逐槽**看：各槽的稀释比例不同，把整张 [B,S,d] 拍平算余弦会把
    # "每个槽都和原来同向"错判成"整体不同向"。
    cos = torch.nn.functional.cosine_similarity(
        wr2.state["content"].detach(), wr1.state["content"].detach(), dim=-1)  # [B,S]
    assert float(cos.min()) > 0.9999
    assert (wr1.utility >= 0).all() and (wr1.utility <= wr1.occupied + 1e-6).all()


def test_zero_strength_write_is_a_true_no_op():
    """strength=0 必须既不占位也不改内容——否则"没写"和"写了个零向量"就分不开。"""
    ws = _ws(decay=0.5)
    st = ws.initial_state(B)
    wr1 = ws.write(st, torch.randn(B, 1, DW))
    state1 = {k: (v.detach().clone() if torch.is_tensor(v) else v)
              for k, v in wr1.state.items()}
    wr_noop = ws.write(state1, torch.randn(B, 1, DW), strengths=torch.zeros(B, 1))
    assert torch.allclose(wr_noop.state["content"], state1["content"], atol=1e-6)   # 内容没动
    assert abs(float(wr_noop.occupied.sum(dim=-1).mean()) - 0.5 * 1.0) < 1e-5       # 只衰减
    assert wr_noop.overflow == []
    # 形状错的 strengths 要当场报错，不能被静默广播成"少写几条"
    try:
        ws.write(state1, torch.randn(B, 2, DW), strengths=torch.ones(B, 1))
    except ValueError:
        pass
    else:
        raise AssertionError("strengths 形状错误没有被拒绝")


def test_write_does_not_mutate_caller_state():
    ws = _ws(capacity=0.5)
    st = _flat_state(0.4)
    st = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in st.items()}
    before = {k: v.clone() for k, v in st.items()}
    ws.write(st, torch.randn(B, 2, DW) * 5.0)
    for k in before:
        assert torch.equal(st[k], before[k]), f"state[{k!r}] 被写入过程就地改了"


def test_backward_survives_overflow_eviction():
    """回归测试：溢出处理曾用 index_put_ 破坏反向图（轻则报错、重则静默算错梯度）。"""
    ws = _ws(capacity=0.8)
    st = {"content": torch.zeros(B, S, DW, requires_grad=True),
          "occupied": torch.full((B, S), 0.7, requires_grad=True),
          "utility": torch.zeros(B, S)}
    src = torch.nn.Parameter(torch.randn(B, 3, DW))
    wr = ws.write(st, src)
    loss = wr.state["content"].sum() + wr.occupied.sum() + wr.utility.sum()
    loss.backward()                               # 不该抛 RuntimeError
    assert src.grad is not None and torch.isfinite(src.grad).all()
    assert st["occupied"].grad is None or torch.isfinite(st["occupied"].grad).all()
    assert float(src.grad.abs().sum()) > 0.0


def test_utility_is_bounded_by_occupancy():
    ws = _ws()
    wr = ws.write(ws.initial_state(B), torch.randn(B, 2, DW))
    assert (wr.utility >= 0).all() and (wr.utility <= wr.occupied + 1e-6).all()
    # 空工作空间的占用读数必须是 0，下游才能区分"没东西"和"有东西"
    st = ws.initial_state(B)
    assert torch.all(ws.read(st, torch.randn(B, Z)).occupied == 0)


def test_pressure_reports_capacity_not_throughput():
    ws = _ws(capacity=1.0)
    wr = ws.write(ws.initial_state(B), torch.randn(B, 2, DW))
    p = ws.pressure(wr.occupied)
    assert set(p) == {"mean_occupied", "crowded_slots", "used_slots", "capacity_slots"}
    assert p["capacity_slots"] == S


def test_bad_dims_are_rejected():
    ws = _ws()
    try:
        ws.write(ws.initial_state(B), torch.randn(B, 2, DW + 3))
    except ValueError as e:
        assert str(DW) in str(e)
    else:
        raise AssertionError("维度错误没有被拒绝")


def test_slots_must_be_positive():
    try:
        GlobalWorkspace(z_dim=Z, d_w=DW, slots=0)
    except ValueError:
        return
    raise AssertionError("slots=0 应该被拒绝")
