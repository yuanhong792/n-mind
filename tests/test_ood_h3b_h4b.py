"""H3b/H4b 生成器的正确性测试：布局合法性、可解性、非退化反转、确定性。

数据可信度先于模型结论——这些测试钉住 OOD 评测集本身。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nmind.learning import ood_layouts as O


def test_topology_families_produce_valid_layouts():
    for family in O.FAMILIES:
        rng = np.random.default_rng(1)
        cells = O.make_topology(family, rng)
        assert len(cells) > 0
        for (x, y) in cells:
            assert 0 <= x < 6 and 0 <= y < 6, f"{family} 越界 {(x, y)}"
        assert len(set(cells)) == len(cells), f"{family} 有重复障碍格"


def test_wall_gap_has_single_gap_on_x3():
    rng = np.random.default_rng(2)
    cells = O.make_topology("wall_gap", rng)
    xs = {x for x, _ in cells}
    assert xs == {3}, "wall_gap 的墙必须在 x=3"
    assert len(cells) == 5, "6 格墙去掉 1 个缺口应剩 5 格"


def test_solvable_bfs_correctness():
    wall = tuple((3, y) for y in range(6) if y != 2)     # 缺口在 y=2
    assert O.solvable((0, 0), (5, 5), wall)              # 经缺口可绕
    solid = tuple((3, y) for y in range(6))              # 无缺口：左右不通
    assert not O.solvable((0, 0), (5, 0), solid)
    assert O.solvable((0, 0), (2, 0), solid)             # 同侧可达


def test_h3b_wall_gap_episodes_have_opposite_sides():
    sp = O.make_h3b_split("dev", n=8, seed=31)
    wall_layout = next(tuple(map(tuple, a["layout"]))
                       for a in sp["manifest"]["audit"] if a["family"] == "wall_gap")
    wall_rows = [r for r in sp["rows"] if r["task"] == "h3b_wall_gap"]
    assert wall_rows
    for r in wall_rows:
        s, g = r["start"], r["goal"]
        assert (s[0] < 3) != (g[0] < 3), f"wall_gap 的 start/goal 必须分居 x=3 两侧：{s}/{g}"
        assert O.solvable(tuple(s), tuple(g), wall_layout), "必须经缺口可解"


def test_h3b_all_episodes_solvable_and_labeled():
    sp = O.make_h3b_split("holdout", n=16, seed=77)
    for r in sp["rows"]:
        family = r["task"].removeprefix("h3b_")
        layout = next(tuple(map(tuple, a["layout"]))
                      for a in sp["manifest"]["audit"] if a["family"] == family)
        assert r["obs"].shape[1] == O.OBS_DIM_WITH_MAP
        encoded_map = r["obs"][0, 26:].reshape(6, 6)
        assert {(x, y) for x in range(6) for y in range(6) if encoded_map[x, y] > 0.5} == set(layout)
        for t in range(r["len"]):
            pos = O.decode_pos(r["obs"][t])
            assert r["actions"][t] == O.shortest_path_action(
                pos, tuple(r["goal"]), layout), f"标签不是 BFS 最短路首步 @t={t}"
        assert any(O.decode_pos(obs) == tuple(r["goal"]) for obs in r["obs"]), \
            "每条 H3b 轨迹必须实际到达目标"


def test_h4b_episodes_are_non_degenerate():
    sp = O.make_h4b_split("holdout", n=6, seed=91)
    assert sp["manifest"]["conflict_frac_min"] >= 0.5, "H4b 每条遮挡段冲突率必须 ≥0.5"
    assert sp["manifest"]["coincident_frac_mean"] < 0.5
    for r in sp["rows"]:
        f, c = O.conflict_fraction(
            __import__("nmind.learning.datasets", fromlist=["Trajectory"]).Trajectory(
                obs=r["obs"], actions=r["actions"], goal=tuple(r["goal"]),
                start=tuple(r["start"]), memory_target=np.zeros(r["len"], dtype=np.int64)),
            vis=4)
        assert f >= 0.5, "重算冲突率不得低于门槛（标签可审计）"


def test_h3b_and_h4b_deterministic():
    a = O.make_h3b_split("holdout", n=4, seed=55)
    b = O.make_h3b_split("holdout", n=4, seed=55)
    assert [r["obs"].tobytes() for r in a["rows"]] == [r["obs"].tobytes() for r in b["rows"]]
    c1 = O.make_h4b_split("holdout", n=3, seed=56)
    c2 = O.make_h4b_split("holdout", n=3, seed=56)
    assert [r["obs"].tobytes() for r in c1["rows"]] == [r["obs"].tobytes() for r in c2["rows"]]
    assert a["manifest"]["manifest_hash"] == b["manifest"]["manifest_hash"]


def test_dev_and_holdout_use_different_layout_instances():
    dev = O.make_h3b_split("dev", n=8, seed=1001)
    hold = O.make_h3b_split("holdout", n=8, seed=2001)
    dev_layouts = {a["family"]: tuple(map(tuple, a["layout"])) for a in dev["manifest"]["audit"]}
    hold_layouts = {a["family"]: tuple(map(tuple, a["layout"])) for a in hold["manifest"]["audit"]}
    assert all(dev_layouts[f] != hold_layouts[f] for f in O.FAMILIES)


def test_manifest_hash_covers_actual_rows():
    a = O.make_h3b_split("holdout", n=8, seed=55)
    b = O.make_h3b_split("holdout", n=8, seed=55)
    b["rows"][0]["actions"][0] = (int(b["rows"][0]["actions"][0]) + 1) % 4
    assert a["manifest"]["manifest_hash"] == O._content_hash(a["manifest"], a["rows"])
    assert O._content_hash(a["manifest"], a["rows"]) != O._content_hash(a["manifest"], b["rows"])


def test_h3b_rejects_ambiguous_sample_count():
    import pytest
    with pytest.raises(ValueError):
        O.make_h3b_split("holdout", n=10, seed=1)


def test_h4b_cued_pairs_expose_rule_and_share_start_goal():
    sp = O.make_h4b_cued_split("holdout", n=8, seed=99)
    assert sp["manifest"]["obs_dim"] == O.OBS_DIM_WITH_RULE_CUE
    for i in range(0, len(sp["rows"]), 2):
        normal, reversed_row = sp["rows"][i:i + 2]
        assert normal["start"] == reversed_row["start"]
        assert normal["goal"] == reversed_row["goal"]
        assert np.all(normal["obs"][:, -1] == 0.0)
        assert np.all(reversed_row["obs"][:, -1] == 1.0)
        assert normal["task"] == "h4b_normal"
        assert reversed_row["task"] == "h4b_reversed"


def test_h4b_cued_split_is_content_hashed_and_deterministic():
    a = O.make_h4b_cued_split("dev", n=6, seed=101)
    b = O.make_h4b_cued_split("dev", n=6, seed=101)
    assert a["manifest"]["manifest_hash"] == b["manifest"]["manifest_hash"]
    assert all(np.array_equal(x["obs"], y["obs"]) for x, y in zip(a["rows"], b["rows"]))
