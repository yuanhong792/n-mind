"""数据生成器的回归测试：划分不泄漏、标签可由 oracle 复算、遮挡掩码正确。

这些测试存在的直接原因是 M2 排查时发现过一次真实泄漏：
旧 make_split 给 test[i] 分配的 RNG 流恰好等于 train[i+1]（seed*1000+2+i），
测试轨迹与训练轨迹逐字节相同（32/32），0.93 的"泛化"其实是背书。
"""

from __future__ import annotations

import numpy as np

from nmind.learning import datasets as D


def _sig(r) -> bytes:
    return r.obs.tobytes() + r.actions.tobytes()


def _pairs(rows) -> set:
    out = set()
    for r in rows:
        o = r.obs[0]
        start = (int(np.argmax(o[0:6])), int(np.argmax(o[6:12])))
        goal = (int(np.argmax(o[12:18])), int(np.argmax(o[18:24])))
        out.add((start, goal))
    return out


def test_splits_never_overlap_in_trajectories_or_pairs():
    task = D.GridTask(goal_visible_steps=4, seed=20261121)
    data = D.make_split(task, n_train=96, n_test=32, n_val=32, T=24, seed=20261121)
    tr, va, te = data["train"], data["val"], data["test"]
    sigs_tr = {_sig(r) for r in tr}
    assert len(sigs_tr) == len(tr), "训练集自身出现完全重复的轨迹"
    assert not ({_sig(r) for r in te} & sigs_tr)
    assert not ({_sig(r) for r in va} & sigs_tr)
    assert not ({_sig(r) for r in te} & {_sig(r) for r in va})
    assert not (_pairs(tr) & _pairs(te))
    assert not (_pairs(tr) & _pairs(va))
    assert not (_pairs(va) & _pairs(te))


def test_same_seed_is_bitwise_reproducible_and_new_seed_differs():
    task = D.GridTask(goal_visible_steps=4, seed=5)
    a = D.make_split(task, n_train=8, n_test=4, T=12, seed=5)
    b = D.make_split(task, n_train=8, n_test=4, T=12, seed=5)
    assert [_sig(r) for r in a["train"]] == [_sig(r) for r in b["train"]]
    c = D.make_split(task, n_train=8, n_test=4, T=12, seed=6)
    assert _sig(c["train"][0]) != _sig(a["train"][0])


def test_labels_match_oracle_replay():
    """从观测重放轨迹，逐步用 oracle 复算动作；任何一步不一致都说明标签不可验证。"""
    task = D.GridTask(goal_visible_steps=4, seed=20261121)
    rows = D.make_split(task, n_train=6, n_test=2, T=24, seed=20261121)["train"]
    for r in rows:
        o = r.obs
        pos = (int(np.argmax(o[0][0:6])), int(np.argmax(o[0][6:12])))
        goal = (int(np.argmax(o[0][12:18])), int(np.argmax(o[0][18:24])))
        last, collided = None, False
        for t in range(len(r.actions)):
            occluded = t >= task.goal_visible_steps
            assert bool(o[t][24] > 0.5) == occluded
            if occluded:                       # 遮挡后目标 one-hot 必须是全 0，否则等于把答案喂回去
                assert float(o[t][12:24].sum()) == 0.0
            expect = D.oracle_action(pos, goal, set(task.obstacles), task.size,
                                     reversed_control=task.reversed_control, last_action=last)
            assert int(r.actions[t]) == expect, f"t={t} 标签与 oracle 不一致"
            dx, dy = D.DIRS[expect]
            nx, ny = pos[0] + dx, pos[1] + dy
            collided = not (0 <= nx < task.size and 0 <= ny < task.size
                            and (nx, ny) not in set(task.obstacles))
            if not collided:
                pos = (nx, ny)
            assert bool(o[t][25] > 0.5) == collided
            last = expect
        # 最后一步之后的位置不再被观测，无需核对


def test_used_pairs_rejects_duplicates():
    task = D.GridTask(seed=3)
    used: set = set()
    a = D.make_trajectory(task, D._split_rng(3, 1, 0), 8, used_pairs=used)
    pairs = _pairs([a])
    assert len(pairs) == 1
    b = D.make_trajectory(task, D._split_rng(3, 1, 0), 8, used_pairs=used)  # 同一流再抽
    assert _pairs([b]) != pairs, "used_pairs 没挡住重复的 (start, goal)"


def test_dataset_hash_carries_split_version():
    task = D.GridTask(seed=11)
    h1 = D.dataset_hash(task, n_train=8, n_test=4, T=12, seed=11)
    h2 = D.dataset_hash(task, n_train=8, n_test=4, T=12, seed=11, n_val=4)
    assert h1 != h2
    assert D.SPLIT_VERSION.startswith("v2")
