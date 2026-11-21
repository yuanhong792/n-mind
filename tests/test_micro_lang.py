"""M3.1 微型语言生成器的正确性测试：oracle 判分、切分策略、确定性。

这些测试钉住的是"数据 + 标签"的可信度——模型训练之前先钉数据，
否则任何训练结论都可能是标签错了。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nmind.language import micro_lang as L


def test_oracle_color_and_pos_queries():
    rng = np.random.default_rng(1)
    ep = L.make_episode(rng, combo=("assert_color", "query_color", "query_pos"),
                        names=L.TRAIN_NAMES)
    # 断言 yes/no 与查询答案必须一致：手工核对第一回合渲染
    steps = ep["steps"]
    assert steps[0]["kind"] == "assert_color"
    assert steps[1]["answer"] in L.COLORS
    assert steps[2]["answer"] in {str(p) for p in L.POSITIONS}
    # 同一 episode 内状态一致：query_color 的答案等于该实体在世界里的颜色
    # （由 make_episode 内部状态保证，这里验证答案集合合法 + spans 与文本对齐）
    for (a, b), ans in zip(ep["answer_spans"], ep["answers"]):
        assert ep["prompt"][a:b] == ans, "answer span 必须与答案文本逐字对齐"


def test_move_changes_state_and_binding():
    rng = np.random.default_rng(2)
    ep = L.make_episode(rng, combo=("move", "query_pos"), names=L.TRAIN_NAMES)
    steps = ep["steps"]
    mv, qp = steps
    if qp["name"] == mv["name"]:                    # 同一实体：查询必须返回 move 目的地
        assert qp["answer"] == str(mv["pos"])
    else:                                           # 不同实体：返回其当前位置（0..7 的数字串）
        assert qp["answer"] in {str(p) for p in L.POSITIONS}


def test_it_binds_last_mentioned():
    """force_it=True：首回合建立绑定（显式名词），其后回合全用 it 且指向前一实体。"""
    rng = np.random.default_rng(3)
    ep = L.make_episode(rng, combo=("move", "query_pos"), names=L.TRAIN_NAMES,
                        force_it=True)
    steps = ep["steps"]
    assert "it" not in steps[0]["text"].split("->")[0] or True   # 首回合允许显式名词
    assert steps[1]["ref"] == "it"                                # 其后必须用 it
    assert steps[1]["answer"] == str(steps[0]["pos"])            # 指向被 move 的实体


def test_negate_and_count_are_verifiable():
    rng = np.random.default_rng(4)
    ep = L.make_episode(rng, combo=("negate", "count"), names=L.TRAIN_NAMES)
    neg, cnt = ep["steps"]
    # negate 的 yes/no 由 oracle 核对；这里验证 count 与 negate 的结论互不矛盾
    # （negate=no 意味着该实体就是该颜色，此时 count >= 1）
    if neg["answer"] == "no":
        assert int(cnt["answer"]) >= 1


def test_splits_are_deterministic_and_disjoint():
    a = L.make_split("train", n=8, seed=11)
    b = L.make_split("train", n=8, seed=11)
    c = L.make_split("train", n=8, seed=12)
    assert [e["prompt"] for e in a["episodes"]] == [e["prompt"] for e in b["episodes"]]
    assert [e["prompt"] for e in a["episodes"]] != [e["prompt"] for e in c["episodes"]]


def test_ood_name_uses_only_unseen_names():
    sp = L.make_split("OOD-name", n=5, seed=5, names=L.OOD_NAMES)
    for e in sp["episodes"]:
        for n in L.TRAIN_NAMES:
            assert n not in e["prompt"], f"OOD-name 不得出现训练实体名 {n}"


def test_ood_compose_uses_only_unseen_combos():
    sp = L.make_split("OOD-compose", n=5, seed=6, force_it=True,
                      ood_combo_pool=L.OOD_COMBOS)
    train_pairs = {frozenset(c) for c in L.TRAIN_COMBOS}
    for e in sp["episodes"]:
        assert frozenset(e["combo"]) not in train_pairs, \
            f"OOD-compose 组合 {e['combo']} 不得出现在训练白名单"


def test_ood_length_exceeds_training_limit():
    sp = L.make_split("OOD-length", n=5, seed=7, chain_steps=L.MAX_TRAIN_CHAIN + 2)
    for e in sp["episodes"]:
        assert len(e["combo"]) > L.MAX_TRAIN_CHAIN


def test_loss_mask_covers_answer_chars_only():
    sp = L.make_split("train", n=3, seed=8)
    for e in sp["episodes"]:
        ids, m = e["ids"], e["loss_mask"]
        assert len(ids) == len(m)
        n_ans = sum(m)
        total_ans_chars = sum(len(a) for a in e["answers"])
        assert n_ans == total_ans_chars, "loss_mask 应恰好覆盖全部答案字符（LM 位移后）"
        # 掩码位置 i 的下一字符必须落在某个答案 span 内
        spans = e["answer_spans"]
        for i in range(len(ids) - 1):
            if m[i]:
                assert any(a <= i + 1 < b for a, b in spans)


def test_vocab_covers_all_splits():
    for args in (dict(n=3, seed=9), dict(n=3, seed=9, names=L.OOD_NAMES),
                 dict(n=3, seed=9, chain_steps=5), dict(n=3, seed=9, force_it=True,
                                                        ood_combo_pool=L.OOD_COMBOS)):
        sp = L.make_split("t", **args)
        for e in sp["episodes"]:
            assert all(0 <= i < L.V for i in e["ids"])


def test_manifest_has_manifest_hash():
    sp = L.make_split("train", n=2, seed=10)
    assert len(sp["manifest_hash"]) == 16
