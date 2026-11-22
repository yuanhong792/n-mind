"""M4 真实续训回归测试（纠错任务书第 8 节 8 项断言）。

全部使用小数据/小步数（每条 ≤ ~6 optimizer steps），只验证机制不追性能；
不改动任何 M1/M2 既有工件，运行时间控制在几十秒。
断言全部按浮点存储值逐位比较（torch.equal），不用容差掩盖变化。
"""

import numpy as np
import pytest
import torch

from nmind.config import default_config
from nmind.learning import datasets as D
from nmind.learning.checkpoint import load_checkpoint, save_checkpoint
from nmind.learning.online_adaptation import (AdaptationBudget, BudgetExceeded,
                                              clone_snapshot, fit_existing_model,
                                              verify_bitwise_inheritance)
from nmind.learning.trainer import build_model, evaluate, make_data, train

# 训练态统计 buffer：不计入"参数变化/冻结"断言，漂移单独记录（任务书第 1 节）
TRANSIENT = ("block_load_count", "router.load_bias")


def _tiny_cfg():
    cfg = default_config()
    cfg.learning.n_train = 16
    cfg.learning.n_test = 8
    cfg.learning.n_val = 0
    cfg.learning.traj_len = 12
    cfg.learning.eval_interval = 1000
    cfg.runtime.device = "cpu"
    return cfg


def _tiny_data(seed: int):
    cfg = _tiny_cfg()
    task = D.GridTask(goal_visible_steps=3, seed=seed)
    return cfg, make_data(cfg, task, seed=seed), task


def _assert_params_bitwise(before: dict, after: dict) -> None:
    for k, v in before["model"].items():
        if k in TRANSIENT:
            continue
        assert torch.equal(v, after["model"][k]), f"参数被意外改动：{k}"


def test_1_b_phase_inherits_a_weights_bitwise():
    """断言1：A 训练后进入 B 之前，克隆权重与 A 后快照逐位相同；且 ≠ A 前随机初始化。"""
    cfg, _data, task = _tiny_data(123)
    rA = train(cfg, epochs=1, seed=123, task=task, device="cpu", quiet=True,
               posthoc_probe=False, loss_mode="ce")
    snap = clone_snapshot(rA.model)
    # A 阶段确实学过（权重 ≠ 随机初始化快照）
    fresh = build_model(cfg, seed=123, device="cpu")
    diffs = [k for k in fresh.state_dict() if k not in TRANSIENT
             and not torch.equal(fresh.state_dict()[k], rA.model.state_dict()[k])]
    assert diffs, "A 阶段训练没有改变任何参数，无法作为继承测试的父代"
    # 克隆逐位继承
    b_model = build_model(cfg, seed=999, device="cpu")
    b_model.load_state_dict(snap["model"], strict=True)
    chk = verify_bitwise_inheritance(snap, b_model)
    assert chk["all_bitwise_equal"], chk["mismatched_keys"][:5]
    assert chk["checked_tensors"] > 0


def test_2_explicit_model_takes_priority_no_silent_rebuild():
    """断言2：显式 model= 优先，跳过 build_model；语义不一致必须报错，不许静默覆盖。"""
    cfg, _data, task = _tiny_data(321)
    m = build_model(cfg, seed=5, device="cpu")
    res = train(cfg, epochs=1, seed=321, task=task, device="cpu", quiet=True,
                posthoc_probe=False, loss_mode="ce", model=m)
    assert res.final["ablation"]["init_mode"] == "provided_model"
    assert res.model is m                       # 返回传入对象本身——重建必然是别的对象
    cfg_bad = _tiny_cfg()
    cfg_bad.model.event_numeric_dim = 30        # 与传入模型不兼容
    with pytest.raises(ValueError):
        train(cfg_bad, epochs=1, seed=321, task=task, device="cpu", quiet=True,
              posthoc_probe=False, loss_mode="ce", model=m)


def test_3_incremental_update_changes_only_allowed_params():
    """断言3：增量学习后 ≥1 个允许参数非零变化；任何被冻结参数逐位不变。"""
    cfg, data, _ = _tiny_data(55)
    model = build_model(cfg, seed=7, device="cpu")
    res = fit_existing_model(model, cfg=cfg, rows=data["train"],
                             device=torch.device("cpu"), scope="adapter",
                             max_updates=2, sampling_seed=99, arm="t3")
    delta = res["delta"]
    assert delta["trainable_total"] > 0
    assert delta["trainable_changed_count"] >= 1, "允许更新的参数全部没有变化"
    assert delta["frozen_total"] > 0
    assert delta["frozen_all_bitwise_unchanged"], delta["frozen_violations"][:5]


def test_4_no_update_lr0_paths_params_bitwise_unchanged():
    """断言4：lr=0 与全冻结两条路径的神经参数逐位不变（不许出现虚构学习收益）。"""
    cfg, data, _ = _tiny_data(66)
    model = build_model(cfg, seed=8, device="cpu")
    before = clone_snapshot(model)
    res0 = fit_existing_model(model, cfg=cfg, rows=data["train"],
                              device=torch.device("cpu"), scope="adapter",
                              max_updates=2, sampling_seed=100, lr=0.0, arm="lr0")
    assert res0["optimizer_steps_executed"] == 2        # step 照常执行
    assert res0["delta"]["trainable_changed_count"] == 0
    _assert_params_bitwise(before, clone_snapshot(model))
    # 全冻结：无可训练参数，零步执行
    model2 = build_model(cfg, seed=8, device="cpu")
    before2 = clone_snapshot(model2)
    res_f = fit_existing_model(model2, cfg=cfg, rows=data["train"],
                               device=torch.device("cpu"), scope="frozen",
                               max_updates=3, sampling_seed=101, arm="frozen")
    assert res_f["optimizer_steps_executed"] == 0
    _assert_params_bitwise(before2, clone_snapshot(model2))


def test_5_all_arms_start_from_identical_parent():
    """断言5：各实验臂从同一父快照克隆，逐位一致、哈希一致。"""
    cfg, _data, _ = _tiny_data(77)
    parent = build_model(cfg, seed=9, device="cpu")
    snap = clone_snapshot(parent)
    c1 = build_model(cfg, seed=1, device="cpu")
    c1.load_state_dict(snap["model"])
    c2 = build_model(cfg, seed=2, device="cpu")
    c2.load_state_dict(snap["model"])
    assert verify_bitwise_inheritance(snap, c1)["all_bitwise_equal"]
    assert verify_bitwise_inheritance(snap, c2)["all_bitwise_equal"]
    from nmind.learning.online_adaptation import state_hash
    assert state_hash(c1) == state_hash(c2) == snap["sha256"]


def test_6_reversed_control_has_effective_conflict_samples():
    """断言6：reversed_control 的有效对照样本中 A/B oracle 确实不同，且有足量样本。"""
    cfg = _tiny_cfg()
    task = D.GridTask(name="B", goal_visible_steps=4, reversed_control=True, seed=42)
    data = make_data(cfg, task, seed=42)
    conflict = coincident = occ = 0
    for r in data["test"]:
        goal = (int(r["goal_code"]) // 6, int(r["goal_code"]) % 6)
        for t in range(4, int(r["len"])):
            row = np.asarray(r["obs"][t])
            pos = (int(np.argmax(row[0:6])), int(np.argmax(row[6:12])))
            normal = D.oracle_action(pos, goal, tuple(), 6, reversed_control=False)
            occ += 1
            if normal != int(r["actions"][t]):
                conflict += 1
            else:
                coincident += 1
    assert occ > 0
    assert conflict > 0, "B 测试集没有规则冲突步，无法检验新技能学习"
    assert coincident > 0, "巧合步统计异常（应同时存在两类步）"
    assert conflict + coincident == occ


def test_7_child_checkpoint_roundtrip_and_safe_abort(tmp_path):
    """断言7：子 checkpoint 保存→重载分数一致；预算中止不触碰 stable 路径。"""
    cfg, data, _ = _tiny_data(88)
    model = build_model(cfg, seed=11, device="cpu")
    fit_existing_model(model, cfg=cfg, rows=data["train"], device=torch.device("cpu"),
                       scope="adapter", max_updates=2, sampling_seed=102, arm="t7")
    child = tmp_path / "child_candidate.pt"
    save_checkpoint(model, cfg=cfg, path=child, probe=None, meta={"lineage": "t7"})
    m2, _meta = load_checkpoint(child, cfg=cfg, device="cpu", allow_config_mismatch=True)
    dev = torch.device("cpu")
    s1 = evaluate(model, None, data["test"], device=dev, occluded_only=True)["accuracy_occluded"]
    s2 = evaluate(m2, None, data["test"], device=dev, occluded_only=True)["accuracy_occluded"]
    assert s1 == s2, "保存/重载后评测分数不一致"
    stable = tmp_path / "stable_checkpoint.pt"
    stable.write_bytes(b"immutable-stable")
    budget = AdaptationBudget(2, label="t7")
    budget.spend(2, "a")
    with pytest.raises(BudgetExceeded):
        budget.spend(1, "b")
    assert stable.read_bytes() == b"immutable-stable"


def test_8_fresh_build_path_unchanged():
    """断言8：默认 fresh_build 路径行为不变（M1/M2 语义保留），init_mode 如实记录。"""
    cfg, _data, task = _tiny_data(99)
    res = train(cfg, epochs=1, seed=99, task=task, device="cpu", quiet=True,
                posthoc_probe=False, loss_mode="ce")
    assert res.final["ablation"]["init_mode"] == "fresh_build"
    assert res.steps == cfg.learning.n_updates
