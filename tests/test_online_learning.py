"""M4 联网自主神经学习回归测试（任务书第 8 节）。

P0 继承断言在 tests/test_true_continual.py（test_1 继承逐位 / test_2 显式模型优先等）；
本文件覆盖联网通路、内容隔离、经验流纪律、候选晋升与预算硬停。
"""

import subprocess

import numpy as np
import pytest
import torch

from nmind.config import default_config
from nmind.evaluation.lifelong import rule_conflict_masks
from nmind.learning import datasets as D
from nmind.learning.checkpoint import load_checkpoint, save_checkpoint
from nmind.learning.online_adaptation import AdaptationBudget, BudgetExceeded, state_hash
from nmind.learning.online_trainer import bounded_stream_update
from nmind.learning.trainer import build_model, evaluate, make_data
from nmind.online import (Experience, FetchConfig, Fetcher, MALICIOUS_URL, RULE_MIRROR_URL,
                          RULE_PAGE_URL, build_learning_stream, build_rule_oracle, check_gates,
                          decide, decide_status, detect_instructions, extract_rule_table,
                          fixture_fetcher, file_sha256, rollback_candidate)


def _tiny_cfg():
    cfg = default_config()
    cfg.learning.n_train = 16
    cfg.learning.n_test = 8
    cfg.learning.n_val = 0
    cfg.learning.traj_len = 12
    cfg.learning.eval_interval = 1000
    cfg.runtime.device = "cpu"
    return cfg


_RULE_SPEC = {"fields": {"rule_id": "control-reversed-v1", "version": "1.0",
                          "policy": "max_distance"}}


def _rule_rows(n: int, seed: int, T: int = 12, task: str = "rule_dir_bias",
               used_pairs: set | None = None) -> list[dict]:
    """按已验证规则 oracle 生成轨迹行（与基础任务同一 26 维观测编码）。"""
    oracle = build_rule_oracle(_RULE_SPEC)
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for _ in range(n):
        for _try in range(200):
            s = (int(rng.integers(0, 6)), int(rng.integers(0, 6)))
            g = (int(rng.integers(0, 6)), int(rng.integers(0, 6)))
            if s == g or (used_pairs is not None and (s, g) in used_pairs):
                continue
            if used_pairs is not None:
                used_pairs.add((s, g))
            break
        else:
            raise RuntimeError("start/goal 采样耗尽")
        obs = np.zeros((T, D.OBS_DIM), dtype=np.float32)
        acts = np.zeros(T, dtype=np.int64)
        pos, collided, last = s, False, None
        for t in range(T):
            obs[t] = D._encode(pos[0], pos[1], g[0], g[1], t >= 4, collided)
            a = oracle(pos, g, tuple(), 6, last_action=last)
            acts[t] = a
            dx, dy = D.DIRS[a]
            nx, ny = pos[0] + dx, pos[1] + dy
            if 0 <= nx < 6 and 0 <= ny < 6:
                collided, pos = False, (nx, ny)
            else:
                collided = True
            last = a
        rows.append({"obs": obs, "actions": acts, "goal_code": g[0] * 6 + g[1],
                     "len": T, "task": task, "start": list(s), "goal": list(g)})
    return rows


def _tiny_setup(seed: int = 55):
    cfg = _tiny_cfg()
    rows = _rule_rows(16, seed=seed)
    replay = make_data(cfg, D.GridTask(goal_visible_steps=3, seed=seed), seed=seed)["train"]
    return cfg, rows, replay


def _mk_exp(i: int, *, verification: str, split: str = "stream_train") -> Experience:
    return Experience(id=f"e{i}", observed_at="2026-11-22T00:00:00+00:00",
                      source_url=None, source_sha256="0" * 64, raw_excerpt="",
                      structured_observation={}, action=None, expected_outcome=None,
                      actual_outcome=None, verifier_id=None, verification=verification,
                      reward=None, task_family="t", train_split=split)


def test_offline_mode_blocks_http(monkeypatch):
    """断言：offline 模式切断真实 HTTP（连 urlopen 都不许被调用）。"""

    def boom(*a, **k):
        raise AssertionError("offline 模式下不得发起真实 HTTP")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    f = Fetcher(FetchConfig(offline=True))
    rec = f.fetch("https://example.com/")
    assert rec.status == "blocked"
    assert rec.content is None and "offline" in (rec.error or "")


def test_web_content_cannot_run_commands_or_change_config(monkeypatch):
    """断言：恶意网页只被记为不可信文本——不执行、不改配置、进不了学习流。"""

    def no_exec(*a, **k):
        raise AssertionError("网页内容不得触发本地执行")

    monkeypatch.setattr(subprocess, "Popen", no_exec)
    monkeypatch.setattr(subprocess, "run", no_exec)
    monkeypatch.setattr("os.system", no_exec)
    f = fixture_fetcher()
    rec = f.fetch(MALICIOUS_URL)
    flags = detect_instructions(rec.content)
    assert flags["instruction_detected"] and flags["script_blocks_stripped"] >= 1
    exp = Experience(id="mal-1", observed_at="2026-11-22T00:00:00+00:00",
                     source_url=MALICIOUS_URL, source_sha256=rec.sha256,
                     raw_excerpt=rec.content[:200],
                     structured_observation={"page": "malicious"}, action=None,
                     expected_outcome=None, actual_outcome=None, verifier_id=None,
                     verification="rejected", reward=None, task_family="adversarial",
                     train_split="stream_train")
    assert build_learning_stream([exp]) == []
    cfg = _tiny_cfg()
    before = cfg.model_dump_json()
    extract_rule_table(rec.content)          # 处理对抗页面不改任何配置
    assert cfg.model_dump_json() == before


def test_duplicate_url_content_is_deduplicated():
    f = fixture_fetcher()
    r1 = f.fetch(RULE_PAGE_URL)
    r2 = f.fetch(RULE_MIRROR_URL)
    assert r1.status == "fetched" and r2.status == "duplicate"
    assert r2.sha256 == r1.sha256 and r2.duplicate_of_url == RULE_PAGE_URL


def test_unverified_experience_cannot_trigger_update():
    exps = [_mk_exp(i, verification="rejected") for i in range(5)] + \
           [_mk_exp(99, verification="verified")]
    stream = build_learning_stream(exps)
    assert [e.id for e in stream] == ["e99"]
    d = decide(verified_count=0, rejected_count=5, uncertain_count=0,
               budget_remaining=100, cooldown_ok=True)
    assert d["decision"] == "MEMORY_ONLY"
    d2 = decide(verified_count=64, rejected_count=5, uncertain_count=0,
                budget_remaining=100, cooldown_ok=True)
    assert d2["decision"] == "CANDIDATE_WEIGHT_UPDATE"


def test_sealed_eval_never_enters_replay():
    exps = [_mk_exp(0, verification="verified", split="stream_train"),
            _mk_exp(1, verification="verified", split="sealed_eval"),
            _mk_exp(2, verification="verified", split="stream_val")]
    stream = build_learning_stream(exps)
    assert [e.id for e in stream] == ["e0"]
    with pytest.raises(ValueError):
        build_learning_stream(exps, allow_splits=("stream_train", "sealed_eval"))


def test_candidate_failure_rolls_back_stable_checkpoint(tmp_path):
    stable = tmp_path / "stable_v1.pt"
    stable.write_bytes(b"stable-weights")
    sha = file_sha256(stable)
    cand = tmp_path / "candidate_v2.pt"
    cand.write_bytes(b"candidate-weights")
    matrix = {"inheritance_ok": True, "frozen_violations": 0, "b_ood_gain": 0.02,
              "negative_b_ood_gain": 0.01, "a_forgetting": 0.01,
              "neural_beats_frozen": False}
    gates = check_gates(matrix)
    assert not gates["all_pass"]
    assert decide_status(gates, ran_all_arms=True) == "REJECTED"
    rb = rollback_candidate(cand, stable_path=stable, stable_sha_before=sha,
                            reason="gates fail")
    assert rb["candidate_deleted"] and rb["stable_unchanged"]
    assert not cand.exists() and file_sha256(stable) == sha


def test_learning_budget_hard_stop():
    cfg, rows, replay = _tiny_setup()
    model = build_model(cfg, seed=3, device="cpu")
    stable = {k: v.clone() for k, v in model.state_dict().items()}
    budget = AdaptationBudget(2, label="t")
    with pytest.raises(BudgetExceeded):
        bounded_stream_update(model, cfg=cfg, rows=rows, device=torch.device("cpu"),
                              stable_state=stable, max_updates=10, budget=budget,
                              arm="hard-stop")


def test_new_skill_metric_is_unseen_holdout():
    """新技能指标只在未见 holdout 上计算：训练流与封存测试共享 (start,goal) 即失败。"""
    used: set = set()
    train_rows = _rule_rows(16, seed=20261201, used_pairs=used)
    test_rows = _rule_rows(8, seed=20261202, used_pairs=used)
    train_pairs = {(tuple(r["start"]), tuple(r["goal"])) for r in train_rows}
    test_pairs = {(tuple(r["start"]), tuple(r["goal"])) for r in test_rows}
    assert not (train_pairs & test_pairs)
    conflict, coincident = rule_conflict_masks(test_rows)
    assert int(conflict.sum()) > 0 and int(coincident.sum()) > 0


def test_no_gradient_when_memory_only():
    """断言：memory-only 路径零梯度、参数逐位不变（统计 buffer 单列）。"""
    cfg, rows, _replay = _tiny_setup()
    model = build_model(cfg, seed=31, device="cpu")
    before = {k: v.clone() for k, v in model.state_dict().items()}
    mem_rows = []
    for r in rows:
        row = dict(r)
        row["mem"] = np.zeros((r["len"], 16), dtype=np.float32)
        row["mem"][:, 0] = 1.0
        mem_rows.append(row)
    evaluate(model, None, mem_rows, device=torch.device("cpu"), occluded_only=True)
    after = {k: v.clone() for k, v in model.state_dict().items()}
    for k, v in before.items():
        if k == "block_load_count":      # 纯统计计数器，前向即累加，无功能影响
            continue
        assert torch.equal(v, after[k]), k
    assert all(p.grad is None for p in model.parameters())


def test_load_resume_preserves_pred_and_state(tmp_path):
    cfg = _tiny_cfg()
    model = build_model(cfg, seed=21, device="cpu")
    p = tmp_path / "cand.pt"
    save_checkpoint(model, cfg=cfg, path=p, probe=None, meta={"lineage": "t9"})
    m1, meta1 = load_checkpoint(p, cfg=cfg, device="cpu", allow_config_mismatch=True)
    m2, _meta2 = load_checkpoint(p, cfg=cfg, device="cpu", allow_config_mismatch=True)
    assert state_hash(m1) == state_hash(m2) == state_hash(model)
    assert m1.pred_mode == model.pred_mode          # pred_mode 随档恢复
    assert meta1.get("missing_keys") == []
