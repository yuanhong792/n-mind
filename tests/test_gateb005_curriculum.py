"""Gate-B 005 方案 C 测试：课程呈现互斥/校验、顺序 vs 交错可执行、cf 双基线 gate。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nmind.bot.learning_loop import run_candidate_update
from nmind.language import binding_data as bd
from nmind.language.lm_update import masked_stream_update
from nmind.language.neural_dialogue import LMConfig, NMindLM, save_lm
from nmind.language.tokenizer import CharTokenizer
from tests.test_gateb004_fixes import (CORPUS, _make_verified, _prompt, _small, _tok)


def _setup(tmp_path):
    tok = _tok(tmp_path)
    torch.manual_seed(19)
    model = NMindLM(_small(tok))
    save_lm(str(tmp_path / "stable.pt"), model=model, tokenizer=tok, cfg=_small(tok),
            steps=0, seed=1, parent_sha="x", manifest_hash="t")
    a_tasks = bd.build_position_qa(16, 20261401, "train")
    b_tasks = bd.build_family("unseen_entity_qa", "train")[:16]
    sealed = bd.build_family("unseen_entity_qa", "sealed")[:6]
    anchor = bd.build_family("position_qa", "sealed")[:8]
    return tok, model, (a_tasks, b_tasks, sealed, anchor)


def _cf_items(n_pairs: int = 3) -> list[dict]:
    tasks = bd.build_binding_swap_wide(n_pairs, 20261711, "counterfactual")
    pairs = bd.counterfactual_pairs(tasks)
    items = []
    for pi, pair in enumerate(pairs):
        for t in pair["tasks"]:
            items.append({"fingerprint": t["fingerprint"], "prompt": t["prompt"],
                           "target": t["target"], "recency_target": pair["recency_target"],
                           "group": f"grp{pi}"})
    return items


def _run_curriculum(tmp_path, tok, curriculum, *, arms=("frozen", "memory", "neural", "negative"),
                     replay_ratio=0.0, cf_min=None, cf_items=None, updates=8):
    tok, _model, (a_tasks, b_tasks, sealed, anchor) = _setup(tmp_path)
    return run_candidate_update(
        stable_checkpoint=str(tmp_path / "stable.pt"), tokenizer_path=str(tmp_path / "tok.json"),
        verified=_make_verified(tok, 16), sealed_prompts=[t.prompt for t in sealed],
        sealed_answers=[t.target for t in sealed], dev=torch.device("cpu"), updates=updates,
        replay_pairs=([t.prompt for t in a_tasks], [t.target for t in a_tasks]),
        retention_pairs=([t.prompt for t in anchor], [t.target for t in anchor]),
        replay_ratio=replay_ratio, trainable="adapter", curriculum=curriculum,
        counterfactual_items=cf_items, counterfactual_follow_min=cf_min,
        arms=arms, wall_limit_s=60.0)


def test_curriculum_replay_mutual_exclusion(tmp_path):
    """005 指令⑤：curriculum 与 replay_ratio>0 互斥（回放权重不得作为并发变量）。"""
    tok = _tok(tmp_path)
    with pytest.raises(ValueError, match="replay_ratio"):
        _run_curriculum(tmp_path, tok,
                         {"mode": "interleaved"}, replay_ratio=0.15)


def test_curriculum_mode_whitelist(tmp_path):
    tok = _tok(tmp_path)
    with pytest.raises(ValueError, match="curriculum.mode"):
        _run_curriculum(tmp_path, tok, {"mode": "random_shuffle"})


def test_sequential_phase_steps_must_sum_to_updates(tmp_path):
    tok = _tok(tmp_path)
    with pytest.raises(ValueError, match="步数和"):
        _run_curriculum(tmp_path, tok,
                         {"mode": "sequential",
                          "phases": [{"steps": 3, "source": "replay_pairs"},
                                      {"steps": 3, "source": "learning_rows"}]},
                         updates=8)


def test_sequential_and_interleaved_both_execute(tmp_path):
    """两组配对设计均可执行：训练发生、预算用满、终判给出（独立封存集已提供）。"""
    tok = _tok(tmp_path)
    for cur in ({"mode": "sequential",
                  "phases": [{"steps": 4, "source": "replay_pairs"},
                              {"steps": 4, "source": "learning_rows"}]},
                 {"mode": "interleaved"}):
        rep = _run_curriculum(tmp_path, tok, cur)
        assert rep["executed"] is True
        assert rep["curriculum"]["mode"] == cur["mode"]
        assert rep["budget_used_steps"] == 8
        assert rep["status"] in ("PROPOSED", "REJECTED")
        assert rep["protocols"]["freegen"] == "freegen-v2-direct-continuation"


def test_cf_double_baseline_gate(tmp_path):
    """cf 双基线：门槛 0.5 时 gate = (rate>=0.5) 且 > 父模型基线 且 > 捷径基线。"""
    tok = _tok(tmp_path)
    cf = _cf_items(3)
    rep = _run_curriculum(tmp_path, tok,
                           {"mode": "interleaved"}, cf_min=0.5, cf_items=cf)
    c = rep["counterfactual"]
    assert c["gate_enforced"] is True and c["gate_min"] == 0.5
    assert c["n_items"] == len(cf) and c["frozen_parent_baseline"] is not None
    expected = (c["follow_rate"] >= 0.5 and c["follow_rate"] > c["frozen_parent_baseline"]
                 and c["follow_rate"] > c["recency_shortcut_baseline"])
    assert rep["gates"]["counterfactual_follow"] is expected
    assert "recency_shortcut_baseline" in c and c["recency_shortcut_baseline"] is not None


def test_lm_update_batch_mix_and_phase_pool_mutual_exclusion(tmp_path):
    """lm_update 层：batch_mix 与 phase_pool 互斥，同时提供直接报错。"""
    tok = _tok(tmp_path)
    torch.manual_seed(29)
    model = NMindLM(_small(tok))
    rows_a = [(tok.encode(_prompt("小明", "1号位置")), (len(tok.encode(_prompt("小明", "1号位置"))),
                                                       len(tok.encode(_prompt("小明", "1号位置")))))]
    with pytest.raises(ValueError, match="互斥"):
        masked_stream_update(model, rows_a, updates=2, dev=torch.device("cpu"),
                              trainable="adapter",
                              batch_mix=(rows_a, rows_a),
                              phase_pool=[(1, rows_a), (1, rows_a)])
