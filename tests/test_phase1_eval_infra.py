from __future__ import annotations

import torch
import pytest

from nmind.evaluation.phase1_archive import generate_archive
from nmind.evaluation.phase1_baselines import (counterfactual_follow_rate,
    recent_entity_predictions, shortcut_predictions)
from nmind.evaluation.phase1_metrics import add_text_distance, answer_metrics, levenshtein
from nmind.evaluation.phase1_stats import paired_bootstrap, summarize
from nmind.language.binding_data import build_family, counterfactual_pairs


def test_continuous_metrics_known_values():
    lp = torch.log(torch.tensor([[0.5, 0.5], [0.25, 0.75]]))
    out = answer_metrics(lp, torch.tensor([0, 1]), candidate_log_likelihoods=[-2.0, -1.0])
    assert out["per_token_nll"] == pytest.approx([-torch.log(torch.tensor(.5)).item(),
                                                   -torch.log(torch.tensor(.75)).item()])
    assert out["candidate_rank"] == 1
    assert levenshtein("小明在1号位置。", "小红在1号位置。") == 1
    assert add_text_distance({}, "a", "ab")["char_edit_distance"] == 1


def test_stats_reproducible_and_insufficient():
    a = summarize([1, 2, 3], seed=7, n_boot=100)
    b = summarize([1, 2, 3], seed=7, n_boot=100)
    assert a == b and a["status"] == "OK"
    assert paired_bootstrap([1, 2], [0, 1])["status"] == "INSUFFICIENT_N"


def test_archive_disjoint_and_fingerprinted():
    tasks = build_family("position_qa", "train", n=20, seed=1)
    rows, audit = generate_archive(families=("position_qa",), n_each=500,
                                   forbidden={"known-learning-flow-fingerprint"})
    assert len(rows["position_qa"]) == 500
    assert audit["overlap_count"] == 0
    assert len(audit["data_fingerprint"]) == 16


def test_shortcuts_and_counterfactual_probe_behavior():
    from nmind.language.binding_data import build_binding_swap_wide
    tasks = build_binding_swap_wide(n_pairs=12, seed=9)
    preds = shortcut_predictions(tasks, seed=2)
    assert set(preds) == {"majority", "random", "fixed_position"}
    assert all(len(v) == len(tasks) for v in preds.values())
    pairs = counterfactual_pairs(tasks)
    mapping = {t.fingerprint: t.target for t in tasks}
    result = counterfactual_follow_rate(pairs, mapping)
    assert result["status"] == "OK" and result["follow_rate"] == 1.0
    assert len(recent_entity_predictions(tasks)) == len(tasks)
