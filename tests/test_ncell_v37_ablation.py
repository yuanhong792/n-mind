"""Bounded offline tests for reproducible neural-only architecture experiments."""
from __future__ import annotations

import json

import pytest
import torch

from nmind.evaluation.ncell_ablation import (
    ARMS, build_dataset, build_tokenizer, encode_batch, evaluate,
    make_model, manifest, markdown_report, run_experiment, state_hash,
    train_candidate, validate_splits, write_report,
)
from nmind.language.tokenizer import EOS


def _data():
    rows = build_dataset(20261123)
    tok = build_tokenizer(rows["train"], rows["dev"] + rows["sealed"])
    return rows, tok


def test_group_split_is_semantic_pair_disjoint_and_reproducible():
    d1 = build_dataset(3)
    d2 = build_dataset(3)
    assert d1 == d2
    assert validate_splits(d1) == {
        "train": {"groups": 60, "samples": 120},
        "dev": {"groups": 20, "samples": 40},
        "sealed": {"groups": 16, "samples": 32},
    }
    assert len({row.prompt for rows in d1.values() for row in rows}) == 192


def test_every_context_has_counterfactual_queries_with_different_targets():
    data = build_dataset(10)
    for split, rows in data.items():
        groups = {}
        for row in rows:
            groups.setdefault(row.group, []).append(row)
        for pair in groups.values():
            assert len(pair) == 2
            assert pair[0].query_entity != pair[1].query_entity
            assert pair[0].answer != pair[1].answer
            assert pair[0].prompt != pair[1].prompt


def test_answer_only_teacher_forcing_mask_exact_alignment():
    data, tok = _data()
    row = data["train"][0]
    x, y, n = encode_batch([row], tok)
    prompt_ids = tok.encode(row.prompt)
    answer_ids = tok.encode(row.answer)
    assert n == len(answer_ids) + 1
    assert (y[0, :len(prompt_ids)-1] == -100).all()
    assert y[0, len(prompt_ids)-1:len(prompt_ids)-1+n].tolist() == answer_ids + [EOS]
    assert x.shape == y.shape


def test_untrained_modes_have_identical_legacy_weights_and_no_router_gradient():
    data, tok = _data()
    legacy, n = make_model(tok.vocab_size, "legacy", 42)
    router, baseline_n = make_model(tok.vocab_size, "router_ste", 42)
    assert n == baseline_n == legacy.n_params() == router.n_params()
    assert state_hash(legacy) == state_hash(router)
    assert not legacy.cfg.use_router_ste and router.cfg.use_router_ste
    x, y, _ = encode_batch(data["train"][:2], tok)
    legacy.eval()
    router.eval()
    with torch.no_grad():
        assert torch.equal(legacy(x)[0], router(x)[0])


def test_zero_step_does_not_change_weights_and_sealed_not_opened(tmp_path):
    torch.set_num_threads(1)
    data, tok = _data()
    model, _ = make_model(tok.vocab_size, "legacy", 15)
    before = state_hash(model)
    t = train_candidate(model, data["train"], tok, steps=0, batch_size=2, seed=15)
    assert t["optimizer_steps_executed"] == 0 and not t["state_changed"]
    assert state_hash(model) == before
    res = run_experiment(seed=15, steps=0, batch_size=2,
                         eval_samples=2, arms=("legacy",))
    assert res["arms"]["legacy"]["sealed"]["status"] == "NOT_RUN"
    assert res["arms"]["legacy"]["training"]["optimizer_steps_executed"] == 0
    assert res["teacher_calls"] == res["memory_reads"] == res["tool_calls"] == 0
    j, md = write_report(res, tmp_path)
    assert json.loads(j.read_text(encoding="utf-8"))["manifest"]["seed"] == 15
    assert "Structural Ablation Results" in md.read_text(encoding="utf-8")
    assert "NOT_RUN" in markdown_report(res)


def test_one_real_candidate_optimizer_step_and_free_generation(tmp_path):
    torch.set_num_threads(1)
    d, tok = _data()
    model, _ = make_model(tok.vocab_size, "router_ste", 8)
    out = train_candidate(model, d["train"], tok, steps=1, batch_size=2,
                          seed=8, max_seconds=60)
    assert out["optimizer_steps_executed"] == 1
    assert out["state_changed"] is True
    assert out["curve"][0]["grad"]["router"] > 0
    result = evaluate(model, d["dev"], tok, limit=2, max_new_tokens=14)
    assert result["evaluated"] == 2
    assert result["paired_contexts"] == 1
    assert all(x["sample_source"] == "neural_only_no_teacher_memory_or_tool"
               for x in result["records"])


def test_invalid_arm_and_training_budget_rejected():
    data, tok = _data()
    m, _ = make_model(tok.vocab_size, "legacy", 1)
    with pytest.raises(ValueError):
        make_model(tok.vocab_size, "unknown", 1)
    with pytest.raises(ValueError):
        train_candidate(m, data["train"], tok, steps=129, batch_size=2, seed=1)
    with pytest.raises(ValueError):
        run_experiment(steps=-1, arms=("legacy",))
