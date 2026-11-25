import json
import hashlib
import multiprocessing as mp
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.zero.budget_ledger import Budget
from experiments.zero.zero6_continual import (
    ContinuousAgent, SkillRecord, SkillWorld, SymbolObjectInterface, load_world_checkpoint,
    run_paired_screening, save_world_checkpoint,
)


def _reserve_worker(state_path: str, queue) -> None:
    budget = Budget("zero6-test-only", 4, state_path)
    try:
        budget.reserve(1, "concurrency-test")
        queue.put("reserved")
    except RuntimeError as exc:
        queue.put(str(exc))


def test_existing_subject_stays_403_parameters_and_initialization_is_paired():
    first, second = ContinuousAgent(101), ContinuousAgent(101)
    assert first.parameter_count == 403
    for a, b in zip(first.predictor.parameters(), second.predictor.parameters()):
        assert torch.equal(a, b)
    for a, b in zip(first.explorer.parameters(), second.explorer.parameters()):
        assert torch.equal(a, b)


def test_historical_selective_gate_loads_frozen_without_task_network_reset():
    for seed in (101, 202, 303):
        checkpoint = Path(f"experiments/zero/checkpoints/zero4_G2_{seed}.pt")
        assert checkpoint.exists()
        agent = ContinuousAgent(seed)
        task_before = [p.detach().clone() for p in agent.predictor.parameters()]
        digest = agent.load_selective_gate(checkpoint)
        assert len(digest) == 64
        assert agent.selective_gate is not None
        assert sum(p.numel() for p in agent.selective_gate.parameters()) == 2
        assert all(not p.requires_grad for p in agent.selective_gate.parameters())
        with torch.no_grad():
            gates = agent.selective_gate(torch.tensor([0.4, 0.7]))
        assert gates.shape == (2,)
        assert bool(torch.isfinite(gates).all()) and bool(((gates >= 0) & (gates <= 1)).all())
        assert all(torch.equal(a, b) for a, b in zip(task_before, agent.predictor.parameters()))


def test_world_rules_contexts_and_features_are_independent_and_sealed_isolated():
    a = SkillWorld(7001, -1.0)
    b = SkillWorld(7002, 0.0)
    assert not torch.equal(a._w, b._w)
    assert a.audit_hashes()["all_disjoint"] is True
    assert all(v == 0 for v in a.audit_hashes()["intersection_counts"].values())
    assert not hasattr(a.agent, "evaluator")
    assert not hasattr(a.agent, "sealed")
    assert not hasattr(a.agent, "_world")
    assert not hasattr(a.agent, "_sets")
    with pytest.raises(ValueError, match="AGENT_QUERY_CAPABILITY_DENIED"):
        a.agent.query("sealed")
    assert not hasattr(a, "query")
    dx, dy = a.evaluator.query()
    assert dx.shape[0] == dy.shape[0] == 128
    assert not hasattr(a.sealed_evaluator, "query")
    sx, sy = a.sealed_evaluator._x.clone(), a.sealed_evaluator._y.clone()
    assert sx.shape[0] == sy.shape[0] == 128
    metrics = ContinuousAgent(13).evaluate(a.evaluator)
    assert set(metrics) == {"accuracy", "balanced_accuracy", "cross_entropy"}


def test_five_context_worlds_recreate_identically_for_paired_arms():
    contexts = (-1.0, -0.5, 0.0, 0.5, 1.0)
    seeds_by_model = {
        101: (611011, 611012, 611013, 611014, 611015),
        202: (612021, 612022, 612023, 612024, 612025),
        303: (613031, 613032, 613033, 613034, 613035),
    }
    for seeds in seeds_by_model.values():
        first = [SkillWorld(seed, context) for seed, context in zip(seeds, contexts)]
        second = [SkillWorld(seed, context) for seed, context in zip(seeds, contexts)]
        assert [w.context for w in first] == list(contexts)
        assert all(torch.equal(a.candidates(), b.candidates()) for a, b in zip(first, second))
        assert all(torch.equal(a._w, b._w) for a, b in zip(first, second))
        assert len({tuple(world._w.tolist()) for world in first}) == len(first)


def test_feedback_interface_blocks_duplicate_and_only_returns_selected_label():
    world = SkillWorld(7111, 1.0)
    before = len(world.agent.seen)
    feature, label = world.agent.feedback(3)
    assert feature.shape == (8,) and int(label) in (0, 1)
    assert len(world.agent.seen) == before + 1
    with pytest.raises(ValueError, match="ILLEGAL_OR_REPEATED_ACTION"):
        world.agent.feedback(3)


def test_symbol_object_binding_roundtrip_contains_features_only():
    world = SkillWorld(8101, -1.0)
    feature = world.agent.observe(7)
    table = SymbolObjectInterface()
    table.bind("object:cup", feature)
    assert torch.equal(table.resolve("object:cup"), feature)
    clone = SymbolObjectInterface()
    clone.load_state_dict(table.state_dict())
    assert torch.equal(clone.resolve("object:cup"), feature)


def test_selective_gate_scales_hidden_and_output_gradients_without_parameter_update():
    agent = ContinuousAgent(202)
    agent.load_selective_gate(Path("experiments/zero/checkpoints/zero4_G2_202.pt"))
    world = SkillWorld(7222, -0.5)
    x, y = world.agent.feedback(0)
    params_before = [p.detach().clone() for p in agent.predictor.parameters()]
    logits = agent.predictor(x.reshape(1, 8))
    loss = torch.nn.functional.cross_entropy(logits, y.reshape(1))
    grads = torch.autograd.grad(loss, tuple(agent.predictor.parameters()))
    grad_norm = torch.sqrt(sum(g.square().sum() for g in grads))
    with torch.no_grad():
        gates = agent.selective_gate(torch.stack((loss.detach(), grad_norm.detach())))
    groups = {"hidden": [], "output": []}
    for (name, _), grad in zip(agent.predictor.named_parameters(), grads):
        group = "hidden" if name.startswith("net.0.") else "output"
        groups[group].append(grad)
        assert torch.isfinite(grad).all()
    for group, gate_index in (("hidden", 0), ("output", 1)):
        before_norm = torch.sqrt(sum(grad.square().sum() for grad in groups[group]))
        after_norm = torch.sqrt(sum((grad * gates[gate_index]).square().sum() for grad in groups[group]))
        assert torch.allclose(after_norm, before_norm * gates[gate_index], atol=1e-7, rtol=1e-5)
    assert all(torch.equal(p, original) for p, original in zip(agent.predictor.parameters(), params_before))


def test_world_boundary_checkpoint_roundtrip_restores_weights_optimizer_rng_and_position(tmp_path):
    state_path = tmp_path / "ledger.json"
    budget = Budget("zero6-test-checkpoint", 20, state_path)
    agent = ContinuousAgent(202)
    path = tmp_path / "world.pt"
    random.seed(3); np.random.seed(4); torch.manual_seed(5)
    gate = Path("experiments/zero/checkpoints/zero4_G2_202.pt")
    agent.load_selective_gate(gate)
    agent.replay_buffer = [{"x": torch.ones(8), "y": torch.tensor(1), "context": -0.5}]
    agent.skills = [SkillRecord(-1.0, 9001, updates=8, curve=[{"updates": 8, "balanced_accuracy": 0.6}]),
                    SkillRecord(-0.5, 9002, updates=8, curve=[{"updates": 8, "balanced_accuracy": 0.7}])]
    agent.history = [{"event": "retention_checkpoint", "payload": {"after_world_index": 1, "per_skill": []}}]
    save_world_checkpoint(path, agent, phase="continual", strategy="selective_update", world_index=1,
                          next_world_index=2, world_seed=9002, budget=budget,
                          protocol_sha256="f" * 64)
    expected_rng = (random.random(), np.random.random(), torch.rand(1))
    expected_params = [p.detach().clone() for p in agent.predictor.parameters()]
    restored = ContinuousAgent(999)
    position = load_world_checkpoint(path, restored, budget=budget,
                                     expected_protocol_sha256="f" * 64)
    actual_rng = (random.random(), np.random.random(), torch.rand(1))
    assert position["next_world_index"] == 2
    assert restored.seed == agent.seed
    assert restored.learning_rate == agent.learning_rate
    assert expected_rng[0] == actual_rng[0] and expected_rng[1] == actual_rng[1]
    assert torch.equal(expected_rng[2], actual_rng[2])
    assert all(torch.equal(a, b) for a, b in zip(expected_params, restored.predictor.parameters()))
    assert restored.selective_gate is not None
    assert all(not p.requires_grad for p in restored.selective_gate.parameters())
    assert len(restored.replay_buffer) == 1
    assert [s.world_seed for s in restored.skills] == [9001, 9002]
    assert restored.history[0]["event"] == "retention_checkpoint"


def test_checkpoint_rejects_ledger_mismatch_and_protocol_mismatch(tmp_path):
    state_path = tmp_path / "ledger.json"
    budget = Budget("zero6-test-mismatch", 20, state_path)
    agent = ContinuousAgent(101)
    path = tmp_path / "world.pt"
    save_world_checkpoint(path, agent, phase="dev", strategy="random", world_index=0,
                          next_world_index=1, world_seed=1, budget=budget,
                          protocol_sha256="a" * 64)
    budget.reserve(1, "external-change")
    with pytest.raises(RuntimeError, match="CHECKPOINT_BUDGET_STATE_MISMATCH"):
        load_world_checkpoint(path, ContinuousAgent(102), budget=budget,
                              expected_protocol_sha256="a" * 64)
    with pytest.raises(RuntimeError, match="CHECKPOINT_PROTOCOL_MISMATCH"):
        load_world_checkpoint(path, ContinuousAgent(102), budget=Budget("zero6-test-mismatch", 20, state_path),
                              expected_protocol_sha256="b" * 64)


def test_checkpoint_rejects_uncertain_or_failed_budget_drift(tmp_path):
    state_path = tmp_path / "ledger.json"
    budget = Budget("zero6-test-counter-drift", 20, state_path)
    agent = ContinuousAgent(101)
    path = tmp_path / "world.pt"
    save_world_checkpoint(path, agent, phase="dev", strategy="random", world_index=0,
                          next_world_index=1, world_seed=1, budget=budget,
                          protocol_sha256="c" * 64)
    budget.reserve(1, "simulated-uncertain-checkpoint-test")
    budget.mark_uncertain(1, "simulated-uncertain-checkpoint-test")
    with pytest.raises(RuntimeError, match="CHECKPOINT_BUDGET_STATE_MISMATCH"):
        load_world_checkpoint(path, ContinuousAgent(101), budget=budget,
                              expected_protocol_sha256="c" * 64)


def test_checkpoint_restore_allows_monotonic_progress_from_other_paired_sessions(tmp_path):
    budget = Budget("zero6-test-forward-ledger", 20, tmp_path / "ledger.json")
    agent = ContinuousAgent(202)
    path = tmp_path / "world.pt"
    save_world_checkpoint(path, agent, phase="continual", strategy="ordinary_sgd",
                          world_index=0, next_world_index=1, world_seed=9001,
                          budget=budget, protocol_sha256="d" * 64)
    budget.reserve(1, "other-session-test-cost")
    budget.complete(1, "other-session-test-cost")
    restored = ContinuousAgent(999)
    position = load_world_checkpoint(path, restored, budget=budget,
                                     expected_protocol_sha256="d" * 64,
                                     allow_later_budget_state=True)
    assert position["next_world_index"] == 1
    assert restored.seed == 202


def test_runner_refuses_uncheckpointed_completed_costs_before_any_training(tmp_path):
    budget = Budget("zero6-test-orphan-cost", 360, tmp_path / "ledger.json")
    budget.reserve(1, "simulated-orphan-update-cost")
    budget.complete(1, "simulated-orphan-update-cost")
    policy_paths = {seed: Path("experiments/zero/checkpoints") / f"zero4_G2_{seed}.pt"
                    for seed in (101, 202, 303)}
    protocol_sha = hashlib.sha256(Path("experiments/zero/ZERO6_TASK_MANIFEST.json").read_bytes()).hexdigest()
    with pytest.raises(RuntimeError, match="ZERO6_UNCHECKPOINTED_OR_UNCERTAIN_BUDGET_CANNOT_RESUME"):
        run_paired_screening(budget, policy_paths, authorization_confirmed=True,
                             checkpoint_root=tmp_path / "checkpoints",
                             protocol_sha256=protocol_sha)


def test_budget_concurrent_reservations_never_exceed_limit(tmp_path):
    state_path = str(tmp_path / "concurrent.json")
    queue = mp.Queue()
    workers = [mp.Process(target=_reserve_worker, args=(state_path, queue)) for _ in range(8)]
    for process in workers:
        process.start()
    for process in workers:
        process.join(10)
        assert process.exitcode == 0
    outcomes = [queue.get(timeout=2) for _ in workers]
    snapshot = Budget("zero6-test-only", 4, state_path).snapshot()
    assert outcomes.count("reserved") == 4
    assert snapshot["reserved"] == 4
    assert snapshot["completed"] == 0


def test_uncertain_update_resolution_never_refunds_budget(tmp_path):
    budget = Budget("zero6-test-uncertain", 3, tmp_path / "uncertain.json")
    budget.reserve(1, "test-update")
    budget.mark_uncertain(1, "interrupt-after-update")
    budget.resolve_uncertain(1, "failed-cost")
    snapshot = budget.snapshot()
    assert snapshot["reserved"] == 1
    assert snapshot["completed"] == 0
    assert snapshot["failed"] == 1
    assert snapshot["uncertain"] == 0


def test_readiness_prediction_and_evaluation_do_not_change_weights():
    agent = ContinuousAgent(303)
    world = SkillWorld(3031, 1.0)
    before = [p.detach().clone() for p in agent.predictor.parameters()]
    predicted = agent.predictor(world.agent.candidates())
    metrics = agent.evaluate(world.evaluator)
    assert predicted.shape == (256, 2)
    assert set(metrics) == {"accuracy", "balanced_accuracy", "cross_entropy"}
    assert all(torch.equal(a, b) for a, b in zip(before, agent.predictor.parameters()))


def test_continual_training_is_denied_without_explicit_authorization(tmp_path):
    budget = Budget("zero6-test-noauth", 20, tmp_path / "noauth.json")
    agent = ContinuousAgent(11)
    world = SkillWorld(1101, 0.0)
    before = [p.detach().clone() for p in agent.predictor.parameters()]
    with pytest.raises(RuntimeError, match="ZERO6_TRAINING_NOT_AUTHORIZED"):
        agent.adapt_skill(world.agent, world.evaluator, world.world_seed, world.context,
                          "ordinary_sgd", budget, updates=1)
    assert budget.snapshot()["reserved"] == 0
    assert all(torch.equal(a, b) for a, b in zip(before, agent.predictor.parameters()))


def test_model_update_source_reserves_budget_before_optimizer_step():
    source = Path("experiments/zero/zero6_continual.py").read_text(encoding="utf-8")
    method = source.split("def learn_feedback", 1)[1].split("@torch.no_grad()", 1)[0]
    assert method.index("budget.reserve(") < method.index("self.optimizer.step()")
    assert "budget.mark_uncertain" in method


def test_budget_plan_is_only_a_request_and_does_not_touch_zero5_ledger():
    request = json.loads(Path("experiments/zero/ZERO6_MINIMUM_BUDGET_REQUEST.json").read_text(encoding="utf-8"))
    assert request["status"] == "REQUEST_ONLY_NOT_AUTHORIZED"
    assert request["new_updates_requested_max"] == 360
    assert request["zero5_remaining_140"] == "NOT_USED"
