"""ZERO-6 continuous-skill learning prototype. Importing/running audits does not train."""
from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
from torch import nn
from torch.nn import functional as F

from .ZERO5_AGENT import Predictor
from .ZERO5_EXPLORATION import Explorer
from .zero4 import Gate
from .budget_ledger import Budget


class SkillWorld:
    """One hidden rule, shared by disjoint pools; agent sees features and context only."""
    SETS = ("pool", "reward", "dev", "sealed")

    def __init__(self, world_seed: int, context: float, n_pool: int = 256, n_query: int = 128):
        self.world_seed, self.context = int(world_seed), float(context)
        rule_rng = torch.Generator().manual_seed(self.world_seed)
        self._w = torch.randn(7, generator=rule_rng)
        self._bias = torch.randn((), generator=rule_rng) * 0.2
        self._sets: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for index, name in enumerate(self.SETS):
            count = n_pool if name == "pool" else n_query
            gen = torch.Generator().manual_seed(self.world_seed * 1009 + index + 17)
            raw = torch.randn(count, 7, generator=gen)
            # Last feature carries an observable, answer-independent context value.
            x = torch.cat((raw, torch.full((count, 1), self.context)), dim=1)
            y = ((raw @ self._w + self._bias) > 0).long()
            self._sets[name] = (x, y)
        self._seen: set[int] = set()
        feedback_channel = FeedbackChannel(self._sets["pool"][0], self._sets["pool"][1])
        self.agent = SkillWorldAgentView(self._sets["pool"][0], feedback_channel)
        self.evaluator = SkillWorldEvaluator("dev", *self._sets["dev"])
        self.sealed_evaluator = SealedEvaluator(*self._sets["sealed"])

    def candidates(self) -> torch.Tensor:
        return self._sets["pool"][0].clone()

    def observe(self, index: int) -> torch.Tensor:
        if not 0 <= int(index) < len(self._sets["pool"][0]):
            raise ValueError("ILLEGAL_ACTION")
        return self._sets["pool"][0][int(index)].clone()

    def feedback(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        index = int(index)
        if not 0 <= index < len(self._sets["pool"][0]) or index in self._seen:
            raise ValueError("ILLEGAL_OR_REPEATED_ACTION")
        self._seen.add(index)
        # Return only the selected label; the hidden rule and remaining labels stay private.
        return self.observe(index), self._sets["pool"][1][index].clone()

    def audit_hashes(self) -> dict[str, Any]:
        hashes = {name: {feature_sha(x) for x in self._sets[name][0]} for name in self.SETS}
        intersections = {f"{a}:{b}": len(hashes[a] & hashes[b])
                         for i, a in enumerate(self.SETS) for b in self.SETS[i + 1:]}
        return {"counts": {k: len(v) for k, v in hashes.items()},
                "intersection_counts": intersections,
                "all_disjoint": all(n == 0 for n in intersections.values())}


class SkillWorldAgentView:
    """Capabilities given to an agent: features, legal feedback, and no query labels."""
    def __init__(self, pool_features: torch.Tensor, feedback_channel: "FeedbackChannel"):
        self._pool_features = pool_features.clone()
        self._feedback_channel = feedback_channel

    @property
    def seen(self) -> frozenset[int]:
        return frozenset(self._feedback_channel.seen)

    def candidates(self) -> torch.Tensor:
        return self._pool_features.clone()

    def observe(self, index: int) -> torch.Tensor:
        if not 0 <= int(index) < len(self._pool_features):
            raise ValueError("ILLEGAL_ACTION")
        return self._pool_features[int(index)].clone()

    def feedback(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._feedback_channel.feedback(index)

    def query(self, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        raise ValueError("AGENT_QUERY_CAPABILITY_DENIED")


class SkillWorldEvaluator:
    """A single-split query capability; development and sealed labels are separate objects."""
    def __init__(self, split: str, x: torch.Tensor, y: torch.Tensor):
        if split not in ("dev", "sealed"):
            raise ValueError("INVALID_EVALUATOR_SPLIT")
        self.split = split
        self._x, self._y = x.clone(), y.clone()

    def query(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.split != "dev":
            raise ValueError("EVALUATOR_QUERY_CAPABILITY_DENIED")
        return self._x.clone(), self._y.clone()


class SealedEvaluator:
    """Final-only evaluator, held by the experiment runner and never passed to the agent."""
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self._x, self._y = x.clone(), y.clone()

    def evaluate(self, predictor: nn.Module, device: torch.device) -> dict[str, float]:
        with torch.no_grad():
            logits = predictor(self._x.to(device)).cpu()
            labels = self._y
            return {"accuracy": float((logits.argmax(-1) == labels).float().mean()),
                    "balanced_accuracy": balanced_accuracy(logits, labels),
                    "cross_entropy": float(F.cross_entropy(logits, labels))}


class FeedbackChannel:
    """Only selected pool labels are available through this feedback capability."""
    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        self._features, self._labels = features.clone(), labels.clone()
        self.seen: set[int] = set()

    def feedback(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        index = int(index)
        if not 0 <= index < len(self._features) or index in self.seen:
            raise ValueError("ILLEGAL_OR_REPEATED_ACTION")
        self.seen.add(index)
        return self._features[index].clone(), self._labels[index].clone()

def feature_sha(feature: torch.Tensor) -> str:
    # Canonical float32, little-endian, contiguous feature bytes only.
    array = np.asarray(feature.detach().cpu().numpy(), dtype="<f4").reshape(-1)
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


class SymbolObjectInterface:
    """Non-neural grounding hook: symbols refer to observed object features, never labels."""
    def __init__(self) -> None:
        self._objects: dict[str, torch.Tensor] = {}

    def bind(self, symbol: str, observed_feature: torch.Tensor) -> None:
        if not symbol or not isinstance(observed_feature, torch.Tensor) or observed_feature.shape != (8,):
            raise ValueError("INVALID_SYMBOL_OBJECT_BINDING")
        self._objects[str(symbol)] = observed_feature.detach().cpu().clone()

    def resolve(self, symbol: str) -> torch.Tensor:
        if symbol not in self._objects:
            raise KeyError(symbol)
        return self._objects[symbol].clone()

    def state_dict(self) -> dict[str, list[float]]:
        return {key: value.tolist() for key, value in sorted(self._objects.items())}

    def load_state_dict(self, state: dict[str, list[float]]) -> None:
        self._objects = {k: torch.tensor(v, dtype=torch.float32) for k, v in state.items()}


@dataclass
class SkillRecord:
    context: float
    world_seed: int
    updates: int = 0
    curve: list[dict[str, float | int]] | None = None

    def __post_init__(self) -> None:
        if self.curve is None:
            self.curve = []


class ContinuousAgent:
    """Keeps a single 403-parameter Predictor+Explorer across world changes."""
    def __init__(self, seed: int, learning_rate: float = 0.1, device: str = "cpu"):
        self.seed, self.learning_rate = int(seed), float(learning_rate)
        self.device = torch.device(device)
        torch.manual_seed(self.seed)
        self.predictor = Predictor().to(self.device)
        self.explorer = Explorer().to(self.device)
        self.optimizer = torch.optim.SGD(self.predictor.parameters(), lr=self.learning_rate)
        self.explorer_optimizer = torch.optim.SGD(self.explorer.parameters(), lr=0.05)
        self.selective_gate: Gate | None = None
        self.replay_buffer: list[dict[str, Any]] = []
        self.replay_capacity = 32
        self.update_audit: dict[str, Any] = {}
        self.symbols = SymbolObjectInterface()
        self.skills: list[SkillRecord] = []
        self.history: list[dict[str, Any]] = []

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for m in (self.predictor, self.explorer) for p in m.parameters())

    def load_selective_gate(self, checkpoint: Path, expected_sha256: str | None = None) -> str:
        raw = Path(checkpoint).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if expected_sha256 and digest != expected_sha256:
            raise RuntimeError("SELECTIVE_GATE_CHECKPOINT_HASH_MISMATCH")
        gate = Gate(dynamic=True).to(self.device)
        gate.load_state_dict(torch.load(Path(checkpoint), map_location=self.device, weights_only=True), strict=True)
        gate.eval()
        for parameter in gate.parameters():
            parameter.requires_grad_(False)
        self.selective_gate = gate
        return digest

    def learn_feedback(self, x: torch.Tensor, y: torch.Tensor, budget: Budget, *,
                       strategy: str = "ordinary_sgd", context: float = 0.0,
                       replay_batch_size: int = 4) -> dict[str, Any]:
        """One real SGD update; requires an explicitly supplied, authorized ledger."""
        if strategy not in ("ordinary_sgd", "experience_replay", "selective_update"):
            raise ValueError(f"UNKNOWN_CONTINUAL_STRATEGY:{strategy}")
        if strategy == "selective_update" and self.selective_gate is None:
            raise RuntimeError("SELECTIVE_GATE_NOT_LOADED")
        batch_x = [x.detach().to(self.device).reshape(1, 8)]
        batch_y = [y.detach().to(self.device).reshape(1)]
        replayed = 0
        if strategy == "experience_replay" and self.replay_buffer:
            count = min(int(replay_batch_size), len(self.replay_buffer))
            indices = torch.randperm(len(self.replay_buffer))[:count].tolist()
            for index in indices:
                batch_x.append(self.replay_buffer[index]["x"].to(self.device).reshape(1, 8))
                batch_y.append(self.replay_buffer[index]["y"].to(self.device).reshape(1))
            replayed = count
        self.optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(self.predictor(torch.cat(batch_x)), torch.cat(batch_y))
        loss.backward()
        grad_norm = torch.sqrt(sum(p.grad.detach().square().sum() for p in self.predictor.parameters()
                                   if p.grad is not None))
        gates = [1.0, 1.0]
        if strategy == "selective_update":
            state = torch.stack((loss.detach(), grad_norm.detach()))
            with torch.no_grad():
                gates = self.selective_gate(state).detach().cpu().tolist()
            for name, parameter in self.predictor.named_parameters():
                group_gate = gates[0] if name.startswith("net.0.") else gates[1]
                if parameter.grad is not None:
                    parameter.grad.mul_(float(group_gate))
        before = [p.detach().clone() for p in self.predictor.parameters()]
        budget.reserve(1, "zero6_task_update")
        try:
            self.optimizer.step()
            budget.complete(1, "zero6_task_update")
        except BaseException:
            # A reserved-but-uncompleted update remains visible as uncertain cost.
            budget.mark_uncertain(1, "zero6_update_exception")
            raise
        delta = torch.sqrt(sum((p.detach() - old).square().sum() for p, old in zip(self.predictor.parameters(), before)))
        per_group = {"hidden": 0.0, "output": 0.0}
        for (name, parameter), old in zip(self.predictor.named_parameters(), before):
            group = "hidden" if name.startswith("net.0.") else "output"
            per_group[group] += float((parameter.detach() - old).square().sum())
        per_group = {key: value ** 0.5 for key, value in per_group.items()}
        self.replay_buffer.append({"x": x.detach().cpu().clone(), "y": y.detach().cpu().clone(),
                                  "context": float(context)})
        if len(self.replay_buffer) > self.replay_capacity:
            self.replay_buffer = self.replay_buffer[-self.replay_capacity:]
        self.update_audit = {"loss": float(loss.detach()), "gradient_norm": float(grad_norm),
                             "gate_hidden": float(gates[0]), "gate_output": float(gates[1]),
                             "hidden_weight_delta": per_group["hidden"],
                             "output_weight_delta": per_group["output"],
                             "weight_delta_norm": float(delta), "replay_examples": replayed,
                             "examples_in_update": 1 + replayed}
        return dict(self.update_audit)

    @torch.no_grad()
    def evaluate(self, evaluator: SkillWorldEvaluator) -> dict[str, float]:
        x, y = evaluator.query()
        logits = self.predictor(x.to(self.device))
        return {"accuracy": float((logits.argmax(-1).cpu() == y).float().mean()),
                "balanced_accuracy": balanced_accuracy(logits.cpu(), y),
                "cross_entropy": float(F.cross_entropy(logits.cpu(), y))}

    def record_skill_curve(self, world_seed: int, context: float, evaluator: SkillWorldEvaluator,
                           updates: int) -> dict[str, Any]:
        metrics = self.evaluate(evaluator)
        row = {"updates": int(updates), **metrics}
        self.history.append({"event": "evaluation", "world_seed": world_seed,
                             "context": context, **row})
        return row

    def adapt_skill(self, agent_view: SkillWorldAgentView, evaluator: SkillWorldEvaluator,
                    world_seed: int, context: float, strategy: str, budget: Budget, *,
                    updates: int = 8, stream_seed: int = 0,
                    authorization_confirmed: bool = False) -> dict[str, Any]:
        """One skill phase; caller retains this same agent for the next world."""
        if not authorization_confirmed:
            raise RuntimeError("ZERO6_TRAINING_NOT_AUTHORIZED")
        if updates <= 0:
            raise ValueError("UPDATES_MUST_BE_POSITIVE")
        record = SkillRecord(context=context, world_seed=world_seed)
        record.curve.append({"updates": 0, **self.evaluate(evaluator)})
        seen = torch.zeros(len(agent_view.candidates()), dtype=torch.bool)
        if updates > len(seen):
            raise ValueError("UPDATES_EXCEED_CANDIDATE_POOL")
        stream = torch.randperm(len(seen), generator=torch.Generator().manual_seed(stream_seed))
        for step in range(updates):
            action = int(stream[step])
            x, y = agent_view.feedback(action)
            seen[action] = True
            update = self.learn_feedback(x, y, budget, strategy=strategy, context=context)
            record.updates += 1
            point = {"updates": record.updates, **self.evaluate(evaluator)}
            record.curve.append(point)
            self.history.append({"event": "feedback_update", "world_seed": world_seed,
                                 "context": context, "step": step, "action": action,
                                 "label": int(y), **update,
                                 "strategy": strategy, "evaluation": point})
        self.skills.append(record)
        return {"skill": vars(record), "final": record.curve[-1]}

    def evaluate_skills(self, targets: list[tuple[int, float, SkillWorldEvaluator]]) -> list[dict[str, Any]]:
        """Measures retention/interference for each previously learned world."""
        results = []
        for world_seed, context, evaluator in targets:
            results.append({"world_seed": world_seed, "context": context,
                            **self.evaluate(evaluator)})
        return results


def run_continual_sequence(agent: ContinuousAgent, worlds: list[SkillWorld], strategy: str,
                          budget: Budget, *, updates_per_world: int = 8,
                          checkpoint_dir: Path | None = None,
                          protocol_sha256: str = "",
                          authorization_confirmed: bool = False,
                          start_world_index: int = 0,
                          sealed_evaluation_enabled: bool = False,
                          development_gate_passed: bool = False) -> dict[str, Any]:
    """A→B→novel-world sequence with one retained network and per-skill curves."""
    if not authorization_confirmed:
        raise RuntimeError("ZERO6_TRAINING_NOT_AUTHORIZED")
    if sealed_evaluation_enabled and not development_gate_passed:
        raise RuntimeError("SEALED_EVALUATION_REQUIRES_DEVELOPMENT_GATE")
    if len({world.context for world in worlds}) != len(worlds):
        raise ValueError("WORLD_CONTEXTS_MUST_BE_DISTINCT")
    if (not 0 <= start_world_index <= len(worlds) or len(agent.skills) < start_world_index or
            (start_world_index == 0 and bool(agent.skills))):
        raise RuntimeError("CHECKPOINT_WORLD_POSITION_INVALID")
    for index in range(start_world_index):
        if (agent.skills[index].world_seed != worlds[index].world_seed or
                agent.skills[index].context != worlds[index].context):
            raise RuntimeError("CHECKPOINT_SKILL_SEQUENCE_MISMATCH")
    curves = {str(worlds[i].world_seed): {"skill": vars(agent.skills[i]),
              "final": agent.skills[i].curve[-1]} for i in range(start_world_index)}
    retention = [event["payload"] for event in agent.history
                 if event.get("event") == "retention_checkpoint" and
                 event["payload"].get("after_world_index", -1) < start_world_index]
    full_matrix = [event["payload"] for event in agent.history
                   if event.get("event") == "full_matrix_checkpoint" and
                   event["payload"].get("after_world_index", -1) < start_world_index]
    for index in range(start_world_index, len(worlds)):
        cpu_start, wall_start = time.process_time(), time.perf_counter()
        world = worlds[index]
        curves[str(world.world_seed)] = agent.adapt_skill(
            world.agent, world.evaluator, world.world_seed, world.context,
            strategy, budget, updates=updates_per_world,
            stream_seed=world.world_seed + agent.seed,
            authorization_confirmed=authorization_confirmed)
        targets = [(w.world_seed, w.context, w.evaluator) for w in worlds[:index + 1]]
        current_metrics = agent.evaluate_skills(targets)
        per_skill = []
        for skill_index, metric in enumerate(current_metrics):
            learned_curve = curves[str(worlds[skill_index].world_seed)]["skill"]["curve"]
            learned_reference = learned_curve[-1]["balanced_accuracy"]
            change = metric["balanced_accuracy"] - learned_reference
            per_skill.append({**metric, "balanced_accuracy_after_acquisition": learned_reference,
                              "change_since_acquisition": change,
                              "forgetting_from_acquisition": max(0.0, -change)})
        retention.append({"after_world_index": index, "per_skill": per_skill})
        agent.history.append({"event": "retention_checkpoint",
                              "payload": retention[-1]})
        all_targets = [(w.world_seed, w.context, w.evaluator) for w in worlds]
        matrix_row = {"after_world_index": index,
                      "per_skill": agent.evaluate_skills(all_targets)}
        full_matrix.append(matrix_row)
        agent.history.append({"event": "full_matrix_checkpoint", "payload": matrix_row})
        mem = psutil.Process().memory_info()
        agent.history.append({"event": "world_resource", "after_world_index": index,
                              "cpu_seconds": time.process_time() - cpu_start,
                              "wall_seconds": time.perf_counter() - wall_start,
                              "peak_rss_bytes": getattr(mem, "peak_wset", mem.rss)})
        if checkpoint_dir is not None:
            save_world_checkpoint(Path(checkpoint_dir) / f"after_world_{index}.pt", agent,
                                  phase="continual", strategy=strategy, world_index=index,
                                  next_world_index=index + 1, world_seed=world.world_seed,
                                  budget=budget, protocol_sha256=protocol_sha256)
    dev_targets = [(w.world_seed, w.context, w.evaluator) for w in worlds]
    feedback_events = [e for e in agent.history if e.get("event") == "feedback_update"]
    resource_events = [e for e in agent.history if e.get("event") == "world_resource"]
    result = {"strategy": strategy, "world_order": [w.world_seed for w in worlds],
            "skill_curves": curves, "retention_by_world": retention,
            "development_matrix_5x5": full_matrix,
            "resource_audit": {"task_parameter_updates": len(feedback_events),
                               "feedback_samples": len(feedback_events),
                               "replayed_examples": sum(e["replay_examples"] for e in feedback_events),
                               "total_examples_in_updates": sum(e["examples_in_update"] for e in feedback_events),
                               "cpu_seconds": sum(e["cpu_seconds"] for e in resource_events),
                               "wall_seconds": sum(e["wall_seconds"] for e in resource_events),
                               "peak_rss_bytes": max((e["peak_rss_bytes"] for e in resource_events), default=0)},
            "final_development_per_skill": agent.evaluate_skills(dev_targets)}
    if sealed_evaluation_enabled:
        result["final_sealed_per_skill"] = evaluate_sealed_skills(agent, worlds)
    else:
        result["final_sealed_per_skill"] = "NOT_READ_DEVELOPMENT_GATE_PENDING"
    return result


def run_paired_screening(budget: Budget, frozen_policies: dict[int, Path], *,
                         authorization_confirmed: bool = False,
                         checkpoint_root: Path | None = None,
                         protocol_sha256: str = "") -> dict[str, Any]:
    """The requested 360-update CPU screen using frozen historical mechanisms."""
    if not authorization_confirmed:
        raise RuntimeError("ZERO6_TRAINING_NOT_AUTHORIZED")
    plan = 3 * 5 * 3 * 8
    current = budget.snapshot()
    manifest = json.loads((Path(__file__).with_name("ZERO6_TASK_MANIFEST.json")).read_text(encoding="utf-8"))
    expected_gates = manifest["frozen_selective_gates"]
    for seed, path in frozen_policies.items():
        expected = expected_gates.get(str(seed), {}).get("sha256")
        if expected is None or hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise RuntimeError("SELECTIVE_GATE_MANIFEST_HASH_MISMATCH")
    manifest_sha = hashlib.sha256(Path(__file__).with_name("ZERO6_TASK_MANIFEST.json").read_bytes()).hexdigest()
    if protocol_sha256 and protocol_sha256 != manifest_sha:
        raise RuntimeError("ZERO6_PROTOCOL_HASH_MISMATCH")
    protocol_sha256 = manifest_sha
    strategies = ("ordinary_sgd", "experience_replay", "selective_update")
    contexts = (-1.0, -0.5, 0.0, 0.5, 1.0)
    seeds = (101, 202, 303)
    worlds_by_session = {
        (strategy, seed): [SkillWorld(610000 + seed * 10 + index, context)
                           for index, context in enumerate(contexts, start=1)]
        for strategy in strategies for seed in seeds
    }
    checkpoint_paths: dict[tuple[str, int], Path | None] = {}
    checkpointed_updates = 0
    if checkpoint_root is not None:
        for strategy in strategies:
            for seed in seeds:
                session_root = Path(checkpoint_root) / f"seed_{seed}" / strategy
                candidates = sorted(session_root.glob("after_world_*.pt"),
                                    key=lambda item: int(item.stem.rsplit("_", 1)[1]))
                latest = candidates[-1] if candidates else None
                checkpoint_paths[(strategy, seed)] = latest
                if latest is not None:
                    saved = torch.load(latest, map_location="cpu", weights_only=False)
                    next_index = int(saved["next_world_index"])
                    worlds = worlds_by_session[(strategy, seed)]
                    if (saved.get("authorization_id") != budget.auth_id or
                            saved.get("protocol_sha256") != protocol_sha256 or
                            saved.get("strategy") != strategy or
                            not 1 <= next_index <= len(worlds) or
                            saved.get("world_seed") != worlds[next_index - 1].world_seed):
                        raise RuntimeError("ZERO6_RESUME_CHECKPOINT_IDENTITY_MISMATCH")
                    checkpointed_updates += next_index * 8
    if checkpoint_root is not None:
        if (current["completed"] != checkpointed_updates or current["reserved"] != current["completed"] or
                current.get("uncertain", 0) != 0 or current.get("failed", 0) != 0):
            raise RuntimeError("ZERO6_UNCHECKPOINTED_OR_UNCERTAIN_BUDGET_CANNOT_RESUME")
    remaining_plan = plan - checkpointed_updates
    if remaining_plan < 0 or current["reserved"] + remaining_plan > current["limit"]:
        raise RuntimeError("ZERO6_BUDGET_INSUFFICIENT")
    rows: list[dict[str, Any]] = []
    baseline_sessions = []
    for strategy in strategies:
        sessions = []
        for seed in seeds:
            agent = ContinuousAgent(seed, learning_rate=0.1, device="cpu")
            if strategy == "selective_update":
                expected = expected_gates[str(seed)]["sha256"]
                agent.load_selective_gate(frozen_policies[seed], expected_sha256=expected)
            worlds = worlds_by_session[(strategy, seed)]
            folder = Path(checkpoint_root) / f"seed_{seed}" / strategy if checkpoint_root else None
            start_world_index = 0
            checkpoint = checkpoint_paths.get((strategy, seed))
            if checkpoint is not None:
                restored = load_world_checkpoint(
                    checkpoint, agent, budget=budget,
                    expected_protocol_sha256=protocol_sha256,
                    allow_later_budget_state=True)
                if restored["strategy"] != strategy:
                    raise RuntimeError("ZERO6_RESUME_STRATEGY_MISMATCH")
                start_world_index = int(restored["next_world_index"])
            result = run_continual_sequence(
                agent, worlds, strategy, budget, updates_per_world=8,
                checkpoint_dir=folder, protocol_sha256=protocol_sha256,
                authorization_confirmed=True, sealed_evaluation_enabled=False,
                start_world_index=start_world_index)
            row = {"seed": seed, **result}
            rows.append(row)
            sessions.append((row, agent, worlds))
        if strategy == "ordinary_sgd":
            baseline_sessions = sessions
            seed_gains = []
            for row, _, _ in sessions:
                deltas = []
                for curve in row["skill_curves"].values():
                    points = curve["skill"]["curve"]
                    deltas.append(points[-1]["balanced_accuracy"] - points[0]["balanced_accuracy"])
                seed_gains.append(sum(deltas) / len(deltas))
            gate_passed = all(gain >= 0.05 for gain in seed_gains)
            if not gate_passed:
                final = budget.snapshot()
                return {"status": "DEV_GATE_FAILED_STOP_BEFORE_SEALED",
                        "development_gate": {"criterion": "each paired-seed mean skill balanced-accuracy gain >= 0.05",
                                              "seed_gains": seed_gains, "passed": False},
                        "rows": rows, "actual_completed_updates": final["completed"] - current["completed"],
                        "cumulative_completed_updates": final["completed"],
                        "planned_max_updates": plan}
            for row, session_agent, session_worlds in baseline_sessions:
                row["final_sealed_per_skill"] = evaluate_sealed_skills(session_agent, session_worlds)
        else:
            # Development sufficiency has already passed on the ordinary continual-SGD control.
            for row, session_agent, session_worlds in sessions:
                row["final_sealed_per_skill"] = evaluate_sealed_skills(session_agent, session_worlds)
    final = budget.snapshot()
    actual = final["completed"] - current["completed"]
    if final["completed"] != plan:
        raise RuntimeError("ZERO6_ACTUAL_UPDATE_COUNT_MISMATCH")
    return {"status": "SCREENING_COMPLETE", "development_gate": {"passed": True,
            "criterion": "each paired-seed mean skill balanced-accuracy gain >= 0.05"},
            "rows": rows, "actual_completed_updates": actual,
            "cumulative_completed_updates": final["completed"], "planned_updates": plan}


def balanced_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred = logits.argmax(-1)
    scores = []
    for label in (0, 1):
        mask = labels == label
        if bool(mask.any()):
            scores.append(float((pred[mask] == labels[mask]).float().mean()))
    return float(sum(scores) / len(scores)) if scores else 0.0


def evaluate_sealed_skills(agent: ContinuousAgent, worlds: list[SkillWorld]) -> list[dict[str, Any]]:
    """Final-only evaluation capability, called after the development gate."""
    return [{"world_seed": world.world_seed, "context": world.context,
             **world.sealed_evaluator.evaluate(agent.predictor, agent.device)}
            for world in worlds]


def save_world_checkpoint(path: Path, agent: ContinuousAgent, *, phase: str, strategy: str,
                          world_index: int, next_world_index: int, world_seed: int,
                          budget: Budget, protocol_sha256: str) -> None:
    """Atomic world-boundary save; refuses ledger rollback relative to prior checkpoint."""
    path = Path(path)
    budget_state = budget.snapshot()
    if budget_state["authorization_id"] != budget.auth_id:
        raise RuntimeError("CHECKPOINT_BUDGET_AUTH_MISMATCH")
    previous = path.with_suffix(path.suffix + ".previous.json")
    if previous.exists():
        old = json.loads(previous.read_text(encoding="utf-8"))
        counters = ("reserved", "completed", "uncertain", "failed")
        if any(budget_state.get(key, 0) < old.get(key, 0) for key in counters):
            raise RuntimeError("CHECKPOINT_BUDGET_ROLLBACK")
    data = {
        "format": "zero6-world-boundary-v1", "phase": phase, "strategy": strategy,
        "world_index": int(world_index), "next_world_index": int(next_world_index),
        "world_seed": int(world_seed), "protocol_sha256": protocol_sha256,
        "agent_seed": agent.seed, "learning_rate": agent.learning_rate,
        "authorization_id": budget_state["authorization_id"], "budget": budget_state,
        "predictor": agent.predictor.state_dict(), "explorer": agent.explorer.state_dict(),
        "optimizer": agent.optimizer.state_dict(), "explorer_optimizer": agent.explorer_optimizer.state_dict(),
        "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(), "skills": [vars(x) for x in agent.skills],
        "history": agent.history, "symbols": agent.symbols.state_dict(),
        "predictor_requires_grad": [p.requires_grad for p in agent.predictor.parameters()],
        "explorer_requires_grad": [p.requires_grad for p in agent.explorer.parameters()],
        "selective_gate": agent.selective_gate.state_dict() if agent.selective_gate is not None else None,
        "replay_capacity": agent.replay_capacity,
        "replay_buffer": agent.replay_buffer,
        "update_audit": agent.update_audit,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(data, temp_name)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    previous.write_text(json.dumps({"authorization_id": budget_state["authorization_id"],
                                    "limit": budget_state["limit"],
                                    "reserved": budget_state["reserved"],
                                    "completed": budget_state["completed"],
                                    "uncertain": budget_state.get("uncertain", 0),
                                    "failed": budget_state.get("failed", 0)}, indent=2), encoding="utf-8")


def load_world_checkpoint(path: Path, agent: ContinuousAgent, *, budget: Budget,
                          expected_protocol_sha256: str,
                          allow_later_budget_state: bool = False) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=agent.device, weights_only=False)
    state = budget.snapshot()
    if checkpoint["protocol_sha256"] != expected_protocol_sha256:
        raise RuntimeError("CHECKPOINT_PROTOCOL_MISMATCH")
    if checkpoint["authorization_id"] != state["authorization_id"]:
        raise RuntimeError("CHECKPOINT_BUDGET_AUTH_MISMATCH")
    saved = checkpoint["budget"]
    accounting_fields = ("authorization_id", "limit", "reserved", "completed", "uncertain", "failed")
    if allow_later_budget_state:
        same_identity = state["authorization_id"] == saved["authorization_id"] and state["limit"] == saved["limit"]
        counters_are_monotonic = all(state.get(key, 0) >= saved.get(key, 0)
                                     for key in ("reserved", "completed"))
        no_new_uncertain_cost = (state.get("uncertain", 0) == saved.get("uncertain", 0) and
                                 state.get("failed", 0) == saved.get("failed", 0))
        if not (same_identity and counters_are_monotonic and no_new_uncertain_cost):
            raise RuntimeError("CHECKPOINT_BUDGET_STATE_MISMATCH")
    elif any(state.get(key, 0) != saved.get(key, 0) for key in accounting_fields):
        raise RuntimeError("CHECKPOINT_BUDGET_STATE_MISMATCH")
    agent.seed = int(checkpoint["agent_seed"])
    agent.learning_rate = float(checkpoint["learning_rate"])
    agent.predictor.load_state_dict(checkpoint["predictor"])
    agent.explorer.load_state_dict(checkpoint["explorer"])
    agent.optimizer.load_state_dict(checkpoint["optimizer"])
    agent.explorer_optimizer.load_state_dict(checkpoint["explorer_optimizer"])
    if checkpoint["selective_gate"] is not None:
        gate = Gate(dynamic=True).to(agent.device)
        gate.load_state_dict(checkpoint["selective_gate"], strict=True)
        gate.eval()
        for parameter in gate.parameters():
            parameter.requires_grad_(False)
        agent.selective_gate = gate
    for parameter, required in zip(agent.predictor.parameters(), checkpoint["predictor_requires_grad"]):
        parameter.requires_grad_(required)
    for parameter, required in zip(agent.explorer.parameters(), checkpoint["explorer_requires_grad"]):
        parameter.requires_grad_(required)
    random.setstate(checkpoint["python_rng"])
    np.random.set_state(checkpoint["numpy_rng"])
    torch.set_rng_state(checkpoint["torch_rng"])
    agent.skills = [SkillRecord(**row) for row in checkpoint["skills"]]
    agent.history = checkpoint["history"]
    agent.symbols.load_state_dict(checkpoint["symbols"])
    agent.replay_capacity = int(checkpoint["replay_capacity"])
    agent.replay_buffer = checkpoint["replay_buffer"]
    agent.update_audit = checkpoint["update_audit"]
    return {k: checkpoint[k] for k in ("phase", "strategy", "world_index", "next_world_index", "world_seed",
                                       "agent_seed", "learning_rate")}


def minimum_budget_plan() -> dict[str, Any]:
    # Three paired seeds × five continuing worlds × three mechanisms × 8 updates.
    # The ordinary-SGD positive control gates whether any sealed labels are read.
    updates = 3 * 5 * 3 * 8
    return {"status": "REQUEST_ONLY_NOT_AUTHORIZED", "new_updates_requested": updates,
            "formula": "3 seeds x 5 worlds x 3 continual-learning mechanisms x 8 feedbacks",
            "strategies": ["ordinary_continual_sgd", "limited_experience_replay", "frozen_selective_update"],
            "world_order": ["A", "B", "C", "D", "E"],
            "per_skill_checkpoints": [0, 1, 2, 4, 8], "device": "cpu",
            "existing_zero5_remaining_140": "NOT_USED",
            "selective_update_source": "reuse frozen historical ZERO-4 G2 two-group gate per seed; ZERO-6 does not train it",
            "development_gate": "for each paired seed, ordinary-SGD mean per-skill development balanced-accuracy gain across A-E >= 0.05; otherwise stop before sealed evaluation",
            "limitation": "eight feedbacks/world is a screening length; skill acquisition is not yet established"}


def capability_matrix() -> dict[str, Any]:
    return {"parameter_count": {"predictor": 178, "explorer": 225, "combined": 403},
            "same_network_across_worlds": "IMPLEMENTED_NOT_TRAINED",
            "new_skill_acquisition": "NOT_RUN_BUDGET",
            "old_skill_retention": "NOT_RUN_BUDGET",
            "interference": "NOT_RUN_BUDGET",
            "unseen_task_adaptation": "NOT_RUN_BUDGET",
            "active_strategy_comparison": "NOT_RUN_BUDGET",
            "symbol_object_interface": "IMPLEMENTED_INTERFACE_ONLY",
            "checkpoint_recovery": "IMPLEMENTED_PENDING_TEST",
            "no_training_self_check": "PENDING"}
