"""ZERO-7 neural symbol grounding; import and preflight never update parameters."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .budget_ledger import Budget
from .zero6_continual import ContinuousAgent, SkillWorld, feature_sha, load_world_checkpoint

ROOT = Path(__file__).parent
ZERO6 = ROOT / "results" / "zero6"
ZERO6_PROTOCOL_SHA = hashlib.sha256((ROOT / "ZERO6_TASK_MANIFEST.json").read_bytes()).hexdigest()
TOKENS = 12
CLASSES = 3


class SymbolWorld:
    """Private token meanings and three object relations: attribute, form, action."""
    def __init__(self, seed: int, n_pool: int = 192, n_query: int = 96):
        self.seed = int(seed)
        rule_rng = np.random.default_rng(seed)
        alphabet = np.array(list("abcdefghjkmnpqrstuvwxyz"))
        self.vocabulary = tuple("".join(rule_rng.choice(alphabet, size=4)) for _ in range(TOKENS))
        self._rules = torch.tensor(rule_rng.standard_normal((3, 3, 7)), dtype=torch.float32)
        meanings = rule_rng.permutation(9)
        self._meaning = torch.tensor(np.concatenate((meanings, meanings[:3])), dtype=torch.long)
        self._canonical = {int(value): index for index, value in enumerate(meanings)}
        self._sets = {}
        for i, split in enumerate(("pool", "dev", "sealed")):
            rng = np.random.default_rng(seed * 101 + i + 19)
            raw = torch.tensor(rng.standard_normal((n_pool if split == "pool" else n_query, 7)), dtype=torch.float32)
            x = torch.cat((raw, torch.full((len(raw), 1), 1.5)), dim=1)
            # Labels stay in this environment; no token table enters the neural model.
            relation_values = torch.stack([(raw @ rule.T).argmax(1) for rule in self._rules])
            matches = torch.stack([(relation_values[int(m)//3] == int(m)%3).long() for m in self._meaning])
            self._sets[split] = (x, matches, relation_values)
        self._seen = set()

    def symbol_text(self, token: int) -> str:
        if not 0 <= token < TOKENS:
            raise ValueError("UNKNOWN_SYMBOL_ID")
        return self.vocabulary[token]

    def candidate_features(self) -> torch.Tensor:
        return self._sets["pool"][0].clone()

    def feedback(self, token: int, object_index: int) -> tuple[torch.Tensor, int, torch.Tensor]:
        if not (0 <= token < TOKENS and 0 <= object_index < len(self._sets["pool"][0])):
            raise ValueError("ILLEGAL_SYMBOL_ACTION")
        key = (token, object_index)
        if key in self._seen:
            raise ValueError("REPEATED_SYMBOL_ACTION")
        self._seen.add(key)
        meaning = int(self._meaning[token]); relation = meaning // 3
        value = int(self._sets["pool"][2][relation, object_index])
        canonical = self._canonical[relation * 3 + value]
        return (self._sets["pool"][0][object_index].clone(), token,
                self._sets["pool"][1][token, object_index].clone(), relation, canonical)

    def query(self, split: str, tokens: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if split not in ("dev", "sealed"):
            raise ValueError("QUERY_SPLIT_DENIED")
        x, y, _ = self._sets[split]
        return x.repeat(len(tokens), 1), torch.tensor(tokens).repeat_interleave(len(x)), y[list(tokens)].reshape(-1).clone()

    def reverse_query(self, split: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if split not in ("dev", "sealed"):
            raise ValueError("QUERY_SPLIT_DENIED")
        x, _, values = self._sets[split]
        relations = torch.arange(3).repeat_interleave(len(x))
        targets = torch.tensor([self._canonical[r * 3 + int(v)]
                                for r in range(3) for v in values[r]], dtype=torch.long)
        return x.repeat(3, 1), relations, targets

    def audit(self) -> dict:
        hashes = {split: {feature_sha(x) for x in pair[0]} for split, pair in self._sets.items()}
        return {"seed": self.seed, "counts": {k: len(v) for k, v in hashes.items()},
                "intersections": {f"{a}:{b}": len(hashes[a] & hashes[b])
                                  for a, b in (("pool", "dev"), ("pool", "sealed"), ("dev", "sealed"))},
                "hidden_rule_sha256": hashlib.sha256(self._rules.numpy().tobytes()).hexdigest()}


class SymbolHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.token = nn.Embedding(TOKENS, 8)
        self.fusion = nn.Sequential(nn.Linear(24, 16), nn.Tanh(), nn.Linear(16, 2))
        self.reverse = nn.Sequential(nn.Linear(19, 16), nn.Tanh(), nn.Linear(16, 9))

    def forward(self, features: torch.Tensor, tokens: torch.Tensor, encoder: nn.Module) -> torch.Tensor:
        object_state = torch.tanh(encoder(features))
        return self.fusion(torch.cat((object_state, self.token(tokens)), dim=-1))

    def name(self, features: torch.Tensor, relations: torch.Tensor, encoder: nn.Module) -> torch.Tensor:
        object_state = torch.tanh(encoder(features))
        return self.reverse(torch.cat((object_state, F.one_hot(relations, 3).float()), dim=-1))


class SymbolAgent:
    def __init__(self, seed: int, inherited: bool, replay: bool = False):
        self.seed, self.inherited, self.replay = seed, inherited, replay
        self.core = ContinuousAgent(seed)
        self.source_checkpoint = None
        if inherited:
            checkpoint = ZERO6 / "checkpoints" / f"seed_{seed}" / "experience_replay" / "after_world_4.pt"
            budget = Budget("zero6-20261125-360-v1", 360, ZERO6 / "ZERO6_BUDGET_STATE.json")
            meta = load_world_checkpoint(checkpoint, self.core, budget=budget,
                                         expected_protocol_sha256=ZERO6_PROTOCOL_SHA,
                                         allow_later_budget_state=True)
            if meta["next_world_index"] != 5 or self.core.seed != seed:
                raise RuntimeError("ZERO6_CHECKPOINT_NOT_FINAL")
            self.source_checkpoint = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        torch.manual_seed(seed + 7000)
        self.head = SymbolHead()
        self.optimizer = torch.optim.SGD(list(self.core.predictor.parameters()) + list(self.head.parameters()), lr=0.05)

    def logits(self, features: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        return self.head(features, tokens, self.core.predictor.net[0])

    @torch.no_grad()
    def evaluate(self, world: SymbolWorld, split: str, tokens: tuple[int, ...]) -> dict:
        x, t, y = world.query(split, tokens)
        logits = self.logits(x, t)
        return {"accuracy": float((logits.argmax(1) == y).float().mean()),
                "cross_entropy": float(F.cross_entropy(logits, y))}

    @torch.no_grad()
    def evaluate_reverse(self, world: SymbolWorld, split: str) -> dict:
        x, relations, targets = world.reverse_query(split)
        logits = self.head.name(x, relations, self.core.predictor.net[0])
        return {"free_symbol_exact_accuracy": float((logits.argmax(1) == targets).float().mean()),
                "cross_entropy": float(F.cross_entropy(logits, targets))}

    def feedback_update(self, sample: tuple[torch.Tensor, int, torch.Tensor, int, int], budget: Budget) -> dict:
        x, token, label, relation, canonical = sample
        self.optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(self.logits(x[None], torch.tensor([token])), label[None])
        reverse_logits = self.head.name(x[None], torch.tensor([relation]), self.core.predictor.net[0])
        loss = loss + F.cross_entropy(reverse_logits, torch.tensor([canonical]))
        replayed = 0
        if self.replay and self.core.replay_buffer:
            old = self.core.replay_buffer[-4:]
            old_x = torch.stack([row["x"] for row in old])
            old_y = torch.stack([row["y"] for row in old])
            loss = loss + 0.25 * F.cross_entropy(self.core.predictor(old_x), old_y)
            replayed = len(old)
        loss.backward()
        before = [p.detach().clone() for group in (self.core.predictor, self.head) for p in group.parameters()]
        budget.reserve(1, "zero7_symbol_feedback_update")
        try:
            self.optimizer.step()
            budget.complete(1, "zero7_symbol_feedback_update")
        except BaseException:
            budget.mark_uncertain(1, "zero7_update_exception")
            raise
        after = [p for group in (self.core.predictor, self.head) for p in group.parameters()]
        return {"loss": float(loss.detach()), "weight_delta": float(torch.sqrt(sum(((a.detach()-b)**2).sum() for a,b in zip(after,before)))),
                "feedback_samples": 1, "replayed_examples": replayed}

    @torch.no_grad()
    def old_skill_accuracy(self, seed: int, *, sealed: bool = False) -> float:
        worlds = [SkillWorld(610000 + seed * 10 + i, context)
                  for i, context in enumerate((-1., -.5, 0., .5, 1.), 1)]
        if sealed:
            return sum(w.sealed_evaluator.evaluate(self.core.predictor, self.core.device)["balanced_accuracy"] for w in worlds) / 5
        return sum(self.core.evaluate(w.evaluator)["balanced_accuracy"] for w in worlds) / 5
