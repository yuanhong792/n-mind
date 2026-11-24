"""Phase 1 evaluation primitives; historical evaluator behavior is untouched."""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch


def answer_metrics(log_probs: torch.Tensor, target: torch.Tensor,
                   *, candidate_log_likelihoods: Sequence[float] | None = None) -> dict:
    """Compute token NLL, sequence log likelihood, rank and character edit distance.

    ``log_probs`` is [T,V] and target is [T]. The caller supplies the tokenizer-level
    character strings for edit distance because tokenization may be sub-character.
    """
    if log_probs.ndim != 2 or target.ndim != 1 or log_probs.shape[0] != target.shape[0]:
        raise ValueError("log_probs 必须为 [T,V]，target 必须为同长 [T]")
    token_nll = (-log_probs.gather(1, target[:, None]).squeeze(1)).detach().cpu()
    seq_ll = float(-token_nll.sum())
    out = {"per_token_nll": [float(x) for x in token_nll],
           "answer_log_likelihood": seq_ll}
    if candidate_log_likelihoods is not None:
        vals = list(candidate_log_likelihoods)
        out["candidate_rank"] = 1 + sum(v > seq_ll for v in vals)
    return out


def levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def add_text_distance(row: dict, predicted: str, target: str) -> dict:
    row = dict(row)
    row["char_edit_distance"] = levenshtein(predicted, target)
    row["exact_match"] = predicted == target
    return row


def metric_metadata(*, torch_version: str, python_version: str, git_revision: str,
                    data_fingerprint: str, config_hash: str, protocol: str) -> dict:
    return {"torch_version": torch_version, "torch_build": torch_version,
            "python_version": python_version, "git_revision": git_revision,
            "data_fingerprint": data_fingerprint, "config_hash": config_hash,
            "protocol": protocol}
