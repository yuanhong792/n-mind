"""Offline, leakage-resistant diagnostics for Language V2."""
from __future__ import annotations
from typing import Any
from .data_v2 import build_counterfactual_pairs

def baseline_report() -> dict[str, Any]:
    cases = build_counterfactual_pairs()
    return {
        "status": "NOT_RUN",
        "reason": "No independent training budget authorized; random model is not scored as learned capability.",
        "counterfactual_pairs": len({c.pair_id for c in cases}),
        "counterfactual_cases": len(cases),
        "arms": {
            "neural_only": "NOT_RUN",
            "memory_assisted": "NOT_RUN",
            "knowledge_retrieval": "NOT_RUN",
            "tool": "NOT_RUN",
        },
        "metrics": {"exact_match": None, "binding_accuracy": None, "update_accuracy": None,
                    "negation_accuracy": None, "pronoun_accuracy": None,
                    "multiturn_accuracy": None, "free_generation": None},
    }
