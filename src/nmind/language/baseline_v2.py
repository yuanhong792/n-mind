"""Measured random-initialization baseline; no training and no oracle in model input."""
from __future__ import annotations
import torch
from .neural_language_v2 import LanguageV2Config, LanguageV2Model
from .tokenizer_v2 import TokenizerV2
from .curriculum import starter_tasks

def measure_untrained(seed: int = 20261123) -> dict:
    torch.manual_seed(seed); tok = TokenizerV2(); model = LanguageV2Model(LanguageV2Config()); model.eval()
    rows = []
    with torch.no_grad():
        for task in starter_tasks():
            ids = torch.tensor([tok.encode(task.prompt)], dtype=torch.long)
            logits, _ = model(ids)
            try:
                pred = tok.decode([int(logits[0, -1].argmax())])
            except UnicodeDecodeError:
                pred = "<invalid-byte>"
            rows.append({"task_id": task.task_id, "prediction": pred, "expected": task.expected, "exact_match": pred == task.expected})
    return {"status": "MEASURED_RANDOM_INITIALIZATION", "seed": seed, "optimizer_steps": 0,
            "tasks": rows, "exact_match": sum(r["exact_match"] for r in rows) / len(rows),
            "answer_leakage": False, "source": "neural_only_random_model"}
