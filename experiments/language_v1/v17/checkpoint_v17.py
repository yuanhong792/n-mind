"""V171 P1: checkpoint helpers enforcing the correct Best/Final save order.

Root cause of the V1.7 loss: `final_state` was captured AFTER restoring
`best_state`, so "final" silently became the best (= parent) weights.
These helpers capture last-step weights first, persist FINAL, then restore and
persist BEST. All writes are atomic (tmp + os.replace) so a crashed run keeps
its most recent valid checkpoint. Stable files are never referenced here.
"""
from __future__ import annotations
import hashlib, os
from pathlib import Path

import torch

from nmind.language.neural_language_v2 import save_language_v2

FINAL_NAME = "candidate_v17_final.pt"
BEST_NAME = "candidate_v17_best.pt"
LAST_NAME = "candidate_v17_last.pt"


def capture_state(model) -> dict:
    """Snapshot all params/buffers as CPU clones (decoupled from later restores)."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_save(path: str | Path, model, tokenizer, *, steps: int = 0,
                seed: int = 0, manifest_hash: str = "") -> str:
    """Write to `<name>.tmp` then os.replace -> never leave a half-written ckpt."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    save_language_v2(tmp, model, tokenizer, steps=steps, seed=seed,
                     manifest_hash=manifest_hash)
    os.replace(tmp, p)
    return sha256(p)


def save_periodic(model, tokenizer, out_dir: str | Path, *, step: int,
                  seed: int = 0, manifest_hash: str = "") -> str:
    """Persist the weights at an eval point; an interrupted run keeps this valid ckpt."""
    return atomic_save(Path(out_dir) / LAST_NAME, model, tokenizer,
                       steps=step, seed=seed, manifest_hash=manifest_hash)


def save_final_and_best(model, tokenizer, out_dir: str | Path, *,
                        final_state: dict, best_state: dict,
                        final_step: int, best_step: int,
                        seed: int = 0, manifest_hash: str = "") -> dict:
    """Correct order: persist last-step FINAL first, then restore and persist BEST.

    `model` must still hold the last training-step weights on entry.
    `final_state` must have been captured (capture_state) BEFORE any best restore.
    """
    out_dir = Path(out_dir)
    # 1) FINAL = genuine last optimizer step, saved before touching best_state.
    model.load_state_dict(final_state)
    final_sha = atomic_save(out_dir / FINAL_NAME, model, tokenizer,
                            steps=final_step, seed=seed, manifest_hash=manifest_hash)
    # 2) BEST = validation-selected checkpoint, restored only after FINAL is on disk.
    model.load_state_dict(best_state)
    best_sha = atomic_save(out_dir / BEST_NAME, model, tokenizer,
                           steps=best_step, seed=seed, manifest_hash=manifest_hash)
    return {"final_sha256": final_sha, "best_sha256": best_sha,
            "final_steps": final_step, "best_steps": best_step}
