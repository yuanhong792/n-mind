"""Large deterministic sealed-set generation and disjointness audit."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from ..language.binding_data import build_family, fingerprint_set


def generate_archive(*, families, n_each=500, seed=20262401, forbidden=None):
    forbidden = set(forbidden or ())
    result, overlap = {}, 0
    for i, family in enumerate(families):
        # Builders are deterministic; distinct seed bands prevent learning-flow reuse.
        rows = build_family(family, "sealed", n=n_each, seed=seed + i)
        result[family] = rows
        overlap += len(fingerprint_set(rows) & forbidden)
    all_fp = sorted(t.fingerprint for rows in result.values() for t in rows)
    fingerprint = hashlib.sha256("\n".join(all_fp).encode()).hexdigest()[:16]
    return result, {"seed": seed, "n_each": n_each, "overlap_count": overlap,
                    "n_total": len(all_fp), "data_fingerprint": fingerprint}


def write_manifest(path: str | Path, audit: dict):
    Path(path).write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
