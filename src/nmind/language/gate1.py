"""Reproducible, offline Gate-1 dataset and leakage audit."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from .corpus_pipeline import build_records, split_by_document

MANUAL_SOURCE = "user-authored-zh-gate1"

def build_gate1(text_path: str | Path) -> dict:
    paragraphs = Path(text_path).read_text(encoding="utf-8").split("\n\n")
    records = build_records(MANUAL_SOURCE, paragraphs, token_limit=20_000)
    # Make source/document identity explicit so sealed material cannot be
    # accidentally constructed from the same paragraph as training material.
    records = [type(r)(f"{MANUAL_SOURCE}:{i:03d}", r.text, r.split, r.sha256, r.token_count, r.quality) for i, r in enumerate(records)]
    sets = split_by_document(records)
    hashes = {name: {r.sha256 for r in rows} for name, rows in sets.items()}
    overlap = (hashes["train"] & hashes["dev"]) | (hashes["train"] & hashes["sealed"]) | (hashes["dev"] & hashes["sealed"])
    manifest = {"format": "language-v2-gate1-manifest/1", "source": MANUAL_SOURCE,
                "license": "user-authored-explicit-research-consent",
                "sets": {k: [r.__dict__ for r in v] for k, v in sets.items()},
                "overlap_hashes": sorted(overlap), "leakage_free": not overlap,
                "tokenizer": "TokenizerV2", "network_bytes": 0, "optimizer_steps": 0}
    manifest["manifest_sha256"] = hashlib.sha256(json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    return manifest
