"""Strict, metadata-gated result writing for phase 1 read-only evaluations."""
from __future__ import annotations
import json
from pathlib import Path

REQUIRED_METADATA = {
    "torch_version", "torch_build", "python_version", "git_revision",
    "data_fingerprint", "config_hash", "seed", "sample_count", "effective_sample_count",
}

def validate_metadata(metadata: dict) -> None:
    missing = sorted(REQUIRED_METADATA - set(metadata))
    if missing:
        raise ValueError("missing required metadata: " + ", ".join(missing))
    if int(metadata["sample_count"]) < 0 or int(metadata["effective_sample_count"]) < 0:
        raise ValueError("sample counts must be non-negative")

def write_result(path: str | Path, *, metadata: dict, result: dict) -> None:
    validate_metadata(metadata)
    payload = {"metadata": metadata, "result": result}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
