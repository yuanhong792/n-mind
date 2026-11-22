"""Fail-closed provenance and permission registry for public language data."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
import hashlib, json
from pathlib import Path
from typing import Any

@dataclass
class LicenseDecision:
    dataset_id: str
    revision: str
    config: str
    origin_url: str
    license_identifier: str | None = None
    license_url: str | None = None
    attribution_requirements: str = ""
    consent_notes: str = ""
    permitted_use: str = ""
    review_status: str = "REVIEW_REQUIRED"
    limits: dict[str, Any] = field(default_factory=dict)
    sample_hashes: list[str] = field(default_factory=list)
    revoked: bool = False

    def approved(self) -> bool:
        return self.review_status == "APPROVED" and not self.revoked and bool(self.permitted_use)

class DataSourceRegistry:
    def __init__(self, path: str | Path):
        self.path = Path(path); self.records: dict[str, LicenseDecision] = {}
        if self.path.exists():
            for raw in json.loads(self.path.read_text(encoding="utf-8")):
                rec = LicenseDecision(**raw); self.records[rec.dataset_id] = rec
    def register(self, decision: LicenseDecision) -> None:
        self.records[decision.dataset_id] = decision; self.save()
    def get(self, dataset_id: str) -> LicenseDecision:
        if dataset_id not in self.records: raise KeyError(dataset_id)
        return self.records[dataset_id]
    def require_approved(self, dataset_id: str) -> LicenseDecision:
        rec = self.get(dataset_id)
        if not rec.approved(): raise PermissionError(f"source not approved: {dataset_id}")
        return rec
    def revoke(self, dataset_id: str) -> None:
        rec = self.get(dataset_id); rec.revoked = True; rec.review_status = "BLOCKED"; self.save()
    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps([asdict(x) for x in self.records.values()], ensure_ascii=False, indent=2), encoding="utf-8")

def sample_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
