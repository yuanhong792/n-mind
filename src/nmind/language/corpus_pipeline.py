"""Small, offline-first corpus pipeline. It never fetches data itself."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib, json, re, unicodedata
from pathlib import Path
from .data_sources import DataSourceRegistry, sample_hash

EMAIL = re.compile(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b")
PHONE = re.compile(r"(?<!\d)(?:\+?\d[\d -]{7,}\d)(?!\d)")
SECRET = re.compile(r"(?i)(?:api[_ -]?key|token|password|secret)\s*[:=]\s*\S+")

@dataclass(frozen=True)
class CorpusRecord:
    source_id: str; text: str; split: str; sha256: str; token_count: int; quality: float

def clean_text(text: str) -> tuple[str, dict[str, int]]:
    text = unicodedata.normalize("NFC", text).replace("\x00", " ")
    hits = {"email": len(EMAIL.findall(text)), "phone": len(PHONE.findall(text)), "secret": len(SECRET.findall(text))}
    text = SECRET.sub("[REDACTED]", text); text = EMAIL.sub("[REDACTED]", text); text = PHONE.sub("[REDACTED]", text)
    lines = [re.sub(r"\s+", " ", x).strip() for x in text.splitlines()]
    return "\n".join(x for x in lines if x), hits

def chinese_ratio(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    return sum("\u4e00" <= c <= "\u9fff" for c in chars) / max(1, len(chars))

def build_records(source_id: str, texts: list[str], *, seed: int = 0, token_limit: int = 1_000_000) -> list[CorpusRecord]:
    out = []; seen = set(); total = 0
    for text in texts:
        clean, _ = clean_text(text); digest = sample_hash(clean)
        if not clean or digest in seen: continue
        seen.add(digest); count = len(clean.encode("utf-8"));
        if total + count > token_limit: break
        total += count; out.append(CorpusRecord(source_id, clean, "train", digest, count, chinese_ratio(clean)))
    return out

def manifest(records: list[CorpusRecord]) -> dict:
    body = [asdict(x) for x in records]
    return {"format": "nmind-language-corpus/1", "count": len(body), "sha256": hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True).encode()).hexdigest(), "token_count": sum(x.token_count for x in records), "records": body}

def import_approved_file(registry: DataSourceRegistry, source_id: str, path: str | Path, *, limit=1_000_000) -> list[CorpusRecord]:
    registry.require_approved(source_id)
    return build_records(source_id, Path(path).read_text(encoding="utf-8").split("\n\n"), token_limit=limit)
