"""来源审计：URL、抓取时间、源哈希、提取器版本、可信等级（M4 任务书第 3 节）。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class Provenance:
    url: str
    fetched_at_utc: str
    source_sha256: str
    extractor_version: str
    trust_level: str          # simulated-fixture | public-read-only
    bytes_read: int
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_provenance(record, extractor_version: str, trust_level: str,
                     notes: str = "") -> Provenance:
    """从 FetchRecord 构造 Provenance（blocked/error 页面不可用于学习流）。"""
    if record.status not in ("fetched", "duplicate"):
        raise ValueError(f"来源不可信（status={record.status}）：{record.url}")
    return Provenance(url=record.url, fetched_at_utc=record.fetched_at_utc,
                      source_sha256=record.sha256, extractor_version=extractor_version,
                      trust_level=trust_level, bytes_read=record.bytes_read, notes=notes)
