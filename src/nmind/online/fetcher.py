"""只读抓取器：allowlist / 限额 / 超时 / 离线硬断 + 内容指纹去重。

安全口径（M4 任务书第 0/3 节）：
  * offline=True 且使用真实 HTTP 传输时，任何 fetch 都直接 blocked——"无网络可运行"
    是硬约束；注入 transport（模拟网页夹具/测试 mock）时不产生真实网络请求，允许离线走通；
  * URL 必须命中 allowlist 前缀；每次调用最多尝试 1 次，禁止无限重试；
  * 每页最多读 max_bytes 字节，超时 timeout_s；
  * 返回 FetchRecord（URL/UTC/内容 sha256/状态），调用方据此写 provenance；
  * 同一内容指纹第二次出现 → status="duplicate"（任务书去重要求）。
"""

from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable


class OfflineModeError(RuntimeError):
    """offline 模式下尝试真实 HTTP。"""


@dataclass
class FetchConfig:
    offline: bool = True
    allowlist: tuple[str, ...] = ("https://example.com/",)
    timeout_s: float = 10.0
    max_bytes: int = 1_000_000
    user_agent: str = "N-MIND-readonly/0.1 (local research; no-exec; contact: local)"


@dataclass
class FetchRecord:
    url: str
    fetched_at_utc: str
    sha256: str
    status: str                 # fetched | duplicate | blocked | error
    http_status: int | None
    content: str | None
    bytes_read: int
    duplicate_of_url: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"url": self.url, "fetched_at_utc": self.fetched_at_utc,
                "sha256": self.sha256, "status": self.status,
                "http_status": self.http_status, "bytes_read": self.bytes_read,
                "duplicate_of_url": self.duplicate_of_url, "error": self.error}


def _default_transport(url: str, cfg: FetchConfig) -> tuple[int, bytes]:
    """真实只读 HTTP（GET，无 cookie、无 JS、无重试）。仅 allowlist 命中后可达此处。"""
    req = urllib.request.Request(url, headers={"User-Agent": cfg.user_agent})
    with urllib.request.urlopen(req, timeout=cfg.timeout_s) as resp:
        return int(resp.status), resp.read(cfg.max_bytes)


class Fetcher:
    """同一条代码路径服务模拟网页（注入 transport）与真实只读 HTTP（默认关）。"""

    def __init__(self, cfg: FetchConfig, *, transport: Callable[[str, FetchConfig],
                                                                 tuple[int, bytes]] | None = None):
        self.cfg = cfg
        self._transport = transport
        self._seen: dict[str, str] = {}     # sha256 -> 首次出现的 url
        self.records: list[FetchRecord] = []

    def fetch(self, url: str) -> FetchRecord:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if self._transport is None:
            if self.cfg.offline:
                rec = FetchRecord(url, now, "", "blocked", None, None, 0,
                                  error="offline 模式：真实 HTTP 被硬切断（ OfflineModeError）")
                self.records.append(rec)
                return rec
            if not any(url.startswith(a) for a in self.cfg.allowlist):
                rec = FetchRecord(url, now, "", "blocked", None, None, 0,
                                  error="URL 不在 allowlist")
                self.records.append(rec)
                return rec
            transport = _default_transport
        else:
            transport = self._transport     # 注入传输（夹具/mock），无真实网络
        try:
            status, raw = transport(url, self.cfg)
            sha = hashlib.sha256(raw).hexdigest()
            text = raw.decode("utf-8", errors="replace")
            dup = self._seen.get(sha)
            rec = FetchRecord(url, now, sha, "duplicate" if dup else "fetched",
                              int(status), text, len(raw), duplicate_of_url=dup)
            if not dup:
                self._seen[sha] = url
        except Exception as e:                  # 单次尝试，失败如实记录，不重试
            rec = FetchRecord(url, now, "", "error", None, None, 0, error=repr(e))
        self.records.append(rec)
        return rec
