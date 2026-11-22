"""Run the explicitly approved Gate-1 experiment.

This script is intentionally narrow: one exact host, robots.txt plus three
fixed pages, no redirects across the approved path set, no training, and an
append-only JSONL audit. It does not enable the idle explorer.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nmind.bot.knowledge_ingest import ingest_extracted  # noqa: E402
from nmind.bot.knowledge_store import KnowledgeStore  # noqa: E402
from nmind.online.safe_transport import fetch_readonly  # noqa: E402

HOST = "docs.python.org"
ROBOTS = "https://docs.python.org/robots.txt"
PAGES = (
    "https://docs.python.org/zh-cn/3.11/tutorial/index.html",
    "https://docs.python.org/zh-cn/3.11/tutorial/introduction.html",
    "https://docs.python.org/zh-cn/3.11/tutorial/controlflow.html",
)
ALLOWED = (
    "docs.python.org/robots.txt",
    "docs.python.org/zh-cn/3.11/tutorial/",
    "docs.python.org/zh-cn/3.11/library/",
)
MAX_REQUESTS = 40
MAX_PAGE_BYTES = 512 * 1024
MAX_TOTAL_BYTES = 12 * 1024 * 1024
MIN_INTERVAL = 5.0


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    out = ROOT / "results" / "online_v1_gate1"
    out.mkdir(parents=True, exist_ok=True)
    audit_path = out / "ONLINE_V1_REQUEST_AUDIT.jsonl"
    store = KnowledgeStore(out / "knowledge.db")
    audit: list[dict] = []
    total_bytes = 0
    started: list[float] = []

    def request(url: str, kind: str) -> dict:
        nonlocal total_bytes
        if len(audit) >= MAX_REQUESTS:
            raise RuntimeError("request_budget_exhausted")
        if started and time.monotonic() - started[-1] < MIN_INTERVAL:
            time.sleep(MIN_INTERVAL - (time.monotonic() - started[-1]))
        started.append(time.monotonic())
        rec = fetch_readonly(url, allowed_sites=ALLOWED, max_bytes=MAX_PAGE_BYTES,
                             timeout_s=10.0, max_redirects=0)
        raw = (rec.get("content") or "").encode("utf-8")
        total_bytes += len(raw)
        event = {"request_no": len(audit) + 1, "kind": kind, "url": url,
                 "host": HOST, "method": "GET", "status": rec.get("status"),
                 "http_status": rec.get("http_status"), "bytes": len(raw),
                 "sha256": hashlib.sha256(raw).hexdigest() if raw else "",
                 "error_code": rec.get("error_code"), "error": rec.get("error"),
                 "elapsed_ms": rec.get("elapsed_ms"), "network_authorized": True,
                 "redirects": 0, "robots_checked": kind != "robots"}
        audit.append(event)
        if total_bytes > MAX_TOTAL_BYTES:
            raise RuntimeError("total_response_budget_exhausted")
        return rec

    robots = request(ROBOTS, "robots")
    if robots.get("status") != "fetched":
        raise RuntimeError(f"robots_check_failed:{robots.get('error_code')}")
    robots_hash = audit[-1]["sha256"]
    for url in PAGES:
        rec = request(url, "page")
        if rec.get("status") != "fetched":
            continue
        result = ingest_extracted(store, url=url, html=rec["content"],
                                  fetched_at_utc=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                  trust_state="public-read-only", http_status=rec["http_status"],
                                  mime="text/html")
        audit[-1].update({"title": result.get("title", ""),
                          "content_sha256": result.get("content_sha256", audit[-1]["sha256"]),
                          "extractor_version": result.get("extractor_version", "unknown"),
                          "knowledge_status": result.get("status"),
                          "doc_id": result.get("doc_id"), "robots_hash": robots_hash,
                          "license_url": "https://docs.python.org/3.11/license.html"})

    with audit_path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in audit:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    _write_json(out / "ONLINE_V1_SOURCE_REGISTRY.json", {
        "status": "APPROVED_FOR_GATE_1", "host": HOST,
        "paths": list(ALLOWED), "robots_checked": True,
        "real_http_requests": len(audit), "pages_attempted": len(PAGES),
        "optimizer_steps": 0, "external_llm_calls": 0})
    _write_json(out / "ONLINE_V1_KNOWLEDGE_METRICS.json", {
        "status": "COMPLETE", "robots_sha256": robots_hash,
        "requests": len(audit), "pages_attempted": len(PAGES),
        "pages_ready": sum(1 for x in audit if x.get("knowledge_status") == "READY"),
        "bytes_total": total_bytes, "optimizer_steps": 0,
        "unauthorized_requests": 0, "stable_checkpoint_modified": False})
    store.close()
    print(json.dumps({"status": "COMPLETE", "requests": len(audit),
                      "pages": len(PAGES), "bytes": total_bytes}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
