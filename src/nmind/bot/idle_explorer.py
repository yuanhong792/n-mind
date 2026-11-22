"""IdleWebExplorer：无人消息时的限量自主探索（fixture 模式 / 受控真实模式）。

挂在 BotRuntime.run_forever 的空闲分支上（不另造后台 while 循环）。每一轮：
  CHECK_ENABLE → CHECK_NETWORK_PERMISSION → CHECK_USER_PENDING → CHECK_QUOTAS
  → PICK_NEXT_JOB → FETCH_READONLY → EXTRACT → HASH_DEDUP → STORE_AND_INDEX
  → SAME-SITE LINKS(深度内) → COOLDOWN → 返回

抢占语义：用户消息到达最迟在**单个网络请求/解析任务完成边界**被检测，
explorer 立即让路并在返回值里标 preempted；绝不把用户消息和网页内容混记。
离线模式下只走注入的 fixture transport（零真实 socket），真实 HTTP 必须
policy.network_enabled=True 且非 --offline 才可能发生。
"""

from __future__ import annotations

import time
from typing import Any
from urllib.parse import urlparse

from .knowledge_ingest import ingest_extracted
from .knowledge_store import KnowledgeStore
from .source_registry import SourceRegistry

DAY_SECONDS = 86400


class IdleWebExplorer:
    def __init__(self, *, registry: SourceRegistry, knowledge_store: KnowledgeStore,
                 policy, offline: bool = True, transport=None,
                 quota_state: dict[str, Any] | None = None,
                 persist_quota=None, clock=time.time) -> None:
        self.registry = registry
        self.store = knowledge_store
        self.policy = policy
        self.offline = bool(offline)
        self._enabled = False
        self.clock = clock
        # transport：None=按模式自动选择（offline→fixture；否则 SafeTransport）
        self._transport_override = transport
        self._quota: dict[str, Any] = quota_state or {"day_stamp": "", "requests_today": 0,
                                                      "bytes_today": 0}
        self._persist_quota = persist_quota or (lambda q: None)

    # ---------------------------------------------------------------- 开关
    def set_enabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def limits(self) -> dict[str, Any]:
        return self.registry.limits

    # ---------------------------------------------------------------- 传输选择
    def _transport(self):
        if self._transport_override is not None:
            return self._transport_override
        if self.offline or not self.policy.network_enabled:
            from ..online.fixtures import fixture_transport
            return fixture_transport
        from ..online.safe_transport import SafeFetchConfig, make_transport
        lim = self.limits
        cfg = SafeFetchConfig(max_bytes=int(lim["max_bytes_per_response"]),
                              timeout_s=float(lim["request_timeout_s"]),
                              max_redirects=int(lim["max_redirects"]),
                              allowed_sites=tuple(self.registry.enabled_hosts()))
        return make_transport(cfg)

    # ---------------------------------------------------------------- 配额
    def _roll_day(self) -> None:
        today = time.strftime("%Y-%m-%d")
        # 只有"确实跨天"才重置计数；首次（day_stamp 为空）不重置——
        # 否则新实例上先计数再 roll 会把刚记的配额清零（测试抓出的真 bug）
        if self._quota.get("day_stamp") and self._quota["day_stamp"] != today:
            self._quota = {"day_stamp": today, "requests_today": 0, "bytes_today": 0}
            self._persist_quota(self._quota)
        elif not self._quota.get("day_stamp"):
            self._quota["day_stamp"] = today

    def quota_snapshot(self) -> dict[str, Any]:
        self._roll_day()
        lim = self.limits
        return {"requests_today": self._quota["requests_today"],
                "requests_per_day": int(lim["requests_per_day"]),
                "bytes_today": self._quota["bytes_today"],
                "max_total_bytes_per_day": int(lim["max_total_bytes_per_day"]),
                "requests_per_idle_cycle": int(lim["requests_per_idle_cycle"]),
                "enabled": self._enabled,
                "offline": self.offline,
                "network_enabled": bool(self.policy.network_enabled),
                "allowed_hosts": self.registry.enabled_hosts()}

    def _quota_allows(self) -> str | None:
        self._roll_day()
        lim = self.limits
        if self._quota["requests_today"] >= int(lim["requests_per_day"]):
            return "daily_request_quota_exhausted"
        if self._quota["bytes_today"] >= int(lim["max_total_bytes_per_day"]):
            return "daily_bytes_quota_exhausted"
        return None

    def _charge(self, nbytes: int) -> None:
        self._quota["requests_today"] = int(self._quota["requests_today"]) + 1
        self._quota["bytes_today"] = int(self._quota["bytes_today"]) + max(0, nbytes)
        self._persist_quota(self._quota)

    # ---------------------------------------------------------------- 探索主入口
    def run_idle_exploration(self, runtime) -> dict[str, Any]:
        out: dict[str, Any] = {"explored": False, "reason": "", "fetches": [],
                               "preempted": False, "stopped_early": False,
                               "paused": False}
        if not self._enabled:
            out["reason"] = "exploring_disabled"
            return out
        real_net = (not self.offline) and bool(self.policy.network_enabled)
        if not self.offline and not real_net:
            out["reason"] = "network_disabled"
            return out
        if not self.registry.enabled_hosts():
            out["reason"] = "no_allowed_sources"
            return out
        # 用户消息优先：队列里有事就不探索（调度层也保证，这里双检）
        if self._user_pending(runtime):
            out["reason"] = "user_pending"
            out["preempted"] = True
            return out
        quota_block = self._quota_allows()
        if quota_block:
            out["reason"] = quota_block
            return out
        lim = self.limits
        jobs = self.store.next_jobs(limit=max(1, int(lim["requests_per_idle_cycle"])))
        if not jobs:
            out["reason"] = "no_pending_jobs"
            return out
        transport = self._transport()
        cooldown = float(lim["cooldown_s"])
        for job in jobs:
            quota_block = self._quota_allows()
            if quota_block:
                out["reason"] = quota_block
                break
            ctrl = runtime._check_control()
            if ctrl == "stop":
                out["stopped_early"] = True
                break
            if ctrl == "pause":
                out["paused"] = True
                break
            if self._user_pending(runtime):
                out["preempted"] = True
                out["reason"] = out["reason"] or "preempted_by_user"
                break
            rec = self._fetch_job(job, transport)
            out["fetches"].append(rec)
        if not out["paused"] and not out["stopped_early"]:
            out["reason"] = out["reason"] or "cycle_done"
        out["quota"] = self.quota_snapshot()
        out["cooldown_s"] = cooldown
        return out

    def _user_pending(self, runtime) -> bool:
        try:
            if runtime.inbox_len() > 0:
                return True
            if len(runtime.user_queue) > 0:
                return True
            if runtime.store.queue_len() > 0:
                return True
        except Exception:                            # noqa: BLE001 - 检查失败按有事处理
            return True
        return False

    def _fetch_job(self, job: dict[str, Any], transport) -> dict[str, Any]:
        url = job["url"]
        lim = self.limits
        policy_hash = self.registry.policy_hash()
        self.store.record_fetch_event(url, None, "started", policy_hash=policy_hash)
        t0 = time.perf_counter()
        try:
            status, raw = transport(url, None)
        except Exception as exc:                      # noqa: BLE001 - 网络异常完整记账
            elapsed = round((time.perf_counter() - t0) * 1000, 2)
            self.store.record_fetch_event(url, None, "error", str(exc)[:300],
                                          bytes_read=0, elapsed_ms=elapsed,
                                          policy_hash=policy_hash)
            self.store.bump_job(job["job_id"], state="COOLDOWN",
                                reason=f"error:{type(exc).__name__}",
                                next_eligible_at=self.clock() + cooldown_s(lim))
            return {"url": url, "status": "error", "error": str(exc)[:300],
                    "elapsed_ms": elapsed}
        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        self._charge(len(raw))
        if status != 200:
            reason = f"http_{status}"
            self.store.record_fetch_event(url, url, "blocked", reason, bytes_read=len(raw),
                                          elapsed_ms=elapsed, policy_hash=policy_hash)
            self.store.bump_job(job["job_id"], state="COOLDOWN", reason=reason,
                                next_eligible_at=self.clock() + cooldown_s(lim) * 4)
            return {"url": url, "status": "blocked", "http_status": int(status),
                    "reason": reason}
        host = urlparse(url).netloc.lower()
        trust = "fixture" if self.offline or url.startswith("fixture://") or \
            host.endswith(".local") or host == "sim.local" else "public-read-only"
        result = ingest_extracted(self.store, url=url,
                                  html=raw.decode("utf-8", errors="replace"),
                                  fetched_at_utc=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                  trust_state=trust, http_status=int(status),
                                  mime="text/html")
        self.store.record_fetch_event(url, url, result["status"], None,
                                      bytes_read=len(raw), elapsed_ms=elapsed,
                                      policy_hash=policy_hash)
        new_links = 0
        if result.get("status") == "READY" and result.get("links"):
            new_links = self._queue_same_site_links(url, host, result["links"],
                                                    depth=int(job.get("depth") or 0))
        self.store.bump_job(job["job_id"], state="DONE",
                            reason=f"ingested:{result['status']}")
        return {"url": url, "status": result["status"], "doc_id": result.get("doc_id"),
                "new_documents": result.get("new_documents", 0),
                "deduplicated": result.get("deduplicated", 0),
                "indexed_chunks": result.get("indexed_chunks", 0),
                "new_links_queued": new_links, "bytes": len(raw), "elapsed_ms": elapsed}

    def _queue_same_site_links(self, base_url: str, host: str, links: list[str],
                               *, depth: int) -> int:
        """新链接只允许同一授权域、同一授权路径、深度内——不能自动扩大 allowlist。"""
        lim = self.limits
        max_depth = int(lim["max_link_depth"])
        queued = 0
        base_path = urlparse(base_url).path or "/"
        for link in links or []:
            try:
                p = urlparse(link)
            except ValueError:
                continue
            if (p.scheme or "").lower() != "https" and not link.startswith("fixture://"):
                continue
            link_host = (p.netloc or "").lower()
            link_path = p.path or "/"
            if link_host and link_host != host:
                continue
            if not self.registry.allowed(link_host or host, link_path):
                continue
            if depth + 1 > max_depth:
                continue
            # 授权路径范围检查（若站点带 path_scope）
            in_scope = True
            for s in self.registry.list():
                if s["host"].lower() == link_host and s.get("path_scope"):
                    scope = s["path_scope"].rstrip("/")
                    if scope and not link_path.startswith(scope + "/") and \
                            link_path != scope:
                        in_scope = False
            if not in_scope:
                continue
            self.store.upsert_job(link, link_host or host, depth=depth + 1,
                                  state="PENDING")
            queued += 1
        _ = base_path
        return queued

    # ---------------------------------------------------------------- 用户主动读取
    def fetch_for_user(self, url: str) -> dict[str, Any]:
        """GUI「读取网页」：独立于探索开关，但同样必须命中授权白名单。"""
        lim = self.limits
        policy_hash = self.registry.policy_hash()
        t0 = time.perf_counter()
        quota_block = self._quota_allows()
        if quota_block:
            rec = {"url": url, "status": "blocked", "reason": quota_block,
                   "error": "联网请求配额已耗尽，未发起请求"}
            self.store.record_fetch_event(url, None, "blocked", quota_block,
                                          policy_hash=policy_hash)
            return rec
        if self.offline and not url.startswith("fixture://"):
            rec = {"url": url, "status": "blocked", "reason": "offline_mode",
                   "error": "当前是离线模式：真实 HTTP 被硬切断（可读取 fixture:// 资料或缓存知识）"}
            self.store.record_fetch_event(url, None, "blocked", rec["reason"],
                                          policy_hash=policy_hash)
            return rec
        real_net = bool(self.policy.network_enabled)
        if not self.offline and not real_net:
            rec = {"url": url, "status": "blocked", "reason": "network_disabled",
                   "error": "网络未在配置中启用（network_enabled=false）：默认安全，不发送真实请求"}
            self.store.record_fetch_event(url, None, "blocked", rec["reason"],
                                          policy_hash=policy_hash)
            return rec
        if url.startswith("fixture://"):
            try:
                from ..online.fixtures import SIM_WEB
                from .knowledge_ingest import FIXTURE_PAGES
                sim = dict(SIM_WEB)
                # 两套夹具都接受：SIM_WEB（M4 规则夹具）+ FIXTURE_PAGES
                #（V1 中文测试资料），均为仓库内确定性夹具、零真实网络
                for page in FIXTURE_PAGES:
                    sim.setdefault(page["url"], page["html"])
                if url not in sim:
                    raise LookupError("fixture 不存在")
                status, raw = 200, sim[url].encode("utf-8")
            except Exception as exc:                  # noqa: BLE001
                return {"url": url, "status": "error", "error": str(exc)}
        else:
            from ..online.safe_transport import fetch_readonly
            rec = fetch_readonly(url, allowed_sites=tuple(self.registry.enabled_hosts()),
                                 max_bytes=int(lim["max_bytes_per_response"]),
                                 timeout_s=float(lim["request_timeout_s"]),
                                 max_redirects=int(lim["max_redirects"]))
            if rec["status"] != "fetched":
                self.store.record_fetch_event(url, rec.get("final_url"), rec["status"],
                                              rec.get("error_code"),
                                              bytes_read=rec.get("bytes_read", 0),
                                              elapsed_ms=rec.get("elapsed_ms", 0),
                                              policy_hash=policy_hash)
                rec["reason"] = rec.get("error_code") or rec["status"]
                return rec
            status, raw = int(rec["http_status"]), rec["content"].encode("utf-8")
        self._charge(len(raw))
        trust = "fixture" if url.startswith("fixture://") else "public-read-only"
        result = ingest_extracted(self.store, url=url,
                                  html=raw.decode("utf-8", errors="replace"),
                                  fetched_at_utc=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                  trust_state=trust, http_status=status, mime="text/html")
        title = result.get("title") or ""
        excerpt = ""
        if result.get("doc_id"):
            chunks = self.store.chunks_of(result["doc_id"])
            if chunks:
                excerpt = chunks[0]["text"][:300]
        self.store.record_fetch_event(url, url, result["status"], None,
                                      bytes_read=len(raw),
                                      elapsed_ms=round((time.perf_counter() - t0) * 1000, 2),
                                      policy_hash=policy_hash)
        return {"url": url, "status": result["status"], "http_status": status,
                "bytes_read": len(raw), "title": title, "excerpt": excerpt,
                "doc_id": result.get("doc_id"),
                "instruction_detected": result.get("instruction_detected", False),
                "deduplicated": result.get("deduplicated", 0)}


def cooldown_s(limits: dict[str, Any]) -> float:
    return float(limits.get("cooldown_s", 60))
