"""站点授权注册表：唯一允许写入真实联网权限的控制面（GUI/CLI）。

权限链与知识链分离：本文件只管"谁、何时、授权了哪个站点、限额多少"，
网页内容、模型输出、检索结果都无权写它。policy_hash 把授权与当时的
安全配置绑定（配置变了旧授权要复核）。
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

DEFAULT_LIMITS = {
    "requests_per_idle_cycle": 2,
    "requests_per_day": 20,
    "max_bytes_per_response": 262144,
    "max_total_bytes_per_day": 2097152,
    "request_timeout_s": 8,
    "max_redirects": 0,
    "max_link_depth": 1,
    "cooldown_s": 60,
}


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


class SourceRegistry:
    def __init__(self, path: str | Path, *, limits: dict[str, Any] | None = None) -> None:
        self.path = Path(path)
        self._limits = {**DEFAULT_LIMITS, **(limits or {})}
        self._data: dict[str, Any] = {"sites": [], "exploring_enabled": False,
                                      "limits": dict(self._limits)}
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self._data = {**self._data, **loaded}
            except ValueError:
                pass                             # 损坏文件按空表处理（不静默授权任何站点）

    # ---------------------------------------------------------------- 站点
    def policy_hash(self) -> str:
        payload = json.dumps({"limits": self._data.get("limits"),
                              "sites": sorted(self._data["sites"], key=lambda s: s["host"])},
                             ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def add(self, host: str, *, path_scope: str = "", created_by: str = "user",
            trust_state: str = "public-read-only") -> dict[str, Any]:
        host = host.strip().lower().rstrip("/")
        if not host or "/" in host:
            raise ValueError(f"站点格式必须是纯域名（可含路径范围）：{host!r}")
        for s in self._data["sites"]:
            if s["host"] == host:
                return s
        rec = {"host": host, "path_scope": path_scope, "enabled": True,
               "created_by_user": created_by, "created_at": _now(),
               "trust_state": trust_state, "policy_hash": self.policy_hash(),
               "note": "只能由用户控制面添加；网页内容/模型输出无权写入"}
        self._data["sites"].append(rec)
        self._save()
        return rec

    def remove(self, host: str) -> bool:
        before = len(self._data["sites"])
        self._data["sites"] = [s for s in self._data["sites"] if s["host"] != host.lower()]
        changed = len(self._data["sites"]) != before
        if changed:
            self._save()
        return changed

    def set_state(self, host: str, enabled: bool) -> bool:
        hit = False
        for s in self._data["sites"]:
            if s["host"] == host.lower():
                s["enabled"] = bool(enabled)
                s["updated_at"] = _now()
                hit = True
        if hit:
            self._save()
        return hit

    def list(self) -> list[dict[str, Any]]:
        return [dict(s) for s in self._data["sites"]]

    def enabled_hosts(self) -> list[str]:
        return [s["host"] for s in self._data["sites"] if s.get("enabled")]

    def allowed(self, host: str, path: str = "/") -> bool:
        """精确主机或明确子域匹配；路径按解析后的 path 边界检查。

        不允许 url.startswith(prefix) 这种当安全边界的写法——
        trusted.example.com.evil.com 必须被拒绝。
        """
        host = host.lower()
        for s in self._data["sites"]:
            if not s.get("enabled"):
                continue
            allowed_host = s["host"].lower()
            if host == allowed_host or host.endswith("." + allowed_host):
                scope = s.get("path_scope") or ""
                if not scope:
                    return True
                p = path or "/"
                if p == scope or p.startswith(scope.rstrip("/") + "/"):
                    return True
        return False

    # ---------------------------------------------------------------- 探索开关
    def set_exploring(self, enabled: bool) -> None:
        self._data["exploring_enabled"] = bool(enabled)
        self._data["exploring_updated_at"] = _now()
        self._save()

    @property
    def exploring_enabled(self) -> bool:
        return bool(self._data.get("exploring_enabled"))

    @property
    def limits(self) -> dict[str, Any]:
        return dict(self._data.get("limits") or DEFAULT_LIMITS)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        tmp.replace(self.path)                      # 原子替换，断电不留半文件
