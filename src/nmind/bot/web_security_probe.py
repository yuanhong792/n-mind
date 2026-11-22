"""内置网络安全探针（bot web test-security）：纯本地校验逻辑，零真实网络请求。

覆盖任务书 §8.1 网络安全组的可离线验证项：协议/用户信息/IP字面量/端口/私网/
DNS混合地址/子域欺骗/路径边界/重定向跨站。所有 resolver/connector 全部注入
假实现——探针无论通过与否都不会产生任何 socket。
真实公网端到端另有单独授权流程，本探针不代跑。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..online.safe_transport import (
    SafeTransportError,
    classify_ip,
    fetch_readonly,
    parse_and_validate_url,
    resolve_public_ips,
    site_matches,
)


class _Fake302Socket:
    """模拟 TLS socket：返回 302 + Location 头（零网络）。"""

    def __init__(self, location: str) -> None:
        self.location = location

    def settimeout(self, _t: float) -> None:
        pass

    def sendall(self, _b: bytes) -> None:
        pass

    def recv(self, _n: int) -> bytes:
        if getattr(self, "_sent", False):
            return b""
        self._sent = True
        head = (f"HTTP/1.1 302 Found\r\nLocation: {self.location}\r\n"
                f"Content-Length: 0\r\nConnection: close\r\n\r\n")
        return head.encode("iso-8859-1")

    def close(self) -> None:
        pass


def _connector_302(location: str):
    def connector(ip, timeout, server_hostname):
        return _Fake302Socket(location)
    return connector


def _probe(name: str, ok: bool, detail: str) -> dict[str, Any]:
    return {"probe": name, "pass": bool(ok), "detail": detail}


def _expect_error(fn: Callable[[], Any], code: str | None = None) -> tuple[bool, str]:
    try:
        fn()
    except SafeTransportError as exc:
        if code is None or exc.code == code:
            return True, f"按预期拒绝（{exc.code}）"
        return False, f"错误码不符：期望 {code}，实际 {exc.code}"
    except Exception as exc:                          # noqa: BLE001
        return False, f"抛出了非 SafeTransportError 异常：{type(exc).__name__}: {exc}"
    return False, "没有抛出异常（校验被绕过）"


def run_probes() -> dict[str, Any]:
    results: list[dict[str, Any]] = []

    # 1) http 明文拒绝
    ok, detail = _expect_error(lambda: parse_and_validate_url("http://example.com/"),
                               "BAD_SCHEME")
    results.append(_probe("reject_http_scheme", ok, detail))

    # 2) URL 内用户名/密码拒绝
    ok, detail = _expect_error(
        lambda: parse_and_validate_url("https://user:pass@example.com/"), "USERINFO")
    results.append(_probe("reject_userinfo_url", ok, detail))

    # 3) IP 字面量拒绝（V1 默认）
    ok, detail = _expect_error(lambda: parse_and_validate_url("https://93.184.216.34/"),
                               "IP_LITERAL")
    results.append(_probe("reject_ip_literal", ok, detail))

    # 4) 非标准端口拒绝
    ok, detail = _expect_error(lambda: parse_and_validate_url("https://example.com:8080/"),
                               "BAD_PORT")
    results.append(_probe("reject_nonstandard_port", ok, detail))

    # 5) localhost/.local/.internal 域名拒绝
    for u in ("https://localhost/", "https://host.local/", "https://box.internal/"):
        ok, detail = _expect_error(lambda u=u: parse_and_validate_url(u), "PRIVATE_ADDR")
        results.append(_probe(f"reject_local_hostname[{u}]", ok, detail))

    # 6) DNS 解析出回环/私网地址拒绝
    ok, detail = _expect_error(
        lambda: resolve_public_ips("example.com", resolver=lambda h: ["127.0.0.1"]),
        "PRIVATE_ADDR")
    results.append(_probe("reject_dns_loopback", ok, detail))

    # 7) DNS 混合 A/AAAA（公网+私网）→ 任一不安全整体拒绝
    ok, detail = _expect_error(
        lambda: resolve_public_ips("example.com",
                                   resolver=lambda h: ["93.184.216.34", "192.168.1.10"]),
        "PRIVATE_ADDR")
    results.append(_probe("reject_dns_mixed_addresses", ok, detail))

    # 8) 云元数据地址分类
    cls = classify_ip("169.254.169.254")
    results.append(_probe("classify_metadata_addr", cls == "metadata",
                          f"169.254.169.254 → {cls}"))

    # 9) 子域欺骗：trusted.example.com.evil.com 不得匹配 trusted.example.com
    spoof_ok = not site_matches(("trusted.example.com",), "trusted.example.com.evil.com", "/")
    results.append(_probe("reject_subdomain_spoof", spoof_ok,
                          "trusted.example.com.evil.com 未通过精确子域匹配" if spoof_ok
                          else "前缀匹配漏洞：子域欺骗被放行"))

    # 10) 同域合法匹配（明确子域规则）
    ok = site_matches(("example.com",), "www.example.com", "/a/b")
    results.append(_probe("allow_exact_and_subdomain", ok,
                          "www.example.com 命中 example.com 授权"))

    # 11) 路径边界：/docs 范围内允许，/docsxx 与 /other 拒绝
    ok = (site_matches(("example.com/docs",), "example.com", "/docs/a")
          and not site_matches(("example.com/docs",), "example.com", "/docsxx")
          and not site_matches(("example.com/docs",), "example.com", "/other"))
    results.append(_probe("path_scope_boundary", ok,
                          "授权 /docs 时 /docs/a 放行、/docsxx 与 /other 拒绝"))

    # 12) 白名单外站点整体拒绝（fetch_readonly 白名单前置；connector 不会被调到）
    def _deny_all(ip, timeout, server_hostname):
        raise AssertionError("白名单外站点不允许进入连接阶段")
    rec = fetch_readonly("https://not-allowed.example.net/",
                         allowed_sites=("example.com",), resolver=lambda h: ["93.184.216.34"],
                         connector=_deny_all)
    ok = rec["status"] == "blocked" and rec["error_code"] == "NOT_ALLOWED"
    results.append(_probe("reject_non_allowed_site", ok,
                          f"status={rec['status']} code={rec.get('error_code')}"))

    # 13) 重定向 0 跳默认拒绝（302 用假 socket 模拟）
    rec = fetch_readonly("https://example.com/x", allowed_sites=("example.com",),
                         max_redirects=0, resolver=lambda h: ["93.184.216.34"],
                         connector=_connector_302("https://example.com/x2"))
    ok = rec["status"] == "blocked" and rec["error_code"] == "REDIRECT_BLOCKED"
    results.append(_probe("redirect_blocked_by_default", ok,
                          f"status={rec['status']} code={rec.get('error_code')}"))

    # 14) 重定向到未授权域拒绝（跨站默认拒绝）
    rec = fetch_readonly("https://example.com/jump", allowed_sites=("example.com",),
                         max_redirects=1, resolver=lambda h: ["93.184.216.34"],
                         connector=_connector_302("https://evil.example.net/steal"))
    ok = rec["status"] == "blocked" and rec.get("error_code") in (
        "REDIRECT_TARGET_FORBIDDEN", "NOT_ALLOWED")
    results.append(_probe("redirect_cross_site_rejected", ok,
                          f"status={rec['status']} code={rec.get('error_code')}"))

    # 15) IDNA 规范化：中文域名统一为 punycode
    t = parse_and_validate_url("https://例子.测试/a")
    ok = t["host"].startswith("xn--")
    results.append(_probe("idna_normalize", ok, f"例子.测试 → {t['host']}"))

    passed = sum(1 for r in results if r["pass"])
    verdict = "PASS" if passed == len(results) else "FAIL"
    return {"gate": "NETWORK_SECURITY_PROBE", "verdict": verdict,
            "passed": passed, "total": len(results), "real_http_sent": False,
            "probes": results,
            "note": "全部探针使用注入的假 resolver/connector，零真实 socket；"
                    "真实公网访问需用户单独授权（REAL_HTTP_NOT_TESTED 状态另行记录）"}
