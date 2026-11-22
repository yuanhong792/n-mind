"""SafeTransport 网络安全测试（任务书 §8.1 网络安全组 10+ 项）。

全部使用注入的假 resolver/connector——本文件零真实 socket。
"""

from __future__ import annotations

import pytest

from nmind.bot.web_security_probe import _Fake302Socket, run_probes
from nmind.online.safe_transport import (
    SafeTransportError,
    classify_ip,
    fetch_readonly,
    parse_and_validate_url,
    resolve_public_ips,
    site_matches,
)


def _c302(location):
    def connector(ip, timeout, server_hostname):
        return _Fake302Socket(location)
    return connector


def _c200(body=b"hello"):
    def connector(ip, timeout, server_hostname):
        return _Fake200Socket(body)
    return connector


def _c_redirect_then_200(location, body=b"final page"):
    """第一跳返回 302，第二跳返回 200（模拟同站跳转链）。"""
    def connector(ip, timeout, server_hostname):
        if getattr(connector, "_used", False):
            return _Fake200Socket(body)
        connector._used = True
        return _Fake302Socket(location)
    return connector


class _Fake200Socket:
    """真实 TCP recv(n) 最多返回 n 字节：夹具必须遵守，否则测不到真实分块行为。"""

    def __init__(self, body: bytes = b"hello") -> None:
        self.body = body
        self._sent = False

    def settimeout(self, _t):
        pass

    def sendall(self, _b):
        pass

    def recv(self, n):
        if self._sent:
            return b""
        self._sent = True
        head = (f"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n"
                f"Content-Length: {len(self.body)}\r\n\r\n").encode() + self.body
        return head[:n]

    def close(self):
        pass


# ---------------------------------------------------------------- URL 校验
def test_reject_http_scheme():
    with pytest.raises(SafeTransportError) as e:
        parse_and_validate_url("http://example.com/")
    assert e.value.code == "BAD_SCHEME"


def test_reject_userinfo_url():
    with pytest.raises(SafeTransportError) as e:
        parse_and_validate_url("https://user:pass@example.com/")
    assert e.value.code == "USERINFO"


def test_reject_ip_literal():
    with pytest.raises(SafeTransportError) as e:
        parse_and_validate_url("https://10.0.0.1/x")
    assert e.value.code == "IP_LITERAL"


def test_reject_nonstandard_port():
    with pytest.raises(SafeTransportError) as e:
        parse_and_validate_url("https://example.com:8443/")
    assert e.value.code == "BAD_PORT"


@pytest.mark.parametrize("url", ["https://localhost/", "https://a.local/",
                                 "https://x.internal/"])
def test_reject_local_hostname(url):
    with pytest.raises(SafeTransportError) as e:
        parse_and_validate_url(url)
    assert e.value.code == "PRIVATE_ADDR"


def test_dns_rebinding_mixed_addresses_rejected():
    """DNS 混合 A/AAAA：公网+私网并存时整体拒绝（任一不安全即拒绝）。"""
    with pytest.raises(SafeTransportError) as e:
        resolve_public_ips("example.com",
                           resolver=lambda h: ["93.184.216.34", "192.168.1.10"])
    assert e.value.code == "PRIVATE_ADDR"


def test_dns_loopback_rejected():
    with pytest.raises(SafeTransportError) as e:
        resolve_public_ips("example.com", resolver=lambda h: ["127.0.0.1"])
    assert e.value.code == "PRIVATE_ADDR"


def test_metadata_addr_classified():
    assert classify_ip("169.254.169.254") == "metadata"
    assert classify_ip("::1") == "loopback"
    assert classify_ip("fd00::1") == "private"


def test_subdomain_spoof_rejected():
    assert not site_matches(("trusted.example.com",), "trusted.example.com.evil.com", "/")


def test_path_scope_boundary():
    assert site_matches(("example.com/docs",), "example.com", "/docs/a")
    assert not site_matches(("example.com/docs",), "example.com", "/docsxx")
    assert not site_matches(("example.com/docs",), "example.com", "/other")


def test_whitelist_enforced_before_connection():
    """白名单外站点必须在连接前被拒（deny connector 一旦被调即断言失败）。"""

    def deny(ip, timeout, server_hostname):
        raise AssertionError("不应进入连接阶段")

    rec = fetch_readonly("https://unknown.example.net/", allowed_sites=("example.com",),
                         resolver=lambda h: ["93.184.216.34"], connector=deny)
    assert rec["status"] == "blocked" and rec["error_code"] == "NOT_ALLOWED"


def test_redirect_blocked_by_default():
    rec = fetch_readonly("https://example.com/x", allowed_sites=("example.com",),
                         max_redirects=0, resolver=lambda h: ["93.184.216.34"],
                         connector=_c302("https://example.com/y"))
    assert rec["status"] == "blocked"
    assert rec["error_code"] == "REDIRECT_BLOCKED"


def test_redirect_cross_site_rejected():
    rec = fetch_readonly("https://example.com/jump", allowed_sites=("example.com",),
                         max_redirects=1, resolver=lambda h: ["93.184.216.34"],
                         connector=_c302("https://evil.example.net/"))
    assert rec["status"] == "blocked"
    assert rec["error_code"] in ("REDIRECT_TARGET_FORBIDDEN", "NOT_ALLOWED")


def test_redirect_same_site_allowed_within_limit():
    rec = fetch_readonly("https://example.com/a", allowed_sites=("example.com",),
                         max_redirects=1, resolver=lambda h: ["93.184.216.34"],
                         connector=_c_redirect_then_200("https://www.example.com/b"))
    assert rec["status"] == "fetched" and rec["bytes_read"] == len(b"final page")


def test_mime_rejection_blocks_binary():
    class _Bin(_Fake200Socket):
        def recv(self, _n):
            if self._sent:
                return b""
            self._sent = True
            return (b"HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\n"
                    b"\r\nBIN")

    def conn(ip, timeout, server_hostname):
        return _Bin()

    rec = fetch_readonly("https://example.com/setup.exe", allowed_sites=("example.com",),
                         resolver=lambda h: ["93.184.216.34"], connector=conn)
    assert rec["status"] == "blocked" and rec["error_code"] == "MIME_REJECTED"


def test_http_429_stops_not_retries():
    class _429:
        def settimeout(self, _t):
            pass

        def sendall(self, _b):
            pass

        def recv(self, _n):
            return b"HTTP/1.1 429 Too Many Requests\r\nRetry-After: 3600\r\n\r\n"

        def close(self):
            pass

    calls = {"n": 0}

    def conn(ip, timeout, server_hostname):
        calls["n"] += 1
        return _429()

    rec = fetch_readonly("https://example.com/", allowed_sites=("example.com",),
                         resolver=lambda h: ["93.184.216.34"], connector=conn)
    assert rec["status"] == "blocked" and rec["error_code"] == "HTTP_ERROR"
    assert calls["n"] == 1                       # 单次尝试，不重试


def test_size_limit_truncates():
    big = b"x" * (70000)
    rec = fetch_readonly("https://example.com/big", allowed_sites=("example.com",),
                         max_bytes=1000, resolver=lambda h: ["93.184.216.34"],
                         connector=_c200(big))
    assert rec["status"] == "fetched"
    assert rec["bytes_read"] <= 1000


def test_idna_normalization():
    t = parse_and_validate_url("https://例子.测试/a")
    assert t["host"].startswith("xn--")


# ---------------------------------------------------------------- 内置探针
def test_builtin_probe_suite_all_pass():
    out = run_probes()
    assert out["verdict"] == "PASS"
    assert out["real_http_sent"] is False
    assert out["passed"] == out["total"] and out["total"] >= 15
