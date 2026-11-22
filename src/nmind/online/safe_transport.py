"""SafeTransport：受控真实 HTTPS 只读传输（先加固，后授权）。

安全设计（任务书 §4.2，逐条对应实现）：
  * 仅 https:// 且默认端口 443；域名 IDNA 规范化后做**精确主机/明确子域**匹配，
    拒绝 url.startswith(prefix) 当安全边界的写法（trusted.example.com.evil.com 必拦）；
  * 拒绝 URL 内 userinfo、IP 字面量（V1 默认）、非标准端口、localhost/.local/
    内网/回环/链路本地/组播/保留地址与云元数据地址；
  * DNS 解析得到多个地址时任一不安全即拒绝（混合 A/AAAA 攻击）；
  * DNS rebinding 防御：解析出的 IP 直接作为 socket 目标（pinned），SNI 与证书
    校验仍按域名——**检查的地址就是连接的地址**，不给传输库重新解析的机会；
  * socket 直连不经任何环境代理（urllib 的 http_proxy 陷阱不复存在）；
  * 默认 0 跳重定向；配置允许时逐跳重新做全部校验，跨站默认拒绝；
  * 不带 Cookie/Authorization/会话信息；仅 GET；无 JS；单次尝试不无限重试；
  * 流式限制响应字节、限制解压后大小（gzip bomb）、限制超时与 Content-Type；
  * 429/403 及时停止，不绕过反爬/登录墙。

与 fetcher.Fetcher 的兼容：fetch() 可直接作为 transport 注入（(url, cfg) -> (status, bytes)）。
全部安全逻辑可用 FakeTransport 独立验证；真实 HTTP 未获用户授权前保持关闭。
"""

from __future__ import annotations

import ipaddress
import socket
import ssl
import time
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

ALLOWED_CONTENT_TYPES = ("text/html", "application/xhtml+xml", "text/plain",
                         "application/xml", "application/rss+xml",
                         "application/atom+xml", "text/xml")
DEFAULT_UA = "N-MIND-readonly/1.0 (local research; no-exec; no-cookies; contact: local)"
MAX_RESPONSE_HEADERS = 64 * 1024
MAX_REDIRECTS_HARD = 3


class SafeTransportError(Exception):
    """带稳定错误码的拒绝/失败（错误链独立可读，中文说明给 UI）。"""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


CODE_TEXT = {
    "BAD_SCHEME": "仅允许 https://（拒绝 http 与其他协议）",
    "USERINFO": "URL 中不允许用户名/密码（@）",
    "IP_LITERAL": "默认拒绝 IP 字面量 URL（只能用域名）",
    "BAD_PORT": "仅允许默认端口 443",
    "NO_HOST": "URL 缺少主机名",
    "NOT_ALLOWED": "目标不在授权站点列表内",
    "PRIVATE_ADDR": "目标是内网/回环/保留/元数据地址，禁止访问",
    "DNS_FAIL": "域名解析失败",
    "DNS_FORBIDDEN_ADDR": "DNS 解析结果包含非公网地址（含混合地址），拒绝",
    "CERT_FAIL": "TLS 证书校验失败",
    "CONN_FAIL": "连接失败",
    "TIMEOUT": "请求超时",
    "REDIRECT_BLOCKED": "重定向被拒绝（默认 0 跳）",
    "REDIRECT_TARGET_FORBIDDEN": "重定向目标未通过安全校验",
    "MIME_REJECTED": "Content-Type 不在允许列表（拒绝二进制/脚本/安装包）",
    "SIZE_LIMIT": "响应超过字节上限",
    "HTTP_ERROR": "HTTP 状态异常（429/403 即停止，不重试不绕过）",
    "BAD_RESPONSE": "响应解析失败",
}


# ---------------------------------------------------------------- IP 分类
def classify_ip(ip: str) -> str:
    """public | private | loopback | link_local | multicast | reserved | metadata | unspecified"""
    addr = ipaddress.ip_address(ip)
    if addr.version == 4:
        if addr.is_loopback:
            return "loopback"
        if addr.is_link_local:
            return "metadata" if str(addr) == "169.254.169.254" else "link_local"
        if addr.is_multicast or addr.is_reserved or addr.is_unspecified:
            return "multicast" if addr.is_multicast else \
                "unspecified" if addr.is_unspecified else "reserved"
        if addr.is_private:
            return "private"
        return "public"
    # IPv6
    if addr.is_loopback:
        return "loopback"
    if addr.is_link_local:
        return "link_local"
    if addr.is_multicast:
        return "multicast"
    if addr.is_unspecified:
        return "unspecified"
    if addr.is_private:          # IPv4-mapped / ULC fc00::/7 / 0:0:0:0:0:ffff:10/8 等
        return "private"
    if addr.is_reserved:
        return "reserved"
    return "public"


SAFE_IP_CLASSES = frozenset({"public"})


def validate_ip_or_raise(ip: str, *, what: str = "地址") -> str:
    cls = classify_ip(ip)
    if cls not in SAFE_IP_CLASSES:
        raise SafeTransportError("PRIVATE_ADDR",
                                 f"{what} {ip} 类别={cls}，{CODE_TEXT['PRIVATE_ADDR']}")
    return ip


def is_ip_literal(host: str) -> bool:
    try:
        ipaddress.ip_address(host.strip("[]"))
        return True
    except ValueError:
        return False


def idna_normalize(host: str) -> str:
    host = host.strip().rstrip(".").lower()
    if not host:
        raise SafeTransportError("NO_HOST", CODE_TEXT["NO_HOST"])
    try:
        return host.encode("idna").decode("ascii").lower()
    except UnicodeError as exc:
        raise SafeTransportError("NO_HOST", f"域名 IDNA 规范化失败：{host}（{exc}）")


def parse_and_validate_url(url: str) -> dict[str, Any]:
    """URL → {scheme, host(idna), port, path}；协议/用户信息/IP 字面量/端口先拦。"""
    p = urlparse(url)
    if (p.scheme or "").lower() != "https":
        raise SafeTransportError("BAD_SCHEME", f"{url} → {CODE_TEXT['BAD_SCHEME']}")
    if p.username or p.password or "@" in (p.netloc or ""):
        raise SafeTransportError("USERINFO", f"{url} → {CODE_TEXT['USERINFO']}")
    raw_host = p.hostname or ""
    if not raw_host:
        raise SafeTransportError("NO_HOST", f"{url} → {CODE_TEXT['NO_HOST']}")
    if is_ip_literal(raw_host):
        raise SafeTransportError("IP_LITERAL", f"{raw_host} → {CODE_TEXT['IP_LITERAL']}")
    port = p.port or 443
    if port != 443:
        raise SafeTransportError("BAD_PORT", f"端口 {port} → {CODE_TEXT['BAD_PORT']}")
    host = idna_normalize(raw_host)
    host_lower = raw_host.lower()
    if host_lower == "localhost" or host_lower.endswith((".local", ".localhost",
                                                        ".internal")):
        raise SafeTransportError("PRIVATE_ADDR", f"{raw_host} → {CODE_TEXT['PRIVATE_ADDR']}")
    path = p.path or "/"
    if p.query:
        path += "?" + p.query
    return {"scheme": "https", "host": host, "port": port, "path": path,
            "raw_host": raw_host}


# ---------------------------------------------------------------- DNS
def resolve_public_ips(host: str, *, resolver: Callable[[str],
                       list[str]] | None = None) -> list[str]:
    """解析并验证全部地址；任一不安全即整体拒绝（混合 A/AAAA）。"""
    if resolver is None:
        def resolver(h):
            infos = socket.getaddrinfo(h, 443, proto=socket.IPPROTO_TCP)
            return [i[4][0] for i in infos]
    try:
        addrs = list(dict.fromkeys(resolver(host)))
    except OSError as exc:
        raise SafeTransportError("DNS_FAIL", f"{host} 解析失败：{exc}") from exc
    if not addrs:
        raise SafeTransportError("DNS_FAIL", f"{host} 无可用解析结果")
    for a in addrs:
        validate_ip_or_raise(a, what="DNS")
    return addrs


# ---------------------------------------------------------------- 配置
@dataclass
class SafeFetchConfig:
    max_bytes: int = 262144
    timeout_s: float = 8.0
    max_redirects: int = 0
    allowed_sites: tuple[str, ...] = ()       # 精确域名或带路径范围（domain[/path]）
    user_agent: str = DEFAULT_UA
    decompress_limit_bytes: int = 1_048_576   # 解压后上限（gzip bomb 防线）


def site_matches(allowed_sites: tuple[str, ...], host: str, path: str) -> bool:
    """精确主机或明确子域 + 路径边界匹配（与 SourceRegistry.allowed 同规则）。"""
    for entry in allowed_sites:
        entry = entry.strip().lower().rstrip("/")
        if not entry:
            continue
        scope = ""
        if "/" in entry:
            dom, _, scope = entry.partition("/")
        else:
            dom = entry
        if host == dom or host.endswith("." + dom):
            if not scope:
                return True
            p = path or "/"
            if p == "/" + scope or p.startswith("/" + scope.rstrip("/") + "/"):
                return True
    return False


# ---------------------------------------------------------------- 传输
@dataclass
class FetchOutcome:
    status: int
    body: bytes
    final_url: str
    content_type: str = ""
    truncated: bool = False
    redirects: int = 0


def _read_until(headers_end: bytes = b"\r\n\r\n", *, limit: int = MAX_RESPONSE_HEADERS,
                recv: Callable[[int], bytes]) -> bytes:
    buf = bytearray()
    while headers_end not in buf:
        chunk = recv(4096)
        if not chunk:
            break
        buf.extend(chunk)
        if len(buf) > limit:
            raise SafeTransportError("BAD_RESPONSE", "响应头超过上限")
    return bytes(buf)


def https_get_pinned(url: str, cfg: SafeFetchConfig, *,
                     resolver: Callable[[str], list[str]] | None = None,
                     connector: Callable[..., Any] | None = None) -> FetchOutcome:
    """单次 HTTPS GET：解析→校验→pinned 连接→手工 HTTP/1.1→流式限量读取。"""
    target = parse_and_validate_url(url)
    host = target["host"]
    ips = resolve_public_ips(host, resolver=resolver)
    pinned = ips[0]                     # 检查过全部安全；连接用解析地址（防 rebinding）
    if connector is None:
        ctx = ssl.create_default_context()

        def connector(ip, timeout, server_hostname):
            raw = socket.create_connection((ip, 443), timeout=timeout)
            return ctx.wrap_socket(raw, server_hostname=server_hostname)
    try:
        tls = connector(pinned, cfg.timeout_s, host)
    except ssl.SSLError as exc:
        raise SafeTransportError("CERT_FAIL", f"{host} 证书校验失败：{exc}") from exc
    except (TimeoutError, OSError) as exc:
        raise SafeTransportError("CONN_FAIL", f"{host}({pinned}) 连接失败：{exc}") from exc
    try:
        tls.settimeout(cfg.timeout_s)
        req_lines = [
            f"GET {target['path']} HTTP/1.1",
            f"Host: {target['raw_host']}",
            f"User-Agent: {cfg.user_agent}",
            "Accept: text/html, application/xhtml+xml, application/xml;q=0.9, text/plain;q=0.8",
            "Accept-Encoding: gzip",
            "Connection: close",
            "", "",
        ]
        try:
            tls.sendall("\r\n".join(req_lines).encode("ascii"))
        except (TimeoutError, OSError) as exc:
            raise SafeTransportError("CONN_FAIL", f"请求发送失败：{exc}") from exc
        head = _read_until(recv=lambda n: tls.recv(n))
        try:
            head_text = head.decode("iso-8859-1")
            status_line, _, header_block = head_text.partition("\r\n")
            parts = status_line.split(" ", 2)
            status = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
            headers: dict[str, str] = {}
            for line in header_block.split("\r\n"):
                if ":" in line:
                    k, _, v = line.partition(":")
                    headers[k.strip().lower()] = v.strip()
        except (ValueError, IndexError) as exc:
            raise SafeTransportError("BAD_RESPONSE", f"响应头解析失败：{exc}") from exc
        if status in (301, 302, 303, 307, 308):
            loc = headers.get("location", "")
            raise _RedirectSignal(status, loc)
        if status in (429, 403):
            raise SafeTransportError("HTTP_ERROR",
                                     f"HTTP {status}：站点限流/拒绝，按规则停止不重试")
        if status >= 400:
            raise SafeTransportError("HTTP_ERROR", f"HTTP {status}")
        ctype = headers.get("content-type", "").split(";")[0].strip().lower()
        # Missing Content-Type is rejected so binary/archive responses cannot
        # be misclassified by the text extractor.
        if not ctype or ctype not in ALLOWED_CONTENT_TYPES:
            raise SafeTransportError("MIME_REJECTED",
                                     f"Content-Type {ctype!r} 不在允许列表")
        # 流式读取 body（含可能残留的头部后字节）
        body = bytearray(head.split(b"\r\n\r\n", 1)[1] if b"\r\n\r\n" in head else b"")
        while len(body) < cfg.max_bytes:
            try:
                chunk = tls.recv(65536)
            except TimeoutError as exc:
                raise SafeTransportError("TIMEOUT", f"读取正文超时：{exc}") from exc
            if not chunk:
                break
            body.extend(chunk)
        truncated = len(body) > cfg.max_bytes
        body = bytes(body[:cfg.max_bytes])
        if headers.get("content-encoding", "").lower() == "gzip":
            body, truncated_gz = _gunzip_limited(body, cfg.decompress_limit_bytes)
            truncated = truncated or truncated_gz
        return FetchOutcome(status=status, body=body, final_url=url,
                            content_type=ctype, truncated=truncated, redirects=0)
    finally:
        try:
            tls.close()
        except OSError:
            pass


class _RedirectSignal(Exception):
    def __init__(self, status: int, location: str) -> None:
        super().__init__(f"redirect {status} -> {location}")
        self.status = status
        self.location = location


def _gunzip_limited(data: bytes, limit: int) -> tuple[bytes, bool]:
    out = bytearray()
    try:
        dobj = zlib.decompressobj(16 + zlib.MAX_WBITS)
        for i in range(0, len(data), 65536):
            out.extend(dobj.decompress(data[i:i + 65536]))
            if len(out) > limit:
                return bytes(out[:limit]), True
        out.extend(dobj.flush())
    except zlib.error:
        return bytes(out), False            # 解压失败按原样交上层（诚实报错）
    return bytes(out), len(out) >= limit


def safe_fetch(url: str, cfg: SafeFetchConfig, *, _depth: int = 0,
               resolver: Callable[[str], list[str]] | None = None,
               connector: Callable[..., Any] | None = None) -> FetchOutcome:
    """带重定向控制的抓取入口：每跳重新校验（scheme/host/path/DNS/IP）。"""
    try:
        return https_get_pinned(url, cfg, resolver=resolver, connector=connector)
    except _RedirectSignal as sig:
        if _depth >= max(0, min(cfg.max_redirects, MAX_REDIRECTS_HARD)):
            raise SafeTransportError(
                "REDIRECT_BLOCKED",
                f"{url} → HTTP {sig.status} 重定向被拒绝（max_redirects={cfg.max_redirects}）")
        nxt = urljoin_or_reject(url, sig.location)
        # 每一跳完整重新校验：协议/主机/路径白名单/DNS/IP
        parse_and_validate_url(nxt)
        if not site_matches(cfg.allowed_sites, parse_and_validate_url(nxt)["host"],
                            parse_and_validate_url(nxt)["path"]):
            raise SafeTransportError("REDIRECT_TARGET_FORBIDDEN",
                                     f"重定向目标 {nxt} 不在授权站点内（跨站默认拒绝）")
        return safe_fetch(nxt, cfg, _depth=_depth + 1, resolver=resolver,
                          connector=connector)


def urljoin_or_reject(base: str, location: str) -> str:
    from urllib.parse import urljoin
    if not location:
        raise SafeTransportError("REDIRECT_TARGET_FORBIDDEN", "重定向缺少 Location")
    return urljoin(base, location)


def make_transport(cfg: SafeFetchConfig, *, resolver=None,
                   connector=None) -> Callable[[str, Any], tuple[int, bytes]]:
    """生成兼容 fetcher.Fetcher 的 transport（仅当调用方已获授权时使用）。"""

    def transport(url: str, _cfg: Any) -> tuple[int, bytes]:
        out = safe_fetch(url, cfg, resolver=resolver, connector=connector)
        return out.status, out.body

    return transport


# ---------------------------------------------------------------- 统一入口
def fetch_readonly(url: str, *, allowed_sites: tuple[str, ...], max_bytes: int = 262144,
                   timeout_s: float = 8.0, max_redirects: int = 0,
                   resolver=None, connector=None) -> dict[str, Any]:
    """一次完整只读抓取（白名单前置 + 安全传输），返回结构化记录。"""
    t0 = time.monotonic()
    base = {"requested_url": url, "status": "error", "http_status": None,
            "content": None, "bytes_read": 0, "elapsed_ms": 0.0, "error_code": None,
            "error": None, "final_url": url}
    try:
        target = parse_and_validate_url(url)
        if not site_matches(tuple(allowed_sites), target["host"], target["path"]):
            raise SafeTransportError("NOT_ALLOWED",
                                     f"{target['host']}{target['path']} 不在授权站点列表")
        cfg = SafeFetchConfig(max_bytes=max_bytes, timeout_s=timeout_s,
                              max_redirects=max_redirects,
                              allowed_sites=tuple(allowed_sites))
        out = safe_fetch(url, cfg, resolver=resolver, connector=connector)
        base.update({"status": "fetched", "http_status": out.status,
                     "content": out.body.decode("utf-8", errors="replace"),
                     "bytes_read": len(out.body), "final_url": out.final_url,
                     "truncated": out.truncated})
        return base
    except SafeTransportError as exc:
        base.update({"status": "blocked", "error_code": exc.code, "error": exc.message})
        return base
    except Exception as exc:                        # noqa: BLE001 - 未知异常完整记账
        base.update({"status": "error", "error_code": "UNEXPECTED",
                     "error": f"{type(exc).__name__}: {exc}"})
        return base
    finally:
        base["elapsed_ms"] = round((time.monotonic() - t0) * 1000, 2)
