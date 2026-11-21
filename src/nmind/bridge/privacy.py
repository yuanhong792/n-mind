"""最小必要 payload 与脱敏。

规则：用户文本、文件路径、记忆内容一律是**不可信数据**，不能当指令执行；
出网前必须先把秘密、token、整份本地文件、私人数据库内容挡下来。

做法是白名单 + 结构化字段，而不是"黑名单替换敏感词"：
黑名单永远漏，白名单只要字段没声明就发不出去。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

# 明显是凭据的形态：一律视为 secret（即便用户说"这是公开信息"也不放——见 policy 的硬规则）
SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9]{10,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9._\-]{8,}"),
    re.compile(r"api[_\-]?key\s*[:=]", re.I),
    re.compile(r"(AKIA|ASIA)[0-9A-Z]{16}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"[A-Za-z0-9+/]{48,}={0,2}"),           # 长 base64 块（疑似 token/权重）
)
PATH_PATTERNS = (
    re.compile(r"(?:[A-Za-z]:\\|\\\\)[^\s\"']{3,}"),    # Windows 路径
    re.compile(r"(?<!\w)/(?:home|Users|mnt|proc|etc|var|root)/[^\s\"']{2,}"),
    re.compile(r"(?:file|sftp|ftp)://\S+"),
)
PERSONAL_PATTERNS = (
    re.compile(r"\b1[3-9]\d{9}\b"),                     # 中国大陆手机号
    re.compile(r"\b\d{17}[\dXx]\b"),                    # 身份证
    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b"),      # 邮箱
    re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),         # IPv4
    re.compile(r"\b\d{6,20}\b"),                        # 长数字（疑似卡号/账号）
)
# 会被误杀的常见正常数字：坐标/时间戳之外的短数字不在长数字规则内（{6,20} 已足够窄）
SAFE_OVERRIDE = (re.compile(r"^(https?|git)://[A-Za-z0-9._\-/]+$"),)


def level_of(text: str) -> str:
    """把一段文本归到 secret > personal > local_path > public 的最高危档。"""
    if not text:
        return "public"
    for pat in SECRET_PATTERNS:
        if pat.search(text):
            return "secret"
    for pat in PERSONAL_PATTERNS:
        if pat.search(text):
            return "personal"
    for pat in PATH_PATTERNS:
        if pat.search(text):
            return "local_path"
    return "public"


def redact(text: str, *, replace_with: str = "[已脱敏]") -> tuple[str, list[str]]:
    """逐类替换，返回脱敏后的文本与命中类别（用于审计，不含原文）。"""
    hits: list[str] = []
    out = text or ""
    for name, pats in (("secret", SECRET_PATTERNS), ("personal", PERSONAL_PATTERNS),
                       ("local_path", PATH_PATTERNS)):
        for pat in pats:
            if pat.search(out):
                hits.append(name)
                out = pat.sub(replace_with, out)
    return out, sorted(set(hits))


@dataclass
class Payload:
    """允许出网的最小结构。字段是封闭集合，extra 一律拒收。"""

    category: str
    question: str = ""
    context_summary: str = ""
    numeric_summary: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    blocked: list[str] = field(default_factory=list)
    original_chars: int = 0
    payload_chars: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {"category": self.category, "question": self.question,
                "context_summary": self.context_summary, "numeric_summary": self.numeric_summary,
                "meta": self.meta}

    @property
    def is_clean(self) -> bool:
        return not self.blocked


def build_payload(*, category: str, question: str = "", context: Iterable[str] = (),
                  numeric: dict[str, Any] | None = None, meta: dict[str, Any] | None = None,
                  max_chars: int = 4000, privacy_levels: dict[str, int] | None = None) -> Payload:
    """组装载荷：分类不是 public_web_query 的一律整块拦下（不部分发送）。

    privacy_levels 来自记忆检索结果的统计（不含正文）：只要含 private/secret，
    就把该条上下文丢弃而不是脱敏后发送——脱敏后的残句仍可能泄露结构。
    """
    levels = privacy_levels or {}
    blocked: list[str] = []
    if category != "public_web_query":
        blocked.append("category_not_allowed")
    if levels.get("private") or levels.get("secret"):
        blocked.append("private_memory_present")
    if "local_path" in question or any(":\\" in c or c.startswith("/") for c in context):
        blocked.append("local_path_present")

    if blocked:
        p = Payload(category=category, blocked=blocked,
                    numeric_summary={"reason": "载荷被拦下，未包含任何原文"},
                    original_chars=len(question) + sum(len(c) for c in context))
        p.payload_chars = 0
        return p

    q, hq = redact(question)
    ctx_lines = []
    hit_categories: list[str] = list(hq)
    for c in context:
        rc, hits = redact(c)
        hit_categories += hits
        if "secret" not in hits:                 # secret 直接丢行，连占位符都不给
            ctx_lines.append(rc)
    summary = " / ".join(ctx_lines)
    if len(summary) > max_chars // 2:
        summary = summary[: max_chars // 2] + "…"
    if len(q) > max_chars:
        q = q[:max_chars] + "…"

    payload = Payload(
        category=category,
        question=q,
        context_summary=summary,
        numeric_summary=dict(numeric or {}),
        meta={"redacted_categories": sorted(set(hit_categories)), "context_items": len(ctx_lines)},
    )
    payload.original_chars = len(question) + sum(len(c) for c in context)
    payload.payload_chars = len(payload.as_dict().__repr__())
    if payload.original_chars > payload.payload_chars + 1_000_000:
        payload.meta["compression_note"] = "已丢弃冗余上下文"
    return payload
