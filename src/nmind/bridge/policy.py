"""云端决策策略与审计日志。

策略是**代码里的硬门**，不是提示词：模型（或用户输入里的自然语言）不能通过
"请忽略之前的规则"绕过它。能被绕过的约束不算约束。

顺序（任何一条不过就拒绝，且不发起任何网络请求）：
  1. 有效模式 == off（含 --offline 强制）         → 拒绝
  2. payload 带 blocked 标记（秘密/路径/私人记忆）  → 拒绝
  3. category 不在白名单                          → 拒绝
  4. 剩余预算 < 本次预估成本                        → 拒绝
  5. require_explicit_approval 且未获批准           → 拒绝（ask 模式停在这里）
  6. 以上全过 → 允许
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .privacy import Payload, redact


@dataclass
class PolicyDecision:
    allowed: bool
    reason: str
    category: str
    requires_approval: bool = False
    payload_chars: int = 0
    original_chars: int = 0
    cost_estimate: float = 0.0
    budget_left_before: float = 0.0
    blocked: list[str] = field(default_factory=list)
    mode: str = "off"


class CloudPolicy:
    def __init__(self, cfg: Any, *, cost_per_call: float | None = None, clock=time.monotonic) -> None:
        self.mode = cfg.effective_cloud_mode()
        self.allowed_categories = list(cfg.cloud.allowed_categories)
        self.require_approval = bool(cfg.cloud.require_explicit_approval)
        self.budget_total = float(cfg.cloud.cost_budget_units_total)
        self.spent = 0.0
        self.cost_per_call = float(cost_per_call if cost_per_call is not None
                                   else cfg.cloud.cost_estimate_per_call)
        self.requested = 0
        self.allowed_count = 0
        self.denied: dict[str, int] = {}
        self._clock = clock

    @property
    def budget_left(self) -> float:
        return max(0.0, self.budget_total - self.spent)

    def evaluate(self, payload: Payload, *, explicit_approval: bool = False,
                 cost_estimate: float | None = None) -> PolicyDecision:
        cost = float(self.cost_per_call if cost_estimate is None else cost_estimate)
        d = PolicyDecision(False, "", payload.category, payload_chars=payload.payload_chars,
                           original_chars=payload.original_chars, cost_estimate=cost,
                           budget_left_before=self.budget_left, mode=self.mode)
        if self.mode == "off":
            d.reason = "cloud_mode_off"
        elif payload.blocked:
            d.reason = "payload_blocked"
            d.blocked = list(payload.blocked)
        elif payload.category not in self.allowed_categories:
            d.reason = "category_not_allowed"
        elif cost > self.budget_left:
            d.reason = "budget_exhausted"
        elif self.require_approval and not explicit_approval:
            d.reason = "needs_approval"
            d.requires_approval = True
        else:
            d.allowed = True
            d.reason = "allowed"
        self.requested += 1
        if d.allowed:
            self.allowed_count += 1
        else:
            self.denied[d.reason] = self.denied.get(d.reason, 0) + 1
        return d

    def commit(self, decision: PolicyDecision) -> None:
        """调用真的发生了才记账；被拒绝的请求不占预算。"""
        if decision.allowed:
            self.spent += float(decision.cost_estimate)

    def status(self) -> dict[str, Any]:
        return {"mode": self.mode, "requested": self.requested, "allowed": self.allowed_count,
                "denied": dict(self.denied), "budget_total": self.budget_total,
                "spent": round(self.spent, 6), "budget_left": round(self.budget_left, 6),
                "allowed_categories": self.allowed_categories,
                "require_explicit_approval": self.require_approval}


class Auditor:
    """本地审计日志：记"请求了哪类、多长、批没批、花了多少"，不记正文与密钥。"""

    def __init__(self, path: str | Path, *, mask: bool = True) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.mask = mask

    def record(self, *, session_id: str, decision: PolicyDecision,
               result_summary: dict[str, Any] | None = None,
               question_preview: str = "") -> None:
        row: dict[str, Any] = {"ts": round(self._now(), 3), "session": session_id,
                               **asdict(decision)}
        if result_summary:
            row["result"] = result_summary
        if question_preview:
            text = question_preview[:80]
            row["question_preview"] = redact(text)[0] if self.mask else text[:80]
        # 双保险：任何键里出现疑似密钥值就整体丢弃该键（宁可少记，不可泄露）
        row = {k: v for k, v in row.items() if not _looks_secret(repr(v))}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    def _now(self) -> float:
        return time.time()

    def tail(self, n: int = 10) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        lines = [ln for ln in self.path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        return [json.loads(ln) for ln in lines[-n:]]


def _looks_secret(blob: str) -> bool:
    lowered = blob.lower()
    return any(tok in lowered for tok in ("sk-", "bearer ", "api_key", "apikey", "private key"))
