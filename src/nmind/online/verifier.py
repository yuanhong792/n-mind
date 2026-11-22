"""验证器：规则页 → 程序化校验 → 可执行 oracle（verified 才能进学习流）。"""

from __future__ import annotations

from typing import Any, Callable

from ..learning import datasets as D

DIRECTIONS = {"up": 0, "right": 1, "down": 2, "left": 3}   # 与 datasets.DIRS 对齐
POLICIES = {"min_distance", "max_distance"}
KNOWN_RULES = {"control-reversed-v1"}
VERIFIER_ID = "rule-verifier-v1"


def verify_rule(spec: dict[str, Any]) -> dict[str, Any]:
    """程序化校验规则表：字段完整、策略已知、模拟网格上策略永远可行。"""
    checks: dict[str, Any] = {}
    f = spec.get("fields", spec)
    for key in ("rule_id", "version", "policy"):
        checks[f"has_{key}"] = key in f and bool(str(f.get(key, "")).strip())
    checks["policy_known"] = str(f.get("policy", "")) in POLICIES
    checks["rule_id_known"] = str(f.get("rule_id", "")) in KNOWN_RULES
    # 模拟校验：空障碍 6x6 上，min/max 距离策略在任意非目标位置都有可行动作
    if checks["policy_known"]:
        legal_everywhere = True
        for x in range(6):
            for y in range(6):
                a = D.oracle_action((x, y), (5, 5), tuple(), 6,
                                    reversed_control=(f["policy"] == "max_distance"))
                legal_everywhere &= 0 <= a < 4
        checks["policy_always_legal"] = legal_everywhere
    ok = all(bool(v) for v in checks.values())
    return {"verification": "verified" if ok else "rejected", "checks": checks,
            "verifier_id": VERIFIER_ID}


def build_rule_oracle(spec: dict[str, Any]) -> Callable:
    """由已验证规则构造确定性 oracle（策略 = min/max 曼哈顿距离）。"""
    f = spec.get("fields", spec)
    rule_id = str(f.get("rule_id", ""))
    if rule_id not in KNOWN_RULES:
        raise ValueError(f"未知规则 {rule_id!r}：拒绝构造 oracle")
    policy = str(f["policy"])
    if policy not in POLICIES:
        raise ValueError(f"未知策略 {policy!r}")
    reversed_fallback = policy == "max_distance"

    def oracle(pos: tuple[int, int], goal: tuple[int, int],
               obstacles, size: int = 6, last_action: int | None = None) -> int:
        return D.oracle_action(pos, goal, obstacles, size,
                               reversed_control=reversed_fallback, last_action=last_action)

    oracle.rule_id = rule_id          # type: ignore[attr-defined]
    return oracle
