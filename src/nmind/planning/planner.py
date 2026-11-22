from __future__ import annotations
import hashlib
from .contracts import GoalSpec, Plan, PlanStep
from .validator import validate_plan

class RulePlanner:
    """Deterministic baseline. Web/memory text never becomes executable instructions."""
    source = "rule"
    def plan(self, goal: GoalSpec) -> Plan:
        text = goal.user_request
        tid = goal.goal_id
        steps: list[PlanStep] = []
        def add(n, kind, objective, tool, deps=(), verifier="evidence"):
            steps.append(PlanStep(f"{tid[:8]}-{n}", tid, kind, objective, tuple(deps), tool,
                                  verifier_id=verifier, idempotency_key=hashlib.sha256(f"{tid}:{n}".encode()).hexdigest()[:16]))
        low = text.lower()
        if any(k in text for k in ("比较", "区别", "对比")):
            add(1, "knowledge_search", "检索相关本地资料", "knowledge_search")
            add(2, "memory_search", "检索历史方法", "memory_search")
            add(3, "compare_evidence", "比较检索证据", "evidence_compare", (f"{tid[:8]}-1", f"{tid[:8]}-2"))
            add(4, "verify", "核对来源与结论", "citation_verify", (f"{tid[:8]}-3",))
        elif any(k in text for k in ("计算", "算", "+", "-", "*", "/")):
            add(1, "calculator", "执行确定性计算", "calculator")
            add(2, "verify", "验证计算结果", "citation_verify", (f"{tid[:8]}-1",))
        else:
            add(1, "knowledge_search", "检索授权本地知识", "knowledge_search")
            add(2, "memory_search", "检索可复用历史经验", "memory_search")
            add(3, "synthesize_extract", "基于证据整理结果", "evidence_compare", (f"{tid[:8]}-1", f"{tid[:8]}-2"))
            add(4, "verify", "检查证据充分性", "citation_verify", (f"{tid[:8]}-3",))
        errors = validate_plan(Plan(tid, tid, 1, self.source, tuple(steps)), max_steps=int(goal.budget.get("max_steps", 8)))
        return Plan(tid, tid, 1, self.source, tuple(steps), validation_errors=tuple(errors))
