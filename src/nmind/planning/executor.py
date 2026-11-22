from __future__ import annotations
from typing import Any, Callable
from .contracts import GoalSpec, Plan, PlanStep, StepResult, now
from .store import PlanningStore
from .validator import validate_plan

class PlanExecutor:
    def __init__(self, store: PlanningStore, tools: dict[str, Callable[..., dict[str, Any]]], *, max_replans=2, memory=None):
        self.store, self.tools, self.max_replans, self.memory = store, tools, max_replans, memory
        self.paused = False; self.cancelled = False
    def pause(self): self.paused = True
    def resume(self): self.paused = False
    def cancel(self): self.cancelled = True
    def run(self, goal: GoalSpec, plan: Plan) -> dict[str, Any]:
        errors = validate_plan(plan, max_steps=int(goal.budget.get("max_steps",8)))
        if errors: return {"task_id": plan.task_id, "status":"BLOCKED", "errors":errors, "results":[]}
        self.store.save_goal(goal); self.store.save_plan(plan); done: dict[str, StepResult] = {}; calls = 0
        while len(done) < len(plan.steps):
            if self.cancelled: return self._finish(plan,"CANCELLED",done)
            if self.paused: self.store.event(plan.task_id,"paused",{}); return self._finish(plan,"PAUSED",done)
            ready = [s for s in plan.steps if s.step_id not in done and all(d in done and done[d].status=="VERIFIED_SUCCESS" for d in s.dependencies)]
            if not ready: return self._finish(plan,"BLOCKED",done)
            for step in ready[:1]:
                if calls >= int(goal.budget.get("max_tool_calls",12)): return self._finish(plan,"BLOCKED",done)
                calls += 1; started=now(); tool=self.tools.get(step.tool_id or "")
                if tool is None:
                    r=StepResult(plan.task_id,step.step_id,"POLICY_BLOCKED","abstain","rule","unverified",failure_code="unsupported_tool")
                else:
                    try:
                        out=tool(goal.user_request, {k:v.output for k,v in done.items()}) or {}
                        ok=bool(out.get("verified", out.get("results") or out.get("value") is not None))
                        r=StepResult(plan.task_id,step.step_id,"VERIFIED_SUCCESS" if ok else "UNKNOWN",out.get("response_source","tool"),"rule", "deterministic_oracle" if ok else "unverified",out, list(out.get("evidence_refs",[])), started, now(), None if ok else "insufficient_evidence")
                    except Exception as exc: r=StepResult(plan.task_id,step.step_id,"VERIFIED_FAILURE","tool","rule","unverified",failure_code=type(exc).__name__,started_at=started,finished_at=now())
                done[step.step_id]=r; self.store.result(r)
        return self._finish(plan,"SUCCEEDED",done)
    def _finish(self, plan, status, done):
        self.store.event(plan.task_id,"task_status",{"status":status,"completed_steps":len(done)})
        result = {"task_id":plan.task_id,"status":status,"results":[asdict_result(r) for r in done.values()]}
        if self.memory is not None and status in ("SUCCEEDED", "FAILED", "BLOCKED"):
            from .reflection import record_experience
            result["memory_id"] = record_experience(self.memory, plan.task_id, plan.goal_id, result)
        return result
def asdict_result(r): return {"task_id":r.task_id,"step_id":r.step_id,"status":r.status,"response_source":r.response_source,"output":r.output,"evidence_refs":r.evidence_refs,"failure_code":r.failure_code}
