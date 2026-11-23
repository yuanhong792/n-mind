from __future__ import annotations
from .contracts import Plan

ALLOWED_KINDS = {"knowledge_search", "memory_search", "calculator", "compare_evidence", "synthesize_extract", "verify", "record_experience", "read_file", "knowledge_query", "experience_search", "structured_report"}
ALLOWED_TOOLS = {"knowledge_search", "memory_search", "calculator", "evidence_compare", "citation_verify", "record_verified_episode", "read_local_file", "knowledge_query", "experience_search", "structured_report"}

def validate_plan(plan: Plan, *, max_steps: int = 8) -> list[str]:
    errors: list[str] = []
    ids = [s.step_id for s in plan.steps]
    if len(ids) != len(set(ids)): errors.append("duplicate_step_id")
    if len(ids) > max_steps: errors.append("max_steps_exceeded")
    known = set(ids)
    for s in plan.steps:
        if s.kind not in ALLOWED_KINDS: errors.append(f"unsupported_kind:{s.kind}")
        if s.tool_id and s.tool_id not in ALLOWED_TOOLS: errors.append(f"unsupported_tool:{s.tool_id}")
        if any(d not in known for d in s.dependencies): errors.append(f"missing_dependency:{s.step_id}")
        if s.max_attempts < 1 or s.max_attempts > 2: errors.append(f"attempt_budget:{s.step_id}")
    # Kahn cycle check
    indeg = {i: 0 for i in ids}; edges = {i: [] for i in ids}
    for s in plan.steps:
        for d in s.dependencies: indeg[s.step_id] += 1; edges[d].append(s.step_id)
    q = [i for i, n in indeg.items() if n == 0]; seen = 0
    while q:
        cur = q.pop(0); seen += 1
        for nxt in edges[cur]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0: q.append(nxt)
    if seen != len(ids): errors.append("cycle")
    return errors
