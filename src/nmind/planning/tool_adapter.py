from __future__ import annotations
import re
from typing import Any

def build_tools(*, knowledge=None, memory=None) -> dict[str, Any]:
    def knowledge_search(q, _): return {"results": knowledge.search(q, limit=5) if knowledge else [], "response_source":"retrieval", "verified": bool(knowledge and knowledge.search(q,limit=1)), "evidence_refs": []}
    def memory_search(q, _):
        c = memory.recall(q) if memory else None
        return {"results": c.hits if c else [], "response_source":"memory", "verified": bool(c and c.hits), "evidence_refs": [f"memory:{h['id']}" for h in (c.hits if c else [])]}
    def calculator(q, _):
        m = re.search(r"(-?\d+(?:\.\d+)?)\s*([+*/-])\s*(-?\d+(?:\.\d+)?)", q)
        if not m: return {"verified":False,"response_source":"tool"}
        a,op,b=float(m[1]),m[2],float(m[3]); val={"+":a+b,"-":a-b,"*":a*b,"/":a/b if b else float("nan")}[op]
        return {"value":val,"verified":val==val,"response_source":"tool"}
    def evidence_compare(_, prev): return {"results":list(prev.values()),"verified":bool(prev),"response_source":"rule"}
    def citation_verify(_, prev): return {"verified":any(v.get("evidence_refs") or v.get("results") for v in prev.values()),"response_source":"rule"}
    return {"knowledge_search":knowledge_search,"memory_search":memory_search,"calculator":calculator,"evidence_compare":evidence_compare,"citation_verify":citation_verify}
