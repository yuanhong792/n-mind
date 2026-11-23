from __future__ import annotations
import json, re
from pathlib import Path

class SafeActionRegistry:
    """白名单、无 shell、无网页脚本的本地动作。"""
    def __init__(self, *, workspace: str | Path, graph=None, knowledge=None, experiences=None):
        self.workspace=Path(workspace).resolve(); self.graph=graph; self.knowledge=knowledge; self.experiences=experiences
    def _local(self, raw):
        p=(self.workspace / str(raw)).resolve()
        if p != self.workspace and self.workspace not in p.parents: raise PermissionError("path outside approved workspace")
        return p
    def read_file(self, request, _prev):
        m=re.search(r"(?:读取|read)\s+([\w.\\/-]+)", request, re.I)
        if not m: return {"verified":False,"response_source":"tool","failure_code":"path_required"}
        p=self._local(m.group(1));
        if not p.is_file(): return {"verified":False,"response_source":"tool","failure_code":"file_not_found"}
        if p.stat().st_size>1024*1024: return {"verified":False,"response_source":"tool","failure_code":"size_limit"}
        return {"verified":True,"response_source":"tool","path":str(p.relative_to(self.workspace)),"text":p.read_text(encoding="utf-8")[:20000],"evidence_refs":[f"file:{p.relative_to(self.workspace)}"]}
    def knowledge_query(self, request, _prev):
        return self.graph.query(request) if self.graph else {"verified":False,"response_source":"retrieval"}
    def experience_search(self, request, _prev):
        hits=self.experiences(request) if self.experiences else []
        return {"verified":bool(hits),"response_source":"memory","results":hits,"evidence_refs":[f"experience:{x['experience_id']}" for x in hits]}
    def report(self, request, prev):
        return {"verified":bool(prev),"response_source":"tool","report":{"request":request,"steps":prev},"evidence_refs":[]}

    def tools(self): return {"read_local_file":self.read_file,"knowledge_query":self.knowledge_query,"experience_search":self.experience_search,"structured_report":self.report}
