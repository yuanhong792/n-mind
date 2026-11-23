from __future__ import annotations
import hashlib, json, re, sqlite3, uuid
from datetime import datetime, timezone
from pathlib import Path

def now(): return datetime.now(timezone.utc).isoformat(timespec="seconds")
def norm(s): return re.sub(r"\s+", " ", str(s).strip().casefold())

class KnowledgeGraphService:
    """本地结构化知识图：证据、断言、版本与冲突均可追溯。"""
    def __init__(self, path: str | Path, events=None):
        self.path=Path(path); self.path.parent.mkdir(parents=True, exist_ok=True); self.events=events
        self.db=sqlite3.connect(self.path, check_same_thread=False); self.db.row_factory=sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL"); self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS v34_entities(entity_id TEXT PRIMARY KEY,canonical_name TEXT NOT NULL,kind TEXT NOT NULL,aliases TEXT NOT NULL,created_at TEXT NOT NULL,updated_at TEXT NOT NULL,status TEXT NOT NULL DEFAULT 'active');
        CREATE TABLE IF NOT EXISTS v34_evidence(evidence_id TEXT PRIMARY KEY,source_kind TEXT NOT NULL,locator TEXT NOT NULL,record_hash TEXT NOT NULL,title TEXT,excerpt TEXT,license_state TEXT NOT NULL,verification_status TEXT NOT NULL,created_at TEXT NOT NULL,revoked_at TEXT);
        CREATE TABLE IF NOT EXISTS v34_assertions(assertion_id TEXT PRIMARY KEY,subject_id TEXT NOT NULL,predicate TEXT NOT NULL,object_value TEXT NOT NULL,object_entity_id TEXT,evidence_ids TEXT NOT NULL,verification_status TEXT NOT NULL,confidence REAL,effective_from TEXT,effective_to TEXT,recorded_at TEXT NOT NULL,supersedes_id TEXT,revoked_at TEXT,content_hash TEXT NOT NULL,FOREIGN KEY(subject_id) REFERENCES v34_entities(entity_id));
        CREATE TABLE IF NOT EXISTS v34_conflicts(conflict_id TEXT PRIMARY KEY,left_id TEXT NOT NULL,right_id TEXT NOT NULL,kind TEXT NOT NULL,status TEXT NOT NULL,created_at TEXT NOT NULL,resolution TEXT);
        CREATE TABLE IF NOT EXISTS v34_experiences(experience_id TEXT PRIMARY KEY,payload TEXT NOT NULL,verification_status TEXT NOT NULL,created_at TEXT NOT NULL,revoked_at TEXT);
        CREATE INDEX IF NOT EXISTS v34_alias_idx ON v34_entities(canonical_name);
        CREATE INDEX IF NOT EXISTS v34_assert_subject ON v34_assertions(subject_id,predicate);
        """); self.db.commit()
    def _emit(self, typ, payload):
        if self.events: self.events.append(typ,payload)
    def add_evidence(self, source_kind, locator, title="", excerpt="", license_state="USER_AUTHORIZED", verification_status="UNVERIFIED"):
        eid=uuid.uuid4().hex; h=hashlib.sha256((locator+excerpt).encode()).hexdigest(); self.db.execute("INSERT INTO v34_evidence VALUES(?,?,?,?,?,?,?,?,?,NULL)",(eid,source_kind,locator,h,title,excerpt[:2000],license_state,verification_status,now())); self.db.commit(); return {"evidence_id":eid,"locator":locator,"verification_status":verification_status}
    def add_entity(self, canonical_name, kind="concept", aliases=None):
        alias=sorted({norm(canonical_name),*(norm(a) for a in (aliases or []))}); existing=self.db.execute("SELECT entity_id FROM v34_entities WHERE canonical_name=? AND kind=? AND status='active'",(norm(canonical_name),kind)).fetchone()
        if existing: return self.entity(existing[0])
        eid=uuid.uuid4().hex; t=now(); self.db.execute("INSERT INTO v34_entities VALUES(?,?,?,?,?,?,?)",(eid,norm(canonical_name),kind,json.dumps(alias,ensure_ascii=False),t,t,"active")); self.db.commit(); self._emit("knowledge.entity_created",{"entity_id":eid,"canonical_name":canonical_name}); return self.entity(eid)
    def entity(self,eid):
        r=self.db.execute("SELECT * FROM v34_entities WHERE entity_id=?",(eid,)).fetchone(); return {**dict(r),"aliases":json.loads(r["aliases"])} if r else None
    def entities(self,q="",limit=50):
        rows=self.db.execute("SELECT * FROM v34_entities WHERE status='active' AND (canonical_name LIKE ? OR aliases LIKE ?) ORDER BY updated_at DESC LIMIT ?",(f"%{norm(q)}%",f"%{norm(q)}%",max(1,min(100,limit)))).fetchall(); return [{**dict(r),"aliases":json.loads(r["aliases"])} for r in rows]
    def add_assertion(self, subject_id, predicate, object_value, *, object_entity_id=None, evidence_ids=None, verification_status="UNVERIFIED", confidence=None, effective_from=None, effective_to=None):
        if not self.entity(subject_id): raise ValueError("unknown subject entity")
        evidence_ids=evidence_ids or []; aid=uuid.uuid4().hex; ch=hashlib.sha256(json.dumps([subject_id,predicate,object_value,effective_from,effective_to],ensure_ascii=False).encode()).hexdigest()
        active=self.db.execute("SELECT * FROM v34_assertions WHERE subject_id=? AND predicate=? AND revoked_at IS NULL AND effective_to IS NULL",(subject_id,predicate)).fetchall()
        for old in active:
            if old["object_value"] != str(object_value):
                cid=uuid.uuid4().hex; self.db.execute("INSERT INTO v34_conflicts VALUES(?,?,?,?,?,?,?)",(cid,old["assertion_id"],aid,"DIRECT_CONTRADICTION","OPEN",now(),None))
        self.db.execute("INSERT INTO v34_assertions VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",(aid,subject_id,predicate,str(object_value),object_entity_id,json.dumps(evidence_ids),verification_status,confidence,effective_from,effective_to,now(),None,None,ch)); self.db.commit(); self._emit("knowledge.assertion_created",{"assertion_id":aid,"subject_id":subject_id,"predicate":predicate,"verification_status":verification_status}); return self.assertion(aid)
    def assertion(self,aid):
        r=self.db.execute("SELECT * FROM v34_assertions WHERE assertion_id=?",(aid,)).fetchone();
        if not r:return None
        d=dict(r); d["evidence_ids"]=json.loads(d["evidence_ids"]); return d
    def assertions(self,subject_id=None, include_revoked=False):
        q="SELECT * FROM v34_assertions"; args=[]
        if subject_id: q+=" WHERE subject_id=?"; args.append(subject_id)
        if not include_revoked: q += " AND " if subject_id else " WHERE "; q += "revoked_at IS NULL"
        return [self.assertion(r[0]) for r in self.db.execute(q,args).fetchall()]
    def revoke_assertion(self,aid,reason="user_revoked"):
        self.db.execute("UPDATE v34_assertions SET revoked_at=? WHERE assertion_id=?",(now(),aid)); self.db.commit(); self._emit("knowledge.assertion_revoked",{"assertion_id":aid,"reason":reason}); return self.assertion(aid)
    def conflicts(self,status="OPEN"): return [dict(r) for r in self.db.execute("SELECT * FROM v34_conflicts WHERE status=? ORDER BY created_at DESC",(status,)).fetchall()]
    def resolve_conflict(self,cid, resolution, chosen_assertion_id=None):
        self.db.execute("UPDATE v34_conflicts SET status='RESOLVED',resolution=? WHERE conflict_id=?",(resolution,cid));
        if resolution=="REVOKE_LEFT":
            c=self.db.execute("SELECT left_id FROM v34_conflicts WHERE conflict_id=?",(cid,)).fetchone(); c and self.revoke_assertion(c[0],"conflict_resolved")
        elif resolution=="REVOKE_RIGHT":
            c=self.db.execute("SELECT right_id FROM v34_conflicts WHERE conflict_id=?",(cid,)).fetchone(); c and self.revoke_assertion(c[0],"conflict_resolved")
        self.db.commit(); self._emit("knowledge.conflict_resolved",{"conflict_id":cid,"resolution":resolution}); return {"conflict_id":cid,"status":"RESOLVED"}
    def relations(self,eid,depth=2,limit=100):
        seen={eid}; frontier=[eid]; out=[]
        for _ in range(max(1,min(2,depth))):
            nxt=[]
            for cur in frontier:
                for a in self.assertions(cur):
                    out.append(a)
                    if a.get("object_entity_id") and a["object_entity_id"] not in seen: seen.add(a["object_entity_id"]); nxt.append(a["object_entity_id"])
            frontier=nxt
        return out[:limit]
    def query(self, text, depth=2):
        es=self.entities(text); hits=[]
        for e in es: hits += self.relations(e["entity_id"],depth)
        return {"query":text,"answer_status":"SUPPORTED" if hits else "INSUFFICIENT_EVIDENCE","entities":es,"assertions":hits,"response_source":"memory","evidence_ids":sorted({x for a in hits for x in a["evidence_ids"]}),"relation_paths":[]}
    def add_experience(self,payload):
        if payload.get("verification_status") not in ("VERIFIED_WITHIN_SCOPE","UNVERIFIED","REVOKED"): raise ValueError("invalid verification_status")
        eid=uuid.uuid4().hex; self.db.execute("INSERT INTO v34_experiences VALUES(?,?,?,?,?)",(eid,json.dumps(payload,ensure_ascii=False),payload["verification_status"],now(),None)); self.db.commit(); self._emit("experience.recorded",{"experience_id":eid,"verification_status":payload["verification_status"]}); return {"experience_id":eid,**payload}
    def experiences(self,task_family=None):
        rows=self.db.execute("SELECT * FROM v34_experiences WHERE revoked_at IS NULL ORDER BY created_at DESC").fetchall(); out=[]
        for r in rows:
            d=json.loads(r["payload"]); d.update({"experience_id":r["experience_id"],"created_at":r["created_at"]});
            if not task_family or d.get("task_family")==task_family: out.append(d)
        return out
    def revoke_experience(self,eid): self.db.execute("UPDATE v34_experiences SET revoked_at=? WHERE experience_id=?",(now(),eid)); self.db.commit(); self._emit("experience.revoked",{"experience_id":eid}); return {"experience_id":eid,"revoked":True}
    def stats(self): return {"entities":self.db.execute("SELECT count(*) FROM v34_entities WHERE status='active'").fetchone()[0],"assertions":len(self.assertions()),"conflicts_open":len(self.conflicts()),"experiences":len(self.experiences()),"schema_version":1,"wal":"enabled"}
    def close(self): self.db.close()
