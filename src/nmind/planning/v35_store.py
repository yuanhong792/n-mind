from __future__ import annotations
import json, sqlite3
from pathlib import Path
from datetime import datetime, timezone

def now(): return datetime.now(timezone.utc).isoformat(timespec="seconds")

class V35PlanStore:
    def __init__(self, path: str | Path):
        self.db=sqlite3.connect(path,check_same_thread=False); self.db.row_factory=sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL"); self.db.execute("CREATE TABLE IF NOT EXISTS v35_plans(plan_id TEXT PRIMARY KEY,payload TEXT NOT NULL,status TEXT NOT NULL,approved INTEGER NOT NULL DEFAULT 0,updated_at TEXT NOT NULL)"); self.db.commit()
    def save(self, plan_id, payload, status="AWAITING_APPROVAL", approved=False): self.db.execute("INSERT OR REPLACE INTO v35_plans VALUES(?,?,?,?,?)",(plan_id,json.dumps(payload,ensure_ascii=False),status,int(approved),now())); self.db.commit()
    def get(self, plan_id):
        r=self.db.execute("SELECT * FROM v35_plans WHERE plan_id=?",(plan_id,)).fetchone(); return ({**dict(r),"payload":json.loads(r["payload"])} if r else None)
    def set_status(self, plan_id, status, approved=None):
        r=self.get(plan_id)
        if not r: raise KeyError(plan_id)
        self.db.execute("UPDATE v35_plans SET status=?,approved=?,updated_at=? WHERE plan_id=?",(status,int(r["approved"] if approved is None else approved),now(),plan_id)); self.db.commit(); return self.get(plan_id)
    def list(self): return [{**dict(r),"payload":json.loads(r["payload"])} for r in self.db.execute("SELECT * FROM v35_plans ORDER BY updated_at DESC LIMIT 50")]
    def close(self): self.db.close()
