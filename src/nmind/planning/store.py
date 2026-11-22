from __future__ import annotations
import json, sqlite3
from pathlib import Path
from .contracts import GoalSpec, Plan, StepResult, as_json, now

class PlanningStore:
    def __init__(self, path: str | Path):
        self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False); self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript("""CREATE TABLE IF NOT EXISTS planning_goals(goal_id TEXT PRIMARY KEY,payload TEXT NOT NULL,updated_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS planning_plans(task_id TEXT,version INTEGER,payload TEXT NOT NULL,PRIMARY KEY(task_id,version));
        CREATE TABLE IF NOT EXISTS planning_steps(task_id TEXT,step_id TEXT,payload TEXT NOT NULL,PRIMARY KEY(task_id,step_id));
        CREATE TABLE IF NOT EXISTS planning_events(id INTEGER PRIMARY KEY AUTOINCREMENT,task_id TEXT,kind TEXT,payload TEXT NOT NULL,created_at TEXT NOT NULL);""")
    def save_goal(self, g: GoalSpec): self.conn.execute("INSERT OR REPLACE INTO planning_goals VALUES(?,?,?)", (g.goal_id,json.dumps(as_json(g),ensure_ascii=False),now())); self.conn.commit()
    def save_plan(self, p: Plan):
        self.conn.execute("INSERT OR REPLACE INTO planning_plans VALUES(?,?,?)",(p.task_id,p.version,json.dumps(as_json(p),ensure_ascii=False)))
        for s in p.steps: self.conn.execute("INSERT OR REPLACE INTO planning_steps VALUES(?,?,?)",(p.task_id,s.step_id,json.dumps(as_json(s),ensure_ascii=False)))
        self.conn.commit()
    def result(self, r: StepResult): self.event(r.task_id,"step_result",as_json(r)); self.conn.execute("UPDATE planning_steps SET payload=? WHERE task_id=? AND step_id=?",(json.dumps(as_json(r),ensure_ascii=False),r.task_id,r.step_id)); self.conn.commit()
    def event(self, task_id: str, kind: str, payload: dict): self.conn.execute("INSERT INTO planning_events(task_id,kind,payload,created_at) VALUES(?,?,?,?)",(task_id,kind,json.dumps(payload,ensure_ascii=False),now())); self.conn.commit()
    def events(self, task_id: str, limit=200): return [dict(r) for r in self.conn.execute("SELECT * FROM planning_events WHERE task_id=? ORDER BY id LIMIT ?",(task_id,limit)).fetchall()]
    def close(self): self.conn.close()
