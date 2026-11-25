"""Persistent append-only budget ledger; reservations cannot silently reset."""
import json, os, time
import msvcrt
from contextlib import contextmanager
from pathlib import Path
LEDGER=Path(__file__).parent/'results'/'ZERO_BUDGET_LEDGER.jsonl'
STATE=Path(__file__).parent/'results'/'ZERO3_1_BUDGET_STATE.json'
def append(kind, amount, run_id, note='', ledger_path=None):
    rec={'ts':time.time(),'kind':kind,'amount':int(amount),'run_id':run_id,'note':note}
    target=Path(ledger_path) if ledger_path else LEDGER
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('a',encoding='utf-8') as f: f.write(json.dumps(rec)+'\n')
    return rec
def total():
    if not LEDGER.exists(): return 0
    return sum(json.loads(x)['amount'] for x in LEDGER.read_text(encoding='utf-8').splitlines() if x.strip())

class Budget:
    def __init__(self, auth_id, limit, state_path=None):
        self.auth_id=auth_id; self.limit=int(limit); self.state_path=Path(state_path) if state_path else STATE
        self.ledger_path=LEDGER if state_path is None else self.state_path.with_suffix('.jsonl')
        self.state_path.parent.mkdir(parents=True,exist_ok=True)
        with self._lock():
            if self.state_path.exists():
                self.s=json.loads(self.state_path.read_text(encoding='utf-8'))
                if self.s['authorization_id']!=auth_id or self.s['limit']!=self.limit: raise RuntimeError('BUDGET_AUTH_MISMATCH')
            else:
                self.s={'authorization_id':auth_id,'limit':self.limit,'reserved':0,'completed':0,'uncertain':0,'failed':0,'status':'OPEN'}; self._save()
    def _save(self):
        tmp=self.state_path.with_suffix('.tmp'); tmp.write_text(json.dumps(self.s,indent=2),encoding='utf-8'); tmp.replace(self.state_path)
    @contextmanager
    def _lock(self):
        lock=self.state_path.with_suffix('.lock'); lock.parent.mkdir(parents=True,exist_ok=True)
        with lock.open('a+b') as f:
            f.seek(0,os.SEEK_END)
            if f.tell()==0: f.write(b'0'); f.flush()
            f.seek(0); msvcrt.locking(f.fileno(),msvcrt.LK_LOCK,1)
            try: yield
            finally: f.seek(0); msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
    def reserve(self,n,kind):
        n=int(n)
        if n<=0: raise ValueError('RESERVATION_MUST_BE_POSITIVE')
        with self._lock():
            self.s=json.loads(self.state_path.read_text(encoding='utf-8'))
            if self.s['reserved']+n>self.limit: raise RuntimeError('BUDGET_EXCEEDED')
            self.s['reserved']+=n; self._save(); append('reserve',n,self.auth_id,kind,self.ledger_path); return n
    def complete(self,n,kind):
        n=int(n)
        with self._lock():
            self.s=json.loads(self.state_path.read_text(encoding='utf-8'))
            if n<=0 or self.s['completed']+self.s.get('uncertain',0)+n>self.s['reserved']: raise RuntimeError('COMPLETION_EXCEEDS_RESERVED')
            self.s['completed']+=n; self._save(); append('complete',n,self.auth_id,kind,self.ledger_path)
    def mark_uncertain(self,n,kind):
        n=int(n)
        with self._lock():
            self.s=json.loads(self.state_path.read_text(encoding='utf-8'))
            accounted=self.s['completed']+self.s.get('uncertain',0)
            if n<=0 or accounted+n>self.s['reserved']: raise RuntimeError('UNCERTAIN_EXCEEDS_RESERVED')
            self.s['uncertain']=self.s.get('uncertain',0)+n; self._save(); append('uncertain',n,self.auth_id,kind,self.ledger_path)
    def resolve_uncertain(self,n,kind):
        n=int(n)
        with self._lock():
            self.s=json.loads(self.state_path.read_text(encoding='utf-8'))
            if n<=0 or n>self.s.get('uncertain',0): raise RuntimeError('UNCERTAIN_RESOLUTION_EXCEEDS_RESERVED')
            # Resolve to failed cost; never refund an attempted update to the budget.
            self.s['uncertain']-=n; self.s['failed']=self.s.get('failed',0)+n
            self._save(); append('failed_cost',n,self.auth_id,kind,self.ledger_path)
    def snapshot(self):
        with self._lock():
            self.s=json.loads(self.state_path.read_text(encoding='utf-8'))
            return dict(self.s)
