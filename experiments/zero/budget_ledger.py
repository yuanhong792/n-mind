"""Persistent append-only budget ledger; reservations cannot silently reset."""
import json, time
from pathlib import Path
LEDGER=Path(__file__).parent/'results'/'ZERO_BUDGET_LEDGER.jsonl'
STATE=Path(__file__).parent/'results'/'ZERO3_1_BUDGET_STATE.json'
def append(kind, amount, run_id, note=''):
    rec={'ts':time.time(),'kind':kind,'amount':int(amount),'run_id':run_id,'note':note}
    with LEDGER.open('a',encoding='utf-8') as f: f.write(json.dumps(rec)+'\n')
    return rec
def total():
    if not LEDGER.exists(): return 0
    return sum(json.loads(x)['amount'] for x in LEDGER.read_text(encoding='utf-8').splitlines() if x.strip())

class Budget:
    def __init__(self, auth_id, limit):
        self.auth_id=auth_id; self.limit=int(limit)
        if STATE.exists():
            self.s=json.loads(STATE.read_text(encoding='utf-8'))
            if self.s['authorization_id']!=auth_id or self.s['limit']!=self.limit: raise RuntimeError('BUDGET_AUTH_MISMATCH')
        else:
            self.s={'authorization_id':auth_id,'limit':self.limit,'reserved':0,'completed':0,'uncertain':0,'status':'OPEN'}; self._save()
    def _save(self):
        tmp=STATE.with_suffix('.tmp'); tmp.write_text(json.dumps(self.s,indent=2),encoding='utf-8'); tmp.replace(STATE)
    def reserve(self,n,kind):
        n=int(n)
        if self.s['reserved']+n>self.limit: raise RuntimeError('BUDGET_EXCEEDED')
        self.s['reserved']+=n; self._save(); append('reserve',n,self.auth_id,kind); return n
    def complete(self,n,kind):
        n=int(n); self.s['completed']+=n; self._save(); append('complete',n,self.auth_id,kind)
    def resolve_uncertain(self,n,kind):
        n=int(n)
        if n>self.s.get('reserved',0): raise RuntimeError('UNCERTAIN_RESOLUTION_EXCEEDS_RESERVED')
        self.s['reserved']-=n; self.s['uncertain']=max(0,self.s.get('uncertain',0)-n); self._save(); append('resolve_uncertain',n,self.auth_id,kind)
    def snapshot(self): return dict(self.s)
