"""Persistent append-only budget ledger; reservations cannot silently reset."""
import json, time
from pathlib import Path
LEDGER=Path(__file__).parent/'results'/'ZERO_BUDGET_LEDGER.jsonl'
def append(kind, amount, run_id, note=''):
    rec={'ts':time.time(),'kind':kind,'amount':int(amount),'run_id':run_id,'note':note}
    with LEDGER.open('a',encoding='utf-8') as f: f.write(json.dumps(rec)+'\n')
    return rec
def total():
    if not LEDGER.exists(): return 0
    return sum(json.loads(x)['amount'] for x in LEDGER.read_text(encoding='utf-8').splitlines() if x.strip())
