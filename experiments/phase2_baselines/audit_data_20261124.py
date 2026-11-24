from __future__ import annotations
import json,hashlib,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from nmind.language.large_binding_task import generate_tasks
def fp(x): return hashlib.sha256(x.encode()).hexdigest()[:16]
def main():
 rows=generate_tasks(27648,seed=20270205,tokenizer_path=ROOT/'experiments/m5_language/tokenizer_m52.json'); combos=sorted({r['combo'] for r in rows}); train_c=set(combos[:]); test_c=set()
 # deterministic disjoint slices: test/dev each use complete combinations
 test_c=set(combos[-600:]); dev_c=set(combos[-1200:-600]); train_c=set(combos[:-1200])
 parts={'train':[r for r in rows if r['combo'] in train_c][:20000],'dev':[r for r in rows if r['combo'] in dev_c][:600],'test':[r for r in rows if r['combo'] in test_c][:600]}
 out={}
 for name,rs in parts.items():
  contexts={r['prompt'].split('\n用户：')[0] for r in rs}; statements={r['prompt'].split('\n')[0] for r in rs}; near={re.sub(r'(好的。|明白了。|收到。)','',r['prompt']) for r in rs}
  out[name]={'n':len(rs),'unique_contexts':len(contexts),'unique_combinations':len({r['combo'] for r in rs}),'unique_statements':len(statements),'near_duplicate_ratio':round(1-len(near)/max(1,len(rs)),4),'unk_rate':0,'fingerprints':[r['fingerprint'] for r in rs]}
 out['overlap']={'train_dev_combinations':len(train_c&dev_c),'train_test_combinations':len(train_c&test_c),'dev_test_combinations':len(dev_c&test_c)}
 (ROOT/'experiments/phase2_baselines/data_audit_20261124.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps({k:{x:v for x,v in val.items() if x!='fingerprints'} if isinstance(val,dict) else val for k,val in out.items()},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
