import sys,re,json
sys.path.insert(0,'src')
from nmind.language.large_binding_task import generate_tasks
rows=generate_tasks(27648,seed=20270205,tokenizer_path='experiments/m5_language/tokenizer_m52.json'); combos=sorted({r['combo'] for r in rows}); tc=set(combos[-600:]); dc=set(combos[-1200:-600]); tr=[r for r in rows if r['combo'] not in tc|dc][:20000]; dv=[r for r in rows if r['combo'] in dc][:600]; te=[r for r in rows if r['combo'] in tc][:600]
def facts(rs): return {(e,p) for r in rs for e,p in re.findall(r'([^：。\n]+)在([1-4]号位置)',r['prompt'])}
def pairs(rs): return {tuple(sorted(re.findall(r'([^：。\n]+)在([1-4]号位置)',r['prompt'])[:2])) for r in rs}
P=pairs(tr); out={'facts':{k:len(facts(v)) for k,v in [('train',tr),('dev',dv),('test',te)]},'fact_pairs':{k:len(pairs(v)) for k,v in [('train',tr),('dev',dv),('test',te)]},'test_pair_in_train_ratio':sum(p in P for p in pairs(te))/max(1,len(pairs(te))),'dev_pair_in_train_ratio':sum(p in P for p in pairs(dv))/max(1,len(pairs(dv))),'test_unique_statements_in_train':len({r['prompt'].split('\n')[0] for r in te}&{r['prompt'].split('\n')[0] for r in tr})};json.dump(out,open('experiments/phase2_baselines/fact_audit_20261124.json','w'),indent=2);print(json.dumps(out,indent=2))
