from __future__ import annotations
import json,sys,time,random,os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
from nmind.language.tokenizer import CharTokenizer
from nmind.language.large_binding_task import make_train_test
from models import GRUBaseline,CausalTransformerBaseline,DeltaRuleLinearBaseline
OUT=ROOT/'experiments/phase2_baselines/ladder_20261124';OUT.mkdir(exist_ok=True)
LEVELS=(1000,3000,10000,30000); LRS={'gru':.001,'transformer':.001,'delta':.001}
def worker(arg):
 name,seed=arg; torch.set_num_threads(1);tok=CharTokenizer.load(ROOT/'experiments/m5_language/tokenizer_m52.json');tr,dev=make_train_test(20000,600,ROOT/'experiments/m5_language/tokenizer_m52.json');V=tok.vocab_size;cls={'gru':GRUBaseline,'transformer':CausalTransformerBaseline,'delta':DeltaRuleLinearBaseline}[name];m=cls(V);opt=torch.optim.AdamW(m.parameters(),lr=LRS[name]);ce=torch.nn.CrossEntropyLoss();enc=[torch.tensor(tok.encode(r['prompt']+r['target'])) for r in tr];t0=time.perf_counter();step=0;events=[]
 for level in LEVELS:
  while step<level:
   order=list(range(len(enc)));random.Random(seed+step//max(1,len(enc)//16)).shuffle(order)
   for j in range(0,len(order),16):
    if step>=level: break
    z=torch.nn.utils.rnn.pad_sequence([enc[i] for i in order[j:j+16]],batch_first=True);opt.zero_grad();q=m(z[:,:-1]);q=q[0] if isinstance(q,tuple) else q;loss=ce(q.reshape(-1,V),z[:,1:].reshape(-1));loss.backward();opt.step();step+=1
  hits=0;devloss=0
  with torch.no_grad():
   for r in dev:
    z=torch.tensor(tok.encode(r['prompt']+r['target']));q=m(z[:-1].unsqueeze(0));q=q[0] if isinstance(q,tuple) else q;p=len(tok.encode(r['prompt']));ans=z[p:];hits+=int(torch.equal(q[0,p-1:p-1+len(ans)].argmax(-1),ans));devloss+=float(ce(q.reshape(-1,V),z[1:].unsqueeze(0).reshape(-1)))
  tag=f'{name}_seed{seed}_step{level}';torch.save({'model':m.state_dict(),'seed':seed,'step':level,'reference':'phase2.3'},OUT/(tag+'.pt')); events.append({'level':level,'dev_rank_accuracy':hits/len(dev),'dev_loss':devloss/len(dev),'checkpoint':tag+'.pt','follow_rate':'NOT_COMPUTED'})
  if time.perf_counter()-t0>3600: return {'model':name,'seed':seed,'status':'INCONCLUSIVE_BUDGET','events':events}
 return {'model':name,'seed':seed,'status':'COMPLETE','events':events,'wall_s':time.perf_counter()-t0}
def main():
 args=[(n,s) for n in ('gru','transformer','delta') for s in (20270101,20270102,20270103)];t=time.perf_counter();rows=[]
 with ProcessPoolExecutor(max_workers=min(9,max(1,(os.cpu_count() or 8)-4))) as ex:
  fs=[ex.submit(worker,a) for a in args]
  for f in as_completed(fs,timeout=3600): rows.append(f.result())
 (OUT/'summary.json').write_text(json.dumps({'wall_s':time.perf_counter()-t,'rows':rows},indent=2),encoding='utf8');print(json.dumps({'wall_s':time.perf_counter()-t,'rows':rows},indent=2))
if __name__=='__main__':main()
