from __future__ import annotations
import json,time,sys,statistics
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
from models import GRUBaseline,CausalTransformerBaseline,DeltaRuleLinearBaseline
from nmind.language.tokenizer import CharTokenizer
from nmind.language.large_binding_task import make_train_test
def main():
 torch.set_num_threads(1);V=32;tok=CharTokenizer.load(ROOT/'experiments/m5_language/tokenizer_m52.json');_,dev=make_train_test(20000,600,ROOT/'experiments/m5_language/tokenizer_m52.json');x=torch.randint(0,V,(16,32));y=torch.randint(0,V,(16,32));ce=torch.nn.CrossEntropyLoss();out=[]
 for name,cls in {'gru':GRUBaseline,'transformer':CausalTransformerBaseline,'delta':DeltaRuleLinearBaseline}.items():
  samples=[]
  for rep in range(6):
   m=cls(V);opt=torch.optim.AdamW(m.parameters(),lr=1e-3);t=time.perf_counter()
   for _ in range(20):
    opt.zero_grad();z=m(x);z=z[0] if isinstance(z,tuple) else z;loss=ce(z.reshape(-1,V),y.reshape(-1));loss.backward();opt.step()
   samples.append((time.perf_counter()-t)*1000/20)
  ds=[];t=time.perf_counter()
  with torch.no_grad():
   for r in dev:
    z=torch.tensor(tok.encode(r['prompt']+r['target']));q=m(z[:-1].unsqueeze(0));q=q[0] if isinstance(q,tuple) else q;ds.append(float(ce(q.reshape(-1,V),z[1:].unsqueeze(0).reshape(-1))))
  out.append({'model':name,'params':sum(p.numel() for p in m.parameters()),'step_ms_median':statistics.median(samples[1:]),'step_ms_min':min(samples[1:]),'step_ms_max':max(samples[1:]),'warmup_discarded':samples[0],'dev_eval_s':time.perf_counter()-t,'dev_n':len(dev),'dev_loss':sum(ds)/len(ds)})
 (ROOT/'experiments/phase2_baselines/timing_matched_20261124.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
