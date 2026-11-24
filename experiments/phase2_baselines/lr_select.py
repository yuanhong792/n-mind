from __future__ import annotations
import json,sys,time,random
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
from nmind.language.tokenizer import CharTokenizer
from nmind.language.large_binding_task import make_train_test
from models import GRUBaseline,CausalTransformerBaseline,DeltaRuleLinearBaseline
def main():
 t0=time.perf_counter();torch.set_num_threads(1);tok=CharTokenizer.load(ROOT/'experiments/m5_language/tokenizer_m52.json');tr,dev=make_train_test(20000,600,ROOT/'experiments/m5_language/tokenizer_m52.json');V=tok.vocab_size; enc=[torch.tensor(tok.encode(r['prompt']+r['target'])) for r in tr]; devenc=[(torch.tensor(tok.encode(r['prompt']+r['target'])),len(tok.encode(r['prompt'])),r) for r in dev]; out=[]; best={}
 for name,cls in {'gru':GRUBaseline,'transformer':CausalTransformerBaseline,'delta_rule':DeltaRuleLinearBaseline}.items():
  for lr in (3e-4,1e-3,3e-3):
   if time.perf_counter()-t0>300: raise RuntimeError('LR selection 5-minute cap exceeded')
   torch.manual_seed(20270401);random.seed(20270401);m=cls(V);opt=torch.optim.AdamW(m.parameters(),lr=lr);ce=torch.nn.CrossEntropyLoss(); last=0
   for step in range(1000):
    idx=[(step*16+j)%len(enc) for j in range(16)]; z=torch.nn.utils.rnn.pad_sequence([enc[i] for i in idx],batch_first=True);opt.zero_grad();log=m(z[:,:-1]);log=log[0] if isinstance(log,tuple) else log;last=ce(log.reshape(-1,V),z[:,1:].reshape(-1));last.backward();opt.step()
   loss=0.;hits=0;count=0
   with torch.no_grad():
    for z,p,_ in devenc:
     log=m(z[:-1].unsqueeze(0));log=log[0] if isinstance(log,tuple) else log; ans=z[p:]; pred=log[0,p-1:p-1+len(ans)].argmax(-1); hits+=int(torch.equal(pred,ans));count+=1;loss+=float(ce(log.reshape(-1,V),z[1:].unsqueeze(0).reshape(-1)))
   rec={'model':name,'lr':lr,'steps':1000,'dev_rank_accuracy':hits/count,'dev_loss':loss/count,'wall_s':round(time.perf_counter()-t0,3)};out.append(rec)
  cand=[x for x in out if x['model']==name];best[name]=sorted(cand,key=lambda x:(-x['dev_rank_accuracy'],x['dev_loss']))[0]
 (ROOT/'experiments/phase2_baselines/lr_selection_20261124.json').write_text(json.dumps({'cap_s':300,'rows':out,'selected':best},indent=2),encoding='utf8');print(json.dumps({'rows':out,'selected':best},indent=2))
if __name__=='__main__':main()
