from __future__ import annotations
import json,sys,random,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
from nmind.language.tokenizer import CharTokenizer
from nmind.language.large_binding_task import generate_tasks
from models import GRUBaseline,CausalTransformerBaseline,DeltaRuleLinearBaseline
def main():
 tok=CharTokenizer.load(ROOT/'experiments/m5_language/tokenizer_m52.json'); rows=generate_tasks(32,seed=20270206,tokenizer_path=ROOT/'experiments/m5_language/tokenizer_m52.json'); V=tok.vocab_size; out=[]
 for name,cls in {'gru':GRUBaseline,'transformer':CausalTransformerBaseline,'delta_rule':DeltaRuleLinearBaseline}.items():
  torch.manual_seed(20270301);m=cls(V);opt=torch.optim.AdamW(m.parameters(),lr=1e-3);ce=torch.nn.CrossEntropyLoss(); enc=[torch.tensor(tok.encode(r['prompt']+r['target'])) for r in rows]; last=None;t=time.perf_counter()
  for step in range(1000):
   random.Random(2027+step//2).shuffle(enc)
   for j in range(0,32,16):
    z=torch.nn.utils.rnn.pad_sequence(enc[j:j+16],batch_first=True); opt.zero_grad();log=m(z[:,:-1]);log=log[0] if isinstance(log,tuple) else log; loss=ce(log.reshape(-1,V),z[:,1:].reshape(-1));loss.backward();opt.step();last=float(loss)
   if last<.05: break
  out.append({'model':name,'steps':step+1,'loss':last,'wall_s':round(time.perf_counter()-t,3),'passed':last<.05})
 (ROOT/'experiments/phase2_baselines/overfit_20261124.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
