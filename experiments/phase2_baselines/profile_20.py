from __future__ import annotations
import json,time,sys
import argparse
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'src')); sys.path.insert(0,str(Path(__file__).parent))
from models import GRUBaseline,CausalTransformerBaseline
from nmind.language.neural_dialogue import NMindLM,LMConfig
def run(name,m,V=32,B=16,T=32):
 x=torch.randint(0,V,(B,T)); y=torch.randint(0,V,(B,T)); ce=torch.nn.CrossEntropyLoss(); opt=torch.optim.AdamW(m.parameters(),lr=1e-3); torch.manual_seed(1); t=time.perf_counter()
 parts={k:0.0 for k in ('forward','loss','backward','optimizer','python_loop')}
 for i in range(20):
  q=time.perf_counter(); z=m(x); z=z[0] if isinstance(z,tuple) else z; parts['forward']+=time.perf_counter()-q
  q=time.perf_counter(); loss=ce(z.reshape(-1,V),y.reshape(-1)); parts['loss']+=time.perf_counter()-q
  q=time.perf_counter(); loss.backward(); parts['backward']+=time.perf_counter()-q
  q=time.perf_counter(); opt.step(); opt.zero_grad(); parts['optimizer']+=time.perf_counter()-q
 total=time.perf_counter()-t; parts={k:round(v*1000,3) for k,v in parts.items()}; parts['python_loop']=round(max(0,total*1000-sum(parts.values())),3)
 keys=parts
 return {'model':name,'params':sum(x.numel() for x in m.parameters()),'batch':B,'seq_len':T,'threads':torch.get_num_threads(),'sparse_routing':name=='nmind','steps':20,'wall_s':round(total,4),'per_step_ms':round(total*1000/20,3),'profile_self_ms':keys}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--threads',type=int,default=20); a=ap.parse_args(); torch.set_num_threads(a.threads)
 V=32; rows=[run('gru',GRUBaseline(V)),run('transformer',CausalTransformerBaseline(V)),run('nmind',NMindLM(LMConfig(vocab_size=V)))]; p=ROOT/f'experiments/phase2_baselines/profile_20_threads{a.threads}_20261124.json'; p.write_text(json.dumps(rows,indent=2),encoding='utf8'); print(json.dumps(rows,indent=2))
if __name__=='__main__': main()
