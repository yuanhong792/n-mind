"""Approved timing-only 20 optimizer-step benchmark; never writes checkpoints."""
from __future__ import annotations
import json,time,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'src')); sys.path.insert(0,str(Path(__file__).parent))
from models import GRUBaseline,CausalTransformerBaseline,DeltaRuleLinearBaseline
from nmind.language.neural_dialogue import NMindLM,LMConfig
def main():
    torch.manual_seed(20270101); V=32; x=torch.randint(0,V,(16,32)); y=torch.randint(0,V,(16,32)); ce=torch.nn.CrossEntropyLoss(); rows=[]
    models={'gru':GRUBaseline(V),'transformer':CausalTransformerBaseline(V),'delta_rule':DeltaRuleLinearBaseline(V),'nmind':NMindLM(LMConfig(vocab_size=V))}
    for name,m in models.items():
        opt=torch.optim.AdamW(m.parameters(),lr=1e-3); t=time.perf_counter(); steps=0
        for _ in range(20):
            opt.zero_grad(); z=m(x); z=z[0] if isinstance(z,tuple) else z; loss=ce(z.reshape(-1,V),y.reshape(-1)); loss.backward(); opt.step(); steps+=1
        rows.append({'model':name,'steps':steps,'wall_s':round(time.perf_counter()-t,4),'timing_only':True,'checkpoint_saved':False})
    out={'timing_only':True,'optimizer_step_cap':20,'rows':rows}; p=ROOT/'experiments/phase2_baselines/timing_only_20261124.json'; p.write_text(json.dumps(out,indent=2),encoding='utf8'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
