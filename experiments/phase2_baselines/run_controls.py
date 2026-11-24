"""Approved conditional control run: 3 models x 3 seeds, no N-MIND."""
from __future__ import annotations
import json,sys,time,random
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'src')); sys.path.insert(0,str(Path(__file__).parent))
from nmind.language.tokenizer import CharTokenizer
from nmind.language.large_binding_task import generate_tasks,split_by_combo
from models import GRUBaseline,CausalTransformerBaseline,DeltaRuleLinearBaseline

MAX_STEPS=100; TOTAL_CAP=900; WALL_CAP=600
def main():
    t0=time.perf_counter(); tok=CharTokenizer.load(ROOT/'experiments/m5_language/tokenizer_m52.json'); rows=generate_tasks(600,tokenizer_path=ROOT/'experiments/m5_language/tokenizer_m52.json'); train,test=split_by_combo(rows)
    V=tok.vocab_size; models={'gru':GRUBaseline,'transformer':CausalTransformerBaseline,'delta_rule':DeltaRuleLinearBaseline}; seeds=[20270101,20270102,20270103]; out=ROOT/'experiments/phase2_baselines/controls_20261124'; out.mkdir(exist_ok=True)
    summary=[]; total=0
    for name,cls in models.items():
      for seed in seeds:
        if total+MAX_STEPS>TOTAL_CAP or time.perf_counter()-t0>WALL_CAP: raise RuntimeError('code budget gate exceeded')
        torch.manual_seed(seed); random.seed(seed); m=cls(V); opt=torch.optim.AdamW(m.parameters(),lr=.003); ce=torch.nn.CrossEntropyLoss(); encoded=[]
        for r in train:
            ids=tok.encode(r['prompt']+r['target']); encoded.append(torch.tensor(ids,dtype=torch.long))
        start=time.perf_counter()
        for step in range(MAX_STEPS):
            opt.zero_grad(); ix=random.randrange(len(encoded)); z=encoded[ix]; logits=m(z[:-1].unsqueeze(0)); loss=ce(logits.reshape(-1,V),z[1:].unsqueeze(0).reshape(-1)); loss.backward(); opt.step()
        total+=MAX_STEPS; samples=[]; hits=0
        with torch.no_grad():
          for r in test:
            ids=torch.tensor(tok.encode(r['prompt']+r['target']),dtype=torch.long); logits=m(ids[:-1].unsqueeze(0))[0]; p=len(tok.encode(r['prompt'])); ans=tok.encode(r['target']); pred=logits[p-1:p-1+len(ans)].argmax(-1).tolist(); em=pred==ans; hits+=int(em); samples.append({'fingerprint':r['fingerprint'],'target_token_ids':ans,'pred_token_ids':pred,'logits':logits[p-1:p-1+len(ans)].tolist(),'em':em})
        tag=f'{name}_seed{seed}'; torch.save({'model':m.state_dict(),'seed':seed,'timing_only':False},out/(tag+'.pt')); (out/(tag+'_logits.json')).write_text(json.dumps({'samples':samples},ensure_ascii=False),encoding='utf8')
        summary.append({'model':name,'seed':seed,'steps':MAX_STEPS,'wall_s':round(time.perf_counter()-start,3),'em':hits/len(test),'follow_rate':'NOT_COMPUTED_FROM_SINGLE_EXAMPLES','checkpoint':tag+'.pt','logits':tag+'_logits.json'})
    (out/'summary.json').write_text(json.dumps({'total_steps':total,'wall_s':round(time.perf_counter()-t0,3),'rows':summary},indent=2),encoding='utf8'); print(json.dumps({'total_steps':total,'wall_s':round(time.perf_counter()-t0,3),'rows':summary},indent=2))
if __name__=='__main__': main()
