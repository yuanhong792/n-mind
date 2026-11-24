from __future__ import annotations
import json,sys,time,os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
from nmind.language.tokenizer import CharTokenizer
from nmind.language.large_binding_task import make_train_test
from models import GRUBaseline,CausalTransformerBaseline,DeltaRuleLinearBaseline
OUT=ROOT/'experiments/phase2_baselines/lr_tasks_20261124'; OUT.mkdir(exist_ok=True)
def task(arg):
    name,lr,seed=arg; torch.set_num_threads(1); tok=CharTokenizer.load(ROOT/'experiments/m5_language/tokenizer_m52.json'); tr,dev=make_train_test(20000,600,ROOT/'experiments/m5_language/tokenizer_m52.json'); V=tok.vocab_size; cls={'gru':GRUBaseline,'transformer':CausalTransformerBaseline,'delta':DeltaRuleLinearBaseline}[name]; m=cls(V); opt=torch.optim.AdamW(m.parameters(),lr=lr); ce=torch.nn.CrossEntropyLoss(); enc=[torch.tensor(tok.encode(r['prompt']+r['target'])) for r in tr]; t=time.perf_counter()
    for step in range(1000):
        idx=[(step*16+j)%len(enc) for j in range(16)]; z=torch.nn.utils.rnn.pad_sequence([enc[i] for i in idx],batch_first=True); opt.zero_grad(); q=m(z[:,:-1]); q=q[0] if isinstance(q,tuple) else q; loss=ce(q.reshape(-1,V),z[:,1:].reshape(-1)); loss.backward(); opt.step()
    hits=0; devloss=0
    with torch.no_grad():
        for r in dev:
            z=torch.tensor(tok.encode(r['prompt']+r['target'])); q=m(z[:-1].unsqueeze(0)); q=q[0] if isinstance(q,tuple) else q; p=len(tok.encode(r['prompt'])); ans=z[p:]; hits+=int(torch.equal(q[0,p-1:p-1+len(ans)].argmax(-1),ans)); devloss+=float(ce(q.reshape(-1,V),z[1:].unsqueeze(0).reshape(-1)))
    rec={'model':name,'lr':lr,'seed':seed,'steps':1000,'dev_rank_accuracy':hits/len(dev),'dev_loss':devloss/len(dev),'wall_s':time.perf_counter()-t,'complete':True}; (OUT/f'{name}_lr{lr}.json').write_text(json.dumps(rec,indent=2),encoding='utf8'); return rec
def main():
    args=[(n,lr,20270501) for n in ('gru','transformer','delta') for lr in (3e-4,1e-3,3e-3)]; t=time.perf_counter(); done=[]
    with ProcessPoolExecutor(max_workers=min(9,max(1,(os.cpu_count() or 8)-4))) as ex:
        fs=[ex.submit(task,a) for a in args]
        for f in as_completed(fs,timeout=900): done.append(f.result())
    selected={}
    for n in ('gru','transformer','delta'):
        xs=[r for r in done if r['model']==n]; selected[n]=max(xs,key=lambda r:(r['dev_rank_accuracy'],-r['dev_loss'])) if len(xs)==3 else {'status':'INCONCLUSIVE_BUDGET'}
    (OUT/'summary.json').write_text(json.dumps({'wall_s':time.perf_counter()-t,'rows':done,'selected':selected},indent=2),encoding='utf8'); print(json.dumps({'wall_s':time.perf_counter()-t,'rows':done,'selected':selected},indent=2))
if __name__=='__main__': main()
