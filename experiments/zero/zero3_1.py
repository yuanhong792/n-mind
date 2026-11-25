"""ZERO-3.1 causal evaluation; 672 authorized task updates, frozen controllers."""
import argparse,json,time,hashlib,torch
from pathlib import Path
from .models import ZeroA
from .zero3 import Controller,forward
from .budget_ledger import Budget
ROOT=Path(__file__).parent; AUTH='zero3.1-causal-672-v1'; LIMIT=672; RATES=[.03,.1,.2,.3]; STEPS=(0,1,2,4,8)
DEV=[410101,410102,420101,420102,430101,430102]; SEALED=[510101,510102,510103,510104,520101,520102,520103,520104,530101,530102,530103,530104]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def task(seed):
 from .environment import E1Environment
 e=E1Environment(seed=seed); return (*e.sample(8,'train'),*e.sample(64,'test'))
def adapt(model,ctrl,sx,sy,qx,qy,rate=None,budget=None,kind='eval'):
 opt=torch.optim.SGD(model.parameters(),lr=1.0); out=[]; lrs=[]; start=time.perf_counter()
 for i in range(9):
  if i in STEPS:
   with torch.no_grad(): out.append({'step':i,'accuracy':float((model(qx).argmax(1)==qy).float().mean()),'loss':float(torch.nn.functional.cross_entropy(model(qx),qy))})
  if i==8: break
  if budget: budget.reserve(1,kind)
  opt.zero_grad(set_to_none=True); loss=torch.nn.functional.cross_entropy(model(sx[i:i+1]),sy[i:i+1]); loss.backward()
  if rate is None:
   state=torch.stack((loss.detach(),torch.sqrt(sum(p.grad.square().sum() for p in model.parameters())+1e-12).detach(),torch.tensor(i/8),torch.sqrt(sum(p.square().sum() for p in model.parameters())),torch.tensor(.03 if not lrs else lrs[-1]),torch.tensor(1.)))
   with torch.no_grad(): eta=float(ctrl(state[None]).squeeze().clamp(1e-4,.1))
  else: eta=float(rate)
  for p in model.parameters(): p.data.add_(p.grad.data,alpha=-eta)
  lrs.append(eta)
  if budget: budget.complete(1,kind)
 return out,lrs,time.perf_counter()-start
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--authorize',action='store_true'); ap.add_argument('--status',action='store_true'); a=ap.parse_args()
 b=Budget(AUTH,LIMIT)
 if a.status: print(json.dumps(b.snapshot(),indent=2)); return
 if not a.authorize: print(json.dumps({'status':'CHECK_ONLY','budget':b.snapshot()},indent=2)); return
 manifest=ROOT/'ZERO3_1_TASK_MANIFEST.json'; source=Path(__file__); weights=[ROOT/'checkpoints'/f'zero3_m1_seed{s}.pt' for s in (101,202,303)]+[ROOT/'checkpoints'/f'zero3_m2_seed{s}.pt' for s in (101,202,303)]
 if not all(p.exists() for p in weights): raise RuntimeError('MISSING_CONTROLLER_WEIGHT')
 b.reserve(0,'preflight'); dev_choice={}
 for seed in (101,202,303):
  scores=[]
  for rate in RATES:
   vals=[]
   for ts in DEV[2*((seed//101)-1):2*((seed//101)-1)+2]:
    sx,sy,qx,qy=task(ts); m=ZeroA(); c,_,_=adapt(m,None,sx,sy,qx,qy,rate=rate,budget=b,kind='dev') ; vals.append(c[-1]['loss'])
   scores.append((sum(vals)/len(vals),rate))
  dev_choice[str(seed)]=min(scores)[1]
 results={'status':'EXECUTED','authorization_id':AUTH,'source_sha256':sha(source),'manifest_sha256':sha(manifest),'controller_sha256':{str(p.name):sha(p) for p in weights},'dev_selected_rates':dev_choice,'rows':[]}
 for seed in (101,202,303):
  c1=Controller('m1'); c1.load_state_dict(torch.load(ROOT/'checkpoints'/f'zero3_m1_seed{seed}.pt',weights_only=True)); c2=Controller('m2'); c2.load_state_dict(torch.load(ROOT/'checkpoints'/f'zero3_m2_seed{seed}.pt',weights_only=True)); torch.manual_seed(seed); c4=Controller('m2')
  cond=[('C0',None,dev_choice[str(seed)]),('C1',c1,None),('C2',c2,None),('C3',None,{101:.03771193,202:.04663801,303:.05086069}[seed]),('C4',c4,None)]
  for j,ts in enumerate(SEALED[4*((seed//101)-1):4*((seed//101)-1)+4]):
   sx,sy,qx,qy=task(ts); torch.manual_seed(seed*10000+j); base=ZeroA();
   for name,ctrl,rate in cond:
    m=ZeroA(); m.load_state_dict(base.state_dict()); curve,lrs,secs=adapt(m,ctrl,sx,sy,qx,qy,rate=rate,budget=b,kind='sealed'); results['rows'].append({'seed':seed,'task':ts,'condition':name,'curve':curve,'lrs':lrs,'seconds':secs})
 (ROOT/'results'/'ZERO3_1_RESULTS.json').write_text(json.dumps(results,indent=2),encoding='utf-8'); (ROOT/'results'/'ZERO3_1_BUDGET_FINAL.json').write_text(json.dumps(b.snapshot(),indent=2),encoding='utf-8'); print(json.dumps({'status':'EXECUTED','budget':b.snapshot()},indent=2))
if __name__=='__main__': main()
