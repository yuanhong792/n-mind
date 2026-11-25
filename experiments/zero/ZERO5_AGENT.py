import argparse,json,torch
from pathlib import Path
from torch import nn
from .ZERO5_ENVIRONMENT import UnknownWorld
from .ZERO5_EXPLORATION import Explorer,random_choice
from .budget_ledger import Budget
ROOT=Path(__file__).parent; AUTH='zero5-active-1860-v1'; LIMIT=1860
class Predictor(nn.Module):
 def __init__(self): super().__init__(); self.net=nn.Sequential(nn.Linear(8,16),nn.Tanh(),nn.Linear(16,2))
 def forward(self,x): return self.net(x)
def run(world_seed,strategy,budget,steps=16,explorer=None,train=True):
 w=UnknownWorld(world_seed); p=Predictor(); opt=torch.optim.SGD(p.parameters(),lr=.1); eopt=torch.optim.SGD(explorer.parameters(),lr=.05) if explorer and train else None; mask=torch.zeros(w.n,dtype=torch.bool); trace=[]; xs=[]; ys=[]
 for k in range(steps):
  feats=w.x; logits=p(feats); uncert=1-torch.softmax(logits,1).max(1).values; state=torch.cat([feats,uncert[:,None],torch.full((w.n,1),1-k/steps),torch.zeros(w.n,2)],1).detach()
  if strategy=='P0': i=random_choice(mask)
  elif strategy=='P1': i=int(uncert.masked_fill(mask,-1).argmax())
  elif strategy=='P2': i=random_choice(mask)
  else: i=int(explorer.scores(state,mask).argmax())
  y=w.feedback(i); mask[i]=True; x=w.observe(i).unsqueeze(0); target=torch.tensor([y]); budget.reserve(1,'agent_task'); opt.zero_grad(); loss=nn.functional.cross_entropy(p(x),target); loss.backward(); opt.step(); budget.complete(1,'agent_task'); xs.append(x.squeeze()); ys.append(y); trace.append({'step':k,'object':i,'feedback':y,'loss':float(loss.detach()),'weight_delta':float(sum(z.grad.norm() for z in p.parameters()))})
  if eopt and k==15:
   eopt.zero_grad(); reward=-loss.detach(); (-reward*torch.log_softmax(explorer.scores(state,mask),0)[i]).backward(); eopt.step(); budget.reserve(1,'strategy'); budget.complete(1,'strategy')
 sealed=torch.arange(16,32); acc=float((p(w.x[sealed]).argmax(1)==torch.tensor([int((w.x[j]@w.w>0).item()) for j in sealed])).float().mean())
 return {'world':world_seed,'strategy':strategy,'accuracy':acc,'trace':trace,'predictor':p.state_dict()}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--authorize',action='store_true'); a=ap.parse_args(); b=Budget(AUTH,LIMIT,ROOT/'results'/'ZERO5_BUDGET_STATE.json')
 if not a.authorize: print(json.dumps({'status':'PREPARED_ONLY','budget':b.snapshot()},indent=2)); return
 results=[]; seeds=[101,202,303];
 for s in seeds:
  for rate in (.03,.1,.2):
   for t in range(2):
    run(700000+s*10+t,'P0',b,16,None,False)
  
 for s in seeds:
  ex=Explorer();
  for t in range(12): run(s*100+t,'P4',b,16,ex,True)
  torch.save(ex.state_dict(),ROOT/'checkpoints'/f'zero5_p4_{s}.pt')
 for s in seeds:
  for t in range(4):
   for name in ('P0','P1','P2','P3','P4'):
    ex=Explorer();
    if name=='P4': ex.load_state_dict(torch.load(ROOT/'checkpoints'/f'zero5_p4_{s}.pt',weights_only=True)); ex.eval()
    r=run(900000+s*10+t,name,b,16,ex,False); r.pop('predictor'); results.append(r)
 out={'status':'EXECUTED','budget':b.snapshot(),'rows':results}; (ROOT/'results').mkdir(exist_ok=True); (ROOT/'results'/'ZERO5_RESULTS.json').write_text(json.dumps(out,indent=2),encoding='utf-8'); print(json.dumps({'status':'EXECUTED','budget':b.snapshot()},indent=2))
if __name__=='__main__': main()
