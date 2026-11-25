"""ZERO-4 minimal continual-learning experiment."""
import argparse,json,torch
from pathlib import Path
from torch import nn
from .budget_ledger import Budget
ROOT=Path(__file__).parent; AUTH='zero4-continual-2568-v1'; LIMIT=2568
class Net(nn.Module):
 def __init__(self): super().__init__(); self.h=nn.Linear(9,16); self.o=nn.Linear(16,2)
 def forward(self,x): return self.o(torch.tanh(self.h(x)))
class Gate(nn.Module):
 def __init__(self,dynamic=True): super().__init__(); self.dynamic=dynamic; self.w=nn.Parameter(torch.zeros(2))
 def forward(self,state): return torch.sigmoid(self.w).expand(2) if not self.dynamic else torch.sigmoid(self.w+state[:2])
def sample(seed,n,ctx):
 g=torch.Generator().manual_seed(seed); x=torch.randn(n,8,generator=g); x9=torch.cat([x,torch.full((n,1),float(ctx))],1); w=torch.tensor([.8,-.5,.4,-.7,.3,.2,-.4,.6])*(1 if ctx==0 else -1); y=((x@w+0.2*torch.sin(x[:,0]*2))>0).long(); return x9,y
def step(net,gate,x,y,budget,rate=.1,kind='train'):
 budget.reserve(1,kind); net.zero_grad(); loss=nn.functional.cross_entropy(net(x),y); grads=torch.autograd.grad(loss,tuple(net.parameters())); g=gate(torch.stack([loss.detach(),torch.sqrt(sum(z.square().sum() for z in grads)).detach()])) if gate else torch.ones(2)
 for p,z,m in zip(net.parameters(),grads,[g[0],g[0],g[1],g[1]]): p.data.add_(z.data,alpha=-rate*float(m))
 budget.complete(1,kind); return float(loss.detach()),g.detach().tolist()
def fwd(p,x): return p['o.weight'] @ torch.tanh(p['h.weight'] @ x.t()+p['h.bias']) + p['o.bias']
def functional_adapt(net,gate,x,y):
 p=dict(net.named_parameters()); traces=[]
 for i in range(len(x)):
  loss=nn.functional.cross_entropy(fwd(p,x[i:i+1]),y[i:i+1]); gs=torch.autograd.grad(loss,tuple(p.values()),create_graph=True)
  st=torch.stack((loss.detach(),torch.sqrt(sum(z.square().sum() for z in gs)).detach())); gates=gate(st); p={k:v-.1*m*g for (k,v),g,m in zip(p.items(),gs,[gates[0],gates[0],gates[1],gates[1]])}; traces.append(gates)
 return p,traces
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--authorize',action='store_true'); ap.add_argument('--status',action='store_true'); a=ap.parse_args(); b=Budget(AUTH,LIMIT,ROOT/'results'/'ZERO4_BUDGET_STATE.json')
 if a.status: print(json.dumps(b.snapshot(),indent=2)); return
 if not a.authorize: print(json.dumps({'status':'PREPARED_ONLY','budget':b.snapshot()},indent=2)); return
 results=[]; seeds=[101,202,303]
 for s in seeds:
  for mode in ('G1','G2'):
   gate=Gate(mode=='G2'); opt=torch.optim.SGD(gate.parameters(),lr=.1)
   for t in range(12):
    net=Net(); sx,sy=sample(s*1000+t*2,16,0); qx,qy=sample(s*1000+t*2+1,16,1)
    for i in range(16): b.reserve(1,'meta_inner')
    p,tr=functional_adapt(net,gate,sx,sy); opt.zero_grad(); q=nn.functional.cross_entropy(fwd(p,qx),qy); q.backward(); opt.step(); b.complete(16,'meta_inner'); b.reserve(1,'meta_outer'); b.complete(1,'meta_outer')
   torch.save(gate.state_dict(),ROOT/'checkpoints'/f'zero4_{mode}_{s}.pt')
 for s in seeds:
  for rate in (.03,.1,.2,.3):
   for t in range(2):
    net=Net(); x,y=sample(700000+s*10+t,16,0)
    for i in range(16): step(net,None,x[i:i+1],y[i:i+1],b,rate=rate,kind='dev')
 for s in seeds:
  for t in range(4):
   for group in ('G0','G1','G2','G3','G4'):
    net=Net(); gate=None
    if group in ('G1','G2'): gate=Gate(group=='G2'); gate.load_state_dict(torch.load(ROOT/'checkpoints'/f'zero4_{group}_{s}.pt',weights_only=True)); gate.eval()
    if group=='G3': gate=Gate(True); torch.manual_seed(s); gate.eval()
    if group=='G4': gate=Gate(False); gate.eval()
    aX,aY=sample(800000+s*100+t*2,16,0); bX,bY=sample(800000+s*100+t*2+1,16,1)
    for i in range(8): step(net,gate,aX[i:i+1],aY[i:i+1],b,kind='sealed')
    a_before=float((net(aX).argmax(1)==aY).float().mean())
    for i in range(8): step(net,gate,bX[i:i+1],bY[i:i+1],b,kind='sealed')
    results.append({'seed':s,'task':t,'group':group,'a_after':float((net(aX).argmax(1)==aY).float().mean()),'b_after':float((net(bX).argmax(1)==bY).float().mean()),'a_before_b':a_before})
 out={'status':'EXECUTED','budget':b.snapshot(),'rows':results,'torch':torch.__version__}; (ROOT/'results').mkdir(exist_ok=True); (ROOT/'results'/'ZERO4_RESULTS.json').write_text(json.dumps(out,indent=2),encoding='utf-8'); print(json.dumps({'status':'EXECUTED','budget':b.snapshot()},indent=2))
if __name__=='__main__': main()
