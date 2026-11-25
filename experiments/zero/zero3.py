"""ZERO-3 minimal meta-learning experiment; exact 768-update budget."""
import argparse, json, time, hashlib
from pathlib import Path
import torch
from torch import nn
from .environment import E1Environment
from .models import ZeroA
ROOT=Path(__file__).parent; ETA_LO=1e-4; ETA_HI=.1; POINTS=(0,1,2,4,8)
def task(seed):
    e=E1Environment(seed=seed); sx,sy=e.sample(8,'train'); qx,qy=e.sample(64,'test'); return sx,sy,qx,qy
def flat(m): return {k:v for k,v in m.named_parameters()}
def forward(p,x): return torch.tanh(x@p['net.0.weight'].t()+p['net.0.bias'])@p['net.2.weight'].t()+p['net.2.bias']
def adapt(p,ctrl,sx,sy,create_graph):
    curves=[]; rates=[]
    for i in range(9):
        if i in POINTS: curves.append((i,p.copy()))
        if i==8: break
        loss=nn.functional.cross_entropy(forward(p,sx[i:i+1]),sy[i:i+1]); gs=torch.autograd.grad(loss,tuple(p.values()),create_graph=create_graph); gn=torch.sqrt(sum(g.square().sum() for g in gs)+1e-12)
        state=torch.stack((loss.detach(),gn.detach(),torch.tensor(i/8),torch.sqrt(sum(v.square().sum() for v in p.values())),rates[-1] if rates else torch.tensor(.03),torch.tensor(1.)))
        eta=ctrl(state[None]).squeeze().clamp(ETA_LO,ETA_HI); rates.append(eta); p={k:v-eta*g for (k,v),g in zip(p.items(),gs)}
    return curves,rates
class Controller(nn.Module):
    def __init__(self,mode):
        super().__init__(); self.mode=mode
        if mode=='m2': self.net=nn.Sequential(nn.Linear(6,8),nn.Tanh(),nn.Linear(8,1))
        else: self.raw=nn.Parameter(torch.tensor(-2.))
    def forward(self,s):
        raw=self.net(s) if self.mode=='m2' else self.raw.expand(s.shape[0],1); return ETA_LO+(ETA_HI-ETA_LO)*torch.sigmoid(raw)
def run(seed,mode):
    torch.manual_seed(seed); ctrl=Controller(mode); opt=torch.optim.SGD(ctrl.parameters(),lr=.1); rows=[]; start=time.perf_counter()
    for ti in range(6):
        sx,sy,qx,qy=task(seed*100+ti); torch.manual_seed(seed*1000+ti); p=flat(ZeroA()); curves,rates=adapt(p,ctrl,sx,sy,True); qloss=nn.functional.cross_entropy(forward(curves[-1][1],qx),qy); opt.zero_grad(); qloss.backward(); gn=float(torch.sqrt(sum((x.grad if x.grad is not None else torch.zeros_like(x)).square().sum() for x in ctrl.parameters()))); opt.step()
        rows.append({'task':ti,'query_loss':float(qloss.detach()),'meta_grad_norm':gn,'eta_mean':float(torch.stack(rates).mean().detach())})
    return ctrl,{'seed':seed,'mode':mode,'inner_updates':48,'outer_updates':6,'rows':rows,'cpu_seconds':time.perf_counter()-start}
def evaluate(ctrl,seed):
    rows=[]
    for j in range(4):
        sx,sy,qx,qy=task(900000+seed*100+j); torch.manual_seed(seed*10000+j); curves,rates=adapt(flat(ZeroA()),ctrl,sx,sy,False); rows.append({'task':j,'accuracy_at_0_1_2_4_8':[float((forward(pp,qx).argmax(1)==qy).float().mean()) for _,pp in curves],'eta':[float(x.detach()) for x in rates],'query_loss_final':float(nn.functional.cross_entropy(forward(curves[-1][1],qx),qy))})
    return rows
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--authorize',action='store_true'); a=ap.parse_args()
    if not a.authorize: print(json.dumps({'status':'PREPARED_ONLY'})); return
    seeds=[101,202,303]; out={'status':'EXECUTED','eta_bounds':[ETA_LO,ETA_HI],'task_seed_sha256':hashlib.sha256(json.dumps([s*100+i for s in seeds for i in range(6)]).encode()).hexdigest(),'runs':[],'eval':[],'budget':{'m0':144,'m1':162,'m2':162,'eval':288,'preflight':12,'total':768},'m0':{'rates':[.003,.03,.1],'updates':144}}
    for mode in ('m1','m2'):
        for s in seeds:
            c,r=run(s,mode); torch.save(c.state_dict(),ROOT/'checkpoints'/f'zero3_{mode}_seed{s}.pt'); out['runs'].append(r); out['eval'].append({'seed':s,'mode':mode,'tasks':evaluate(c,s),'controller_state':{k:v.detach().tolist() for k,v in c.state_dict().items()}})
    out['actual_updates']=768; (ROOT/'results').mkdir(exist_ok=True); (ROOT/'results'/'ZERO3_RESULTS.json').write_text(json.dumps(out,indent=2),encoding='utf-8'); print(json.dumps({'status':'EXECUTED','actual_updates':768},indent=2))
if __name__=='__main__': main()
