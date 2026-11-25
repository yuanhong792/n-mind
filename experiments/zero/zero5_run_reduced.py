import argparse,json,random,hashlib
import numpy as np
import torch
from .zero5_data import World
from .ZERO5_AGENT import Predictor,ROOT,AUTH,LIMIT
from .ZERO5_EXPLORATION import Explorer,random_choice,diversity_choice
from .budget_ledger import Budget

def episode(seed,init,policy,strategy,lr,budget,training=False,query='dev'):
    torch.manual_seed(init); random.seed(init); np.random.seed(init)
    world=World(seed); x=world.candidates(); model=Predictor(); opt=torch.optim.SGD(model.parameters(),lr=lr)
    popt=torch.optim.SGD(policy.parameters(),lr=.05); mask=torch.zeros(256,dtype=torch.bool)
    qx,qy=world._sets[query]; rx,ry=world._sets['reward']; trace=[]; curves=[]; logs=[]; rewards=[]
    for step in range(17):
        if step in (0,1,2,4,8,16):
            with torch.no_grad(): curves.append({'step':step,'accuracy':float((model(qx).argmax(1)==qy).float().mean()),'loss':float(torch.nn.functional.cross_entropy(model(qx),qy))})
        if step==16: break
        with torch.no_grad():
            prob=model(x).softmax(1); uncertainty=1-prob.max(1).values
            state=torch.cat((x,uncertainty[:,None],torch.full((256,1),1-step/16),torch.zeros(256,2)),1).detach().clone()
        scores=policy.scores(state,mask.clone()); dist=torch.distributions.Categorical(logits=scores)
        if strategy=='P0': action=random_choice(mask)
        elif strategy=='P1': action=int(uncertainty.masked_fill(mask,-1).argmax())
        elif strategy=='P2': action=diversity_choice(x,mask)
        else: action=int(dist.sample())
        logp=dist.log_prob(torch.tensor(action)); logs.append(logp)
        with torch.no_grad(): before=float(torch.nn.functional.cross_entropy(model(rx),ry)) if training else None
        label=world.feedback(action); mask[action]=True
        old=[p.detach().clone() for p in model.parameters()]
        budget.reserve(1,'reduced_task'); opt.zero_grad(); loss=torch.nn.functional.cross_entropy(model(x[action:action+1]),label.reshape(1)); loss.backward(); opt.step(); budget.complete(1,'reduced_task')
        with torch.no_grad(): reward=before-float(torch.nn.functional.cross_entropy(model(rx),ry))-.001 if training else None
        rewards.append(reward); trace.append({'action':action,'feedback':int(label),'log_prob':float(logp.detach()),'reward':reward,'lr':opt.param_groups[0]['lr'],'delta':float(sum((p-o).square().sum() for p,o in zip(model.parameters(),old)).sqrt())})
    gradient=None
    if training:
        returns=torch.tensor([sum(rewards[i:]) for i in range(16)])
        ploss=-(torch.stack(logs)*returns).mean(); popt.zero_grad(); ploss.backward()
        gradient=float(sum(p.grad.square().sum() for p in policy.parameters()).sqrt())
        if not np.isfinite(gradient) or gradient==0: raise RuntimeError('INVALID_POLICY_GRADIENT')
        budget.reserve(1,'reduced_policy'); popt.step(); budget.complete(1,'reduced_policy')
    return {'world':seed,'strategy':strategy,'curve':curves,'trace':trace,'policy_gradient':gradient}, {'model':model.state_dict(),'policy':policy.state_dict(),'optimizer':opt.state_dict(),'policy_optimizer':popt.state_dict(),'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'environment_seed':seed,'seen':mask,'budget':budget.snapshot()}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--authorize',action='store_true'); args=ap.parse_args()
    plan=3*2*3*16+3*8*17+3*3*5*16
    assert plan==1416
    if not args.authorize: print({'plan':plan}); return
    budget=Budget(AUTH,LIMIT,ROOT/'results'/'ZERO5_BUDGET_STATE.json')
    if budget.s['completed']!=304 or budget.s['reserved']!=304: raise RuntimeError('NO_AUTOMATIC_REPLAY')
    rows=[]; selected={}; outdir=ROOT/'results'/'zero5_reduced'; outdir.mkdir(exist_ok=True)
    def save(row,cp):
        rows.append(row); cp['position']=len(rows); cp['protocol']='reduced1416'; tmp=outdir/'checkpoint.tmp'; torch.save(cp,tmp); tmp.replace(outdir/'checkpoint.pt')
        (outdir/'rows.json').write_text(json.dumps(rows,indent=2))
    for s in (101,202,303):
        scores=[]
        for lr in (.03,.1,.2):
            vals=[]
            for t in range(2):
                torch.manual_seed(s); policy=Explorer(); row,cp=episode(710000+s*10+t,s*100+t,policy,'P0',lr,budget); save(row,cp); vals.append(row['curve'][-1]['loss'])
            scores.append((sum(vals)/2,lr))
        selected[s]=min(scores)[1]
    for s in (101,202,303):
        torch.manual_seed(s); policy=Explorer()
        for t in range(8):
            row,cp=episode(610000+s*10+t,s*100+t,policy,'P4',selected[s],budget,True); save(row,cp)
        torch.save(policy.state_dict(),outdir/f'policy_{s}.pt')
    for s in (101,202,303):
        for t in range(3):
            for group in ('P0','P1','P2','P3','P4'):
                torch.manual_seed(s); policy=Explorer()
                if group=='P4': policy.load_state_dict(torch.load(outdir/f'policy_{s}.pt',weights_only=True))
                row,cp=episode(810000+s*10+t,s*100+t,policy,group,selected[s],budget,query='sealed'); row['phase']='sealed'; save(row,cp)
    print(json.dumps({'budget':budget.snapshot(),'selected':selected}))
if __name__=='__main__': main()
