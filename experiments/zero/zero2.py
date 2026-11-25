"""ZERO-2 transfer experiment. Training is blocked unless --authorize is explicit."""
import argparse, json, time
from pathlib import Path
import torch
from .environment import E1Environment
from .models import ZeroA
from .learning import accuracy

ROOT=Path(__file__).parent
def task_pair(seed):
    return E1Environment(seed=seed), E1Environment(seed=seed+1000)

def prepare(seed=101, steps=128):
    a,b=task_pair(seed); ax,ay=a.sample(steps,'train'); bx,by=b.sample(steps,'train');
    _,aseal_y=a.sample(512,'test'); _,bseal_y=b.sample(512,'test')
    return {"seed":seed,"steps":steps,"environment_a_seed":seed,"environment_b_seed":seed+1000,
            "same_sample_order":True,"a_train_labels":ay.tolist(),"b_train_labels":by.tolist(),
            "a_sealed_labels":aseal_y.tolist(),"b_sealed_labels":bseal_y.tolist()}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--authorize',action='store_true'); ap.add_argument('--seeds',default='101,202,303'); args=ap.parse_args()
    if not args.authorize:
        print(json.dumps({"status":"PREPARED_ONLY","message":"Training blocked; rerun with explicit --authorize after budget approval.","proposed_updates":768},indent=2)); return
    rows=[]; total_updates=0
    for seed in map(int,args.seeds.split(',')):
        a,b=task_pair(seed); ax,ay=a.sample(128,'train'); bx,by=b.sample(128,'train'); av,avy=a.sample(128,'dev'); bv,bvy=b.sample(128,'dev'); aseal,asealy=a.sample(512,'test'); bseal,bsealy=b.sample(512,'test')
        torch.manual_seed(seed); fresh=ZeroA(); inherited=ZeroA(); inherited.load_state_dict(torch.load(ROOT/'checkpoints'/f'zero_a_seed{seed}.pt',weights_only=True))
        t0=0.5; t1=accuracy(inherited,bv,bvy)
        def train_curve(model):
            opt=torch.optim.SGD(model.parameters(),lr=0.03); curve=[]; start=time.perf_counter(); before={k:v.detach().clone() for k,v in model.state_dict().items()}
            for i in range(129):
                if i in (0,1,5,10,20,50,100,128): curve.append({"step":i,"dev_accuracy":accuracy(model,bv,bvy)})
                if i==128: break
                opt.zero_grad(set_to_none=True); loss=torch.nn.functional.cross_entropy(model(bx[i:i+1]),by[i:i+1]); loss.backward(); opt.step()
            delta={k:float((model.state_dict()[k]-before[k]).norm()) for k in before}
            return curve,time.perf_counter()-start,delta
        c2,sec2,d2=train_curve(fresh); c3,sec3,d3=train_curve(inherited); total_updates+=256
        torch.save(fresh.state_dict(),ROOT/'checkpoints'/f'zero2_t2_seed{seed}.pt'); torch.save(inherited.state_dict(),ROOT/'checkpoints'/f'zero2_t3_seed{seed}.pt')
        def auc(c): return sum((c[i]['dev_accuracy']+c[i+1]['dev_accuracy'])/2*(c[i+1]['step']-c[i]['step']) for i in range(len(c)-1))/128
        def threshold(c,t):
            for p in c:
                if p['dev_accuracy']>=t:return p['step']
            return None
        rows.append({"seed":seed,"t0_random":t0,"t1_frozen_inherited_b":t1,"t2_curve":c2,"t3_curve":c3,"t2_auc":auc(c2),"t3_auc":auc(c3),"t2_to_80":threshold(c2,.8),"t3_to_80":threshold(c3,.8),"t2_to_90":threshold(c2,.9),"t3_to_90":threshold(c3,.9),"t2_b_sealed":accuracy(fresh,bseal,bsealy),"t3_b_sealed":accuracy(inherited,bseal,bsealy),"t2_a_sealed_after":accuracy(fresh,aseal,asealy),"t3_a_sealed_after":accuracy(inherited,aseal,asealy),"t2_seconds":sec2,"t3_seconds":sec3,"t2_weight_delta":d2,"t3_weight_delta":d3,"b_train_class1":float(by.float().mean()),"b_dev_class1":float(bvy.float().mean()),"b_sealed_class1":float(bsealy.float().mean()),"a_sealed_class1":float(asealy.float().mean())})
    out={"status":"EXECUTED","rows":rows,"total_new_updates":total_updates,"budget":768,"torch":torch.__version__}
    (ROOT/'results'/'ZERO2_TRANSFER_RESULTS.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    (ROOT/'results'/'ZERO2_LEARNING_CURVES.json').write_text(json.dumps({"rows":[{"seed":r['seed'],"t2":r['t2_curve'],"t3":r['t3_curve']} for r in rows]},indent=2),encoding='utf-8')
    (ROOT/'results'/'ZERO2_RETENTION_ANALYSIS.json').write_text(json.dumps({"rows":[{"seed":r['seed'],"t2_a":r['t2_a_sealed_after'],"t3_a":r['t3_a_sealed_after']} for r in rows]},indent=2),encoding='utf-8')
    (ROOT/'results'/'ZERO2_WEIGHT_AUDIT.json').write_text(json.dumps({"rows":[{"seed":r['seed'],"t2":r['t2_weight_delta'],"t3":r['t3_weight_delta']} for r in rows]},indent=2),encoding='utf-8')

if __name__=='__main__': main()
