"""ZERO-2 transfer experiment. Training is blocked unless --authorize is explicit."""
import argparse, json
from pathlib import Path
import torch
from .environment import E1Environment
from .models import ZeroA
from .learning import online_train, accuracy

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
    rows=[]
    for seed in map(int,args.seeds.split(',')):
        a,b=task_pair(seed); ax,ay=a.sample(128,'train'); bx,by=b.sample(128,'train'); _,aseal=a.sample(512,'test'); _,bseal=b.sample(512,'test')
        torch.manual_seed(seed); fresh=ZeroA(); inherited=ZeroA(); inherited.load_state_dict(torch.load(ROOT/'checkpoints'/f'zero_a_seed{seed}.pt',weights_only=True))
        t2=online_train(fresh,bx,by,128); t3=online_train(inherited,bx,by,128)
        rows.append({"seed":seed,"t2_b_accuracy":accuracy(fresh,*bseal),"t3_b_accuracy":accuracy(inherited,*bseal),"t3_a_after_b":accuracy(inherited,*aseal),"t2_stats":t2,"t3_stats":t3})
    (ROOT/'results'/'ZERO2_TRANSFER_RESULTS.json').write_text(json.dumps({"status":"EXECUTED","rows":rows},indent=2),encoding='utf-8')

if __name__=='__main__': main()
