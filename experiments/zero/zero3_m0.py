import json, torch
from .zero3 import task, flat, forward
from .models import ZeroA
def main():
 rows=[]
 for rate in (.003,.03,.1):
  vals=[]
  for seed in (101,202,303):
   for ti in range(6):
    sx,sy,qx,qy=task(seed*100+ti); torch.manual_seed(seed*1000+ti); p=flat(ZeroA())
    for i in range(8):
     loss=torch.nn.functional.cross_entropy(forward(p,sx[i:i+1]),sy[i:i+1]); gs=torch.autograd.grad(loss,tuple(p.values())); p={k:v-rate*g for (k,v),g in zip(p.items(),gs)}
    vals.append(float(torch.nn.functional.cross_entropy(forward(p,qx),qy)))
  rows.append({'rate':rate,'mean_query_loss':sum(vals)/len(vals),'n':len(vals)})
 out={'status':'EXECUTED','updates':144,'rows':rows,'selected_rate':min(rows,key=lambda x:x['mean_query_loss'])['rate']}
 open('experiments/zero/results/ZERO3_M0_RESULTS.json','w',encoding='utf-8').write(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
