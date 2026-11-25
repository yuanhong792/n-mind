import torch
from torch import nn
class Explorer(nn.Module):
    def __init__(self): super().__init__(); self.net=nn.Sequential(nn.Linear(12,16),nn.Tanh(),nn.Linear(16,1))
    def scores(self,features,mask): return self.net(features).squeeze(-1).masked_fill(mask,float('-inf'))
def random_choice(mask,generator=None):
    ids=torch.where(~mask)[0]; return int(ids[torch.randint(len(ids),(1,),generator=generator)])
def diversity_choice(features, mask):
    ids=torch.where(~mask)[0]
    if len(ids)==0: raise RuntimeError('NO_LEGAL_ACTION')
    seen=torch.where(mask)[0]
    if len(seen)==0: return int(ids[0])
    d=torch.cdist(features[ids],features[seen]).min(1).values
    return int(ids[d.argmax()])
