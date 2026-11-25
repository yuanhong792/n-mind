import torch
class UnknownWorld:
    def __init__(self,seed=0,n=256):
        self.seed=seed; self.n=n; g=torch.Generator().manual_seed(seed); self.x=torch.randn(n,8,generator=g); self.w=torch.randn(8,generator=g); self.seen=set()
    def observe(self,i): return self.x[i]
    def feedback(self,i): self.seen.add(int(i)); return int((self.x[i]@self.w>0).item())
    def hidden_labels(self): raise RuntimeError('hidden labels are inaccessible')
 
import torch
from torch import nn
class Explorer(nn.Module):
    def __init__(self): super().__init__(); self.net=nn.Sequential(nn.Linear(12,16),nn.Tanh(),nn.Linear(16,1))
    def scores(self,features,mask):
        s=self.net(features).squeeze(-1); return s.masked_fill(mask,float('-inf'))
def random_choice(mask,generator=None):
    ids=torch.where(~mask)[0]; return int(ids[torch.randint(len(ids),(1,),generator=generator)])
