"""Size-matched baseline model definitions; construction only, no training."""
from __future__ import annotations
import torch
from torch import nn

class GRUBaseline(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 257):
        super().__init__(); self.emb=nn.Embedding(vocab_size,d_model); self.gru=nn.GRU(d_model,d_model,batch_first=True); self.head=nn.Linear(d_model,vocab_size)
    def forward(self,x): return self.head(self.gru(self.emb(x))[0])

class CausalTransformerBaseline(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128, nhead: int = 4, layers: int = 2, ff_dim: int = 448):
        super().__init__(); self.emb=nn.Embedding(vocab_size,d_model); self.pos=nn.Embedding(512,d_model); enc=nn.TransformerEncoderLayer(d_model,nhead,dim_feedforward=ff_dim,batch_first=True); self.enc=nn.TransformerEncoder(enc,layers); self.head=nn.Linear(d_model,vocab_size)
    def forward(self,x):
        t=torch.arange(x.shape[1],device=x.device); mask=torch.triu(torch.ones(x.shape[1],x.shape[1],device=x.device),1).bool(); return self.head(self.enc(self.emb(x)+self.pos(t),mask=mask))

class DeltaRuleLinearBaseline(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 620):
        super().__init__(); self.emb=nn.Embedding(vocab_size,d_model); self.in_proj=nn.Linear(d_model,d_model); self.head=nn.Linear(d_model,vocab_size)
    def forward(self,x):
        h=torch.zeros(x.shape[0],self.in_proj.out_features,device=x.device); ys=[]
        for t in range(x.shape[1]): h=torch.tanh(self.in_proj(self.emb(x[:,t]))+h); ys.append(self.head(h))
        return torch.stack(ys,1)

def parameter_count(model): return sum(p.numel() for p in model.parameters())
def within_tolerance(count, reference, tolerance=0.05): return abs(count-reference)/reference <= tolerance
def assert_reference_match(models, reference=434553, tolerance=0.05):
    counts={name: parameter_count(model) for name,model in models.items()}
    bad={name:n for name,n in counts.items() if not within_tolerance(n,reference,tolerance)}
    if bad: raise ValueError(f'parameter hard gate failed: {bad}')
    return counts
