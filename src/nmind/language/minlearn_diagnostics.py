"""Zero-step learnability diagnostics and isolated architecture candidates."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import copy, torch
from torch import nn
from torch.nn import functional as F
from .neural_language_v2 import LanguageV2Config, LanguageV2Model
from .tokenizer_v2 import TokenizerV2

@dataclass
class GradientAudit:
    loss: float
    grad_norms: dict[str,float]
    parameter_changed: bool
    optimizer_steps: int
    note: str = "backward-only; no optimizer.step"

def gradient_audit(model: nn.Module, tokens: torch.Tensor, target: torch.Tensor) -> GradientAudit:
    model=copy.deepcopy(model).train()
    before={n:p.detach().clone() for n,p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    logits,_=model(tokens)
    loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),target.reshape(-1))
    loss.backward()
    norms={n:float(p.grad.norm()) if p.grad is not None else 0.0 for n,p in model.named_parameters()}
    changed=any(not torch.equal(before[n],p.detach()) for n,p in model.named_parameters())
    return GradientAudit(float(loss.detach()),norms,changed,0)

class SelectiveBlockRead(nn.Module):
    """Diagnostic head that reads a selected block plus workspace summary."""
    def __init__(self, block_hidden: int, workspace_dim: int, vocab_size: int):
        super().__init__()
        self.proj=nn.Linear(block_hidden+workspace_dim,vocab_size)
    def forward(self,h,workspace):
        return self.proj(torch.cat([h[:, -1, :],workspace["content"].mean(dim=1)],dim=-1))

class FullBlockRead(nn.Module):
    """Diagnostic head that preserves block identity before aggregation."""
    def __init__(self,n_blocks,block_hidden,workspace_dim,vocab_size):
        super().__init__()
        self.proj=nn.Linear(n_blocks*block_hidden+workspace_dim,vocab_size)
    def forward(self,h,workspace):
        return self.proj(torch.cat([h.flatten(1),workspace["content"].mean(dim=1)],dim=-1))

class TinyGRUBaseline(nn.Module):
    """Small randomly initialized recurrent comparator; no external weights."""
    def __init__(self,vocab_size,hidden=32):
        super().__init__(); self.emb=nn.Embedding(vocab_size,hidden); self.gru=nn.GRU(hidden,hidden,batch_first=True); self.head=nn.Linear(hidden,vocab_size)
    def forward(self,tokens):
        x=self.emb(tokens); y,_=self.gru(x); return self.head(y)

def byte_to_char_metrics(pred_ids:list[int], gold_ids:list[int], tok:TokenizerV2)->dict[str,Any]:
    pred=tok.decode(pred_ids); gold=tok.decode(gold_ids)
    n=min(len(pred),len(gold)); return {"pred":pred,"gold":gold,"char_exact":pred==gold,"char_accuracy":sum(a==b for a,b in zip(pred,gold))/max(1,len(gold)),"eos_exact":pred_ids[-1:]==gold_ids[-1:],"length_pred":len(pred),"length_gold":len(gold)}

def candidate_registry()->dict[str,Any]:
    return {"original":"NCell sparse route + h.mean + head","selective_block_workspace":"one selected block + workspace mean","full_block_workspace":"concatenate all block states + workspace mean","tiny_gru":"random-init GRU comparator","primary_variable_per_experiment":"readout architecture only","training_steps":0,"status":"PENDING_USER_APPROVAL"}

