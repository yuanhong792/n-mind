import torch
from .ZERO5_EXPLORATION import Explorer,diversity_choice
from .ZERO5_AGENT import Predictor
def test_mask_and_diversity_are_distinct():
 f=torch.tensor([[0.,0.],[1.,0.],[0.,2.]])
 m=torch.tensor([True,False,False]); assert diversity_choice(f,m)==2
 e=Explorer(); s=e.scores(torch.zeros(3,12),m); assert torch.isneginf(s[0])
def test_explorer_has_trainable_parameters():
 assert sum(p.numel() for p in Explorer().parameters())>0
def test_predictor_updates_can_be_measured():
 p=Predictor(); before=[x.detach().clone() for x in p.parameters()]; x=torch.randn(1,8); y=torch.tensor([1]); loss=torch.nn.functional.cross_entropy(p(x),y); loss.backward()
 for q,g in zip(p.parameters(),[q.grad for q in p.parameters()]): q.data.add_(g,alpha=-.1)
 assert any(not torch.equal(a,b) for a,b in zip(before,p.parameters()))
