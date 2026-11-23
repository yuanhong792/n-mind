import torch
from nmind.language.neural_language_v2 import LanguageV2Config,LanguageV2Model
from nmind.language.minlearn_diagnostics import gradient_audit,SelectiveBlockRead,FullBlockRead,TinyGRUBaseline,candidate_registry
def test_gradient_audit_is_backward_only():
    torch.manual_seed(7); m=LanguageV2Model(LanguageV2Config(vocab_size=260,d_model=8,n_blocks=2,block_hidden=8,workspace_dim=4,workspace_slots=2))
    x=torch.randint(0,260,(1,5)); y=torch.randint(0,260,(1,5)); a=gradient_audit(m,x,y)
    assert a.optimizer_steps==0 and not a.parameter_changed
    assert any(v>0 for k,v in a.grad_norms.items() if "head" in k)
def test_readout_candidates_preserve_shapes():
    assert SelectiveBlockRead(8,4,260)(torch.randn(1,2,8),{"content":torch.randn(1,2,4)}).shape==(1,260)
    assert FullBlockRead(2,8,4,260)(torch.randn(1,2,8),{"content":torch.randn(1,2,4)}).shape==(1,260)
    assert TinyGRUBaseline(260)(torch.randint(0,260,(1,5))).shape==(1,5,260)
def test_registry_is_not_training_authorization():
    assert candidate_registry()["training_steps"]==0

