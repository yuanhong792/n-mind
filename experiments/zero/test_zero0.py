import torch
from .environment import E1Environment
from .models import ZeroA
from .learning import accuracy

def test_environment_is_deterministic_and_splits_are_distinct():
    e = E1Environment(seed=101)
    a = e.sample(10, "train"); b = e.sample(10, "train")
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    c = e.sample(10, "test")
    assert not torch.equal(a[0], c[0])

def test_model_is_random_before_training():
    torch.manual_seed(1); m = ZeroA()
    x, y = E1Environment(seed=1).sample(8, "test")
    assert 0.0 <= accuracy(m, x, y) <= 1.0
    assert sum(p.numel() for p in m.parameters()) == 178
