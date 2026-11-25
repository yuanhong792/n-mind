import torch
from torch import nn

class ZeroA(nn.Module):
    def __init__(self, input_dim=8, hidden=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden), nn.Tanh(), nn.Linear(hidden, 2))
    def forward(self, x):
        return self.net(x)

class FrozenZeroA(ZeroA):
    def __init__(self, input_dim=8, hidden=16):
        super().__init__(input_dim, hidden)
        for p in self.parameters(): p.requires_grad_(False)
