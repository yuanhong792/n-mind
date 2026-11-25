"""E1: random hidden nonlinear environment with disjoint train/sealed samples."""
from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class E1Environment:
    input_dim: int = 8
    classes: int = 2
    hidden: int = 12
    seed: int = 0

    def _rule(self):
        g = torch.Generator().manual_seed(self.seed)
        w1 = torch.randn(self.input_dim, self.hidden, generator=g) / self.input_dim**0.5
        b1 = torch.randn(self.hidden, generator=g) * 0.2
        w2 = torch.randn(self.hidden, generator=g) / self.hidden**0.5
        b2 = torch.randn((), generator=g) * 0.2
        return w1, b1, w2, b2

    def sample(self, n: int, split: str):
        seed = self.seed * 10000 + (101 if split == "train" else 202)
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(n, self.input_dim, generator=g)
        w1, b1, w2, b2 = self._rule()
        y = ((torch.tanh(x @ w1 + b1) @ w2 + b2) > 0).long()
        return x, y

    def oracle(self, x: torch.Tensor) -> torch.Tensor:
        w1, b1, w2, b2 = self._rule()
        return ((torch.tanh(x @ w1 + b1) @ w2 + b2) > 0).long()
