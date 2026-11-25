"""ZERO-1 local plasticity experiment. No autograd is used by B2."""
import json, time
from pathlib import Path
import torch
from torch import nn
from .environment import E1Environment
from .models import ZeroA
from .learning import accuracy

ROOT = Path(__file__).parent

class ZeroB2(nn.Module):
    """Frozen random encoder + normalized Delta output synapses."""
    def __init__(self, input_dim=8, hidden=16, classes=2):
        super().__init__()
        torch.manual_seed(torch.initial_seed() % (2**63 - 1))
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden), nn.Tanh())
        self.weight = nn.Parameter(torch.zeros(classes, hidden), requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(classes), requires_grad=False)
        for p in self.encoder.parameters(): p.requires_grad_(False)

    def features(self, x):
        return self.encoder(x)

    def forward(self, x):
        h = self.features(x)
        return h @ self.weight.t() + self.bias

    @torch.no_grad()
    def local_delta(self, x, y, gain=1.0, eps=1e-6):
        h = self.features(x).squeeze(0)
        target = torch.zeros_like(self.bias); target[y.item()] = 1.0
        pred = self.weight @ h + self.bias
        err = target - pred
        denom = h.square().sum() + eps
        self.weight.add_(gain * err[:, None] * h[None, :] / denom)
        self.bias.add_(gain * err)
        return int(h.numel()), float(err.square().mean())

def run(seed, steps=128):
    torch.manual_seed(seed)
    env = E1Environment(seed=seed)
    train_x, train_y = env.sample(steps, "train")
    test_x, test_y = env.sample(512, "test")
    torch.manual_seed(seed); model = ZeroB2()
    before = {k: v.detach().clone() for k,v in model.state_dict().items()}
    curves = [{"step": 0, "sealed_accuracy": accuracy(model, test_x, test_y)}]
    start = time.perf_counter(); local_updates=0; synapses=0; losses=[]
    for i in range(steps):
        n, loss = model.local_delta(train_x[i:i+1], train_y[i:i+1]); local_updates += 1; synapses += n; losses.append(loss)
        if i + 1 in (10, 50, 100, 128): curves.append({"step": i+1, "sealed_accuracy": accuracy(model, test_x, test_y)})
    elapsed = time.perf_counter() - start
    delta = float(sum((model.state_dict()[k]-before[k]).square().sum() for k in before).sqrt())
    return {"seed": seed, "steps": steps, "initial_sealed_accuracy": curves[0]["sealed_accuracy"],
            "learned_sealed_accuracy": curves[-1]["sealed_accuracy"], "curves": curves,
            "weight_l2_delta": delta, "local_updates": local_updates, "local_synapse_updates": synapses,
            "backward_calls": 0, "optimizer_steps": 0, "cpu_seconds": elapsed,
            "loss_mean": sum(losses)/len(losses), "model_parameters": sum(p.numel() for p in model.parameters())}

def main():
    (ROOT/"results").mkdir(exist_ok=True); (ROOT/"checkpoints").mkdir(exist_ok=True)
    rows=[]
    for seed in (101,202,303):
        rows.append(run(seed))
        torch.manual_seed(seed); m=ZeroB2();
        # save the trained state from a deterministic rerun, avoiding hidden artifacts
        env=E1Environment(seed=seed); x,y=env.sample(128,"train")
        for i in range(128): m.local_delta(x[i:i+1], y[i:i+1])
        torch.save(m.state_dict(), ROOT/"checkpoints"/f"zero_b2_seed{seed}.pt")
    out={"status":"EXECUTED", "task":"E1_hidden_nonlinear", "model":"ZERO-B2", "rows":rows,
         "budget_updates":384, "budget_limit":768, "torch":torch.__version__}
    (ROOT/"results"/"ZERO1_RESULTS.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
    (ROOT/"results"/"ZERO1_LEARNING_CURVES.json").write_text(json.dumps({"rows":rows},indent=2),encoding="utf-8")
    print(json.dumps(out,indent=2))
if __name__ == "__main__": main()
