"""ZERO-0 E1 reproducible experiment; default stays within 3*128 updates."""
import json, os, sys, time
from pathlib import Path
import torch
from .environment import E1Environment
from .models import ZeroA, FrozenZeroA
from .learning import online_train, accuracy
from .evaluate import weight_delta

ROOT = Path(__file__).parent
def run(seed, steps=128):
    torch.manual_seed(seed)
    env = E1Environment(seed=seed)
    train_x, train_y = env.sample(steps, "train")
    test_x, test_y = env.sample(512, "test")
    torch.manual_seed(seed); model = ZeroA(); before = [p.detach().clone() for p in model.parameters()]
    initial = accuracy(model, test_x, test_y)
    stats = online_train(model, train_x, train_y, steps)
    learned = accuracy(model, test_x, test_y)
    frozen = FrozenZeroA(); frozen_acc = accuracy(frozen, test_x, test_y)
    return {"seed": seed, "steps": steps, "initial_sealed_accuracy": initial, "learned_sealed_accuracy": learned,
            "frozen_sealed_accuracy": frozen_acc, "random_baseline": 0.5, "weight_l2_delta": weight_delta(before, model),
            **stats, "model_parameters": sum(p.numel() for p in model.parameters()), "state_dict": model.state_dict()}

def main():
    (ROOT/"results").mkdir(exist_ok=True); (ROOT/"checkpoints").mkdir(exist_ok=True)
    seeds=[101,202,303]; rows=[]
    for s in seeds:
        r=run(s); state=r.pop("state_dict"); rows.append(r)
        torch.save(state, ROOT / "checkpoints" / f"zero_a_seed{s}.pt")
    out={"status":"EXECUTED", "task":"E1_hidden_nonlinear", "seeds":seeds, "steps_per_seed":128, "rows":rows,
         "budget_updates":len(seeds)*128, "budget_limit":768, "torch":torch.__version__}
    (ROOT/"results"/"ZERO0_RESULTS.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
if __name__ == "__main__": main()
