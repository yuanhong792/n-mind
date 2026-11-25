import time
import torch
from torch import nn

def accuracy(model, x, y):
    model.eval()
    with torch.no_grad(): return float((model(x).argmax(1) == y).float().mean())

def online_train(model, x, y, steps, lr=0.03):
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    start = time.perf_counter(); updates = 0; losses = []
    for i in range(min(steps, len(x))):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x[i:i+1]), y[i:i+1])
        loss.backward(); opt.step(); updates += 1; losses.append(float(loss.detach()))
    elapsed = time.perf_counter() - start
    return {"updates": updates, "loss_last": losses[-1], "loss_mean": sum(losses)/len(losses), "cpu_seconds": elapsed}
