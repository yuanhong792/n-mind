import torch
from .learning import accuracy

def evaluate(model, x, y):
    return {"sealed_accuracy": accuracy(model, x, y)}

def weight_delta(before, model):
    return float(sum((a-b).pow(2).sum() for a,b in zip(before, model.parameters())).sqrt())
