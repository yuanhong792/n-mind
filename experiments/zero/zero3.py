"""ZERO-3 dry-run implementation. No meta-training occurs without --authorize."""
import argparse, json
from pathlib import Path
import torch
from torch import nn
from .models import ZeroA

ROOT=Path(__file__).parent
class RateController(nn.Module):
    def __init__(self, dynamic=True):
        super().__init__()
        self.dynamic=dynamic
        self.net=nn.Sequential(nn.Linear(6,8),nn.Tanh(),nn.Linear(8,1)) if dynamic else nn.Parameter(torch.zeros(1))
    def forward(self,state):
        raw=self.net(state) if self.dynamic else self.net.expand(state.shape[0],1)
        return 1e-4 + (0.1-1e-4)*torch.sigmoid(raw)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--authorize',action='store_true'); a=ap.parse_args()
    status='READY_TO_RUN' if a.authorize else 'PREPARED_ONLY'
    if not a.authorize: print(json.dumps({'status':status,'message':'No ZERO-3 training budget is authorized.'},indent=2)); return
    raise RuntimeError('ZERO-3 budget must be filled into preregistration before training')
if __name__=='__main__': main()
