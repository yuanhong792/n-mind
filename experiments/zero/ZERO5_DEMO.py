"""Real no-training active-learning demo; no labels are exposed before choice."""
import argparse
from .ZERO5_ENVIRONMENT import UnknownWorld
from .ZERO5_EXPLORATION import random_choice
def main():
 p=argparse.ArgumentParser(); p.add_argument('--steps',type=int,default=4); a=p.parse_args(); w=UnknownWorld(505); import torch
 mask=torch.zeros(w.n,dtype=torch.bool)
 for _ in range(a.steps):
  i=random_choice(mask); y=w.feedback(i); mask[i]=True; print({'chosen':i,'feedback':y,'remaining':int((~mask).sum())})
if __name__=='__main__': main()
