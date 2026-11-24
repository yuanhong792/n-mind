from __future__ import annotations
import math,random

def epoch_order(n, seed):
    x=list(range(n)); random.Random(seed).shuffle(x); return x
def epoch_count(n,batch_size=16,seen_samples=None):
    if batch_size < 16: raise ValueError('batch size must be >=16')
    return math.ceil((seen_samples if seen_samples is not None else n)/n)
def validate_param_match(count, reference, tolerance=.05):
    if abs(count-reference)/reference > tolerance: raise ValueError('parameter count outside ±5%')
    return True
def staircase_plan(train_n=20000,batch_size=16,levels=(32,128,512,1000)):
    return [{'level':i+1,'steps':s,'seen_samples':s*batch_size,'epochs':(s*batch_size+train_n-1)//train_n,'cumulative_steps':sum(levels[:i+1])} for i,s in enumerate(levels)]
