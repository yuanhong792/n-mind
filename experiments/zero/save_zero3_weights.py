import json, torch
from .zero3 import Controller, ROOT
d=json.loads((ROOT/'results'/'ZERO3_RESULTS.json').read_text())
(ROOT/'checkpoints').mkdir(exist_ok=True)
for x in d['eval']:
    c=Controller(x['mode']); c.load_state_dict({k:torch.tensor(v) for k,v in x['controller_state'].items()})
    torch.save(c.state_dict(), ROOT/'checkpoints'/f"zero3_{x['mode']}_seed{x['seed']}.pt")
print('saved',len(d['eval']))
