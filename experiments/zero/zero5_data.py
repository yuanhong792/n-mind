import hashlib
import numpy as np
import torch

def feature_hashes(x):
    a=x.detach().cpu().numpy().astype('<f4')
    return [hashlib.sha256(np.ascontiguousarray(row.reshape(8)).tobytes()).hexdigest() for row in a]

class World:
    def __init__(self,seed):
        self.seed=seed
        self._rule=torch.randn(8,generator=torch.Generator().manual_seed(seed))
        self._sets={}
        for j,name in enumerate(('pool','reward','dev','sealed')):
            x=torch.randn(256 if name=='pool' else 128,8,generator=torch.Generator().manual_seed(seed*10+j+1))
            self._sets[name]=(x,(x@self._rule>0).long())
        self.seen=set()
    def candidates(self): return self._sets['pool'][0].clone()
    def feedback(self,i):
        if not 0<=i<256 or i in self.seen: raise ValueError('ILLEGAL_ACTION')
        self.seen.add(i); return self._sets['pool'][1][i].clone()
    def audit(self):
        hs={k:set(feature_hashes(v[0])) for k,v in self._sets.items()}
        return {'counts':{k:len(v) for k,v in hs.items()},'intersections':{a+'-'+b:len(hs[a]&hs[b]) for a in hs for b in hs if a<b}}
