"""Balanced anti-shortcut curriculum and audited controls for Gate-2.4."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import hashlib, json, random

@dataclass(frozen=True)
class Gate24Item:
    item_id: str; group_id: str; split: str; prompt: str; answer: str
    target: str; queried: str; target_position: str; last_position: str
    order: str; kind: str; pair_id: str | None = None

ENTITIES=("\u5c0f\u660e","\u5c0f\u7ea2","\u5c0f\u521a","\u5c0f\u82b3","\u5c0f\u674e","\u5c0f\u738b")
PLACES=("\u4e00","\u4e8c","\u4e09","\u56db")
PLACE_WORD=tuple(x+chr(0x53f7)+chr(0x4f4d)+chr(0x7f6e) for x in PLACES)

def _sentence(a,p,b,q,order):
    return f"{a}在{p}。{b}在{q}。" if order=="target_first" else f"{b}在{q}。{a}在{p}。"

def build_gate24(seed: int=20261123):
    train=[]; dev=[]; sealed=[]
    def add(dst,group,split,prompt,answer,target,queried,tp,lp,order,kind,pair=None):
        dst.append(Gate24Item(f"{split}-{len(dst):04d}",group,split,prompt,answer,target,queried,tp,lp,order,kind,pair))
    for i in range(80):
        a,b=ENTITIES[i%6],ENTITIES[(i+2)%6]; p,q=PLACE_WORD[i%4],PLACE_WORD[(i//2+1)%4]
        query=a if i%2==0 else b; answer=p if query==a else q; order="target_first" if i%2==0 else "target_last"
        dst,split=(train,"train") if i<40 else (dev,"dev") if i<52 else (sealed,"sealed")
        add(dst,f"g-{i:03d}",split,_sentence(a,p,b,q,order)+f"问题：{query}在哪里？",answer,a,query,p if query==a else q,q if query==a else p,order,"location")
    for i in range(40):
        a,b=ENTITIES[(i+1)%6],ENTITIES[(i+3)%6]; p,q=PLACE_WORD[(i+1)%4],PLACE_WORD[(i+3)%4]; pair=f"cf-{i:03d}"
        for flip,(x,y) in enumerate(((p,q),(q,p))):
            add(sealed,f"sg-{i:03d}-{flip}","sealed",_sentence(a,x,b,y,"target_first")+f"问题：{a}在哪里？",x,a,a,x,y,"target_first","counterfactual",pair)
    for i in range(20):
        a=ENTITIES[(i+4)%6]; old,new=PLACE_WORD[i%4],PLACE_WORD[(i+2)%4]
        add(sealed,f"up-{i:03d}","sealed",f"{a}在{old}。{a}移动到{new}。{a}现在在哪里？",new,a,a,new,new,"target_first","update",f"up-{i:03d}")
    return {"train":train,"dev":dev,"sealed":sealed}

def manifest(data):
    body={k:[asdict(x) for x in v] for k,v in data.items()}
    return {"format":"language-v2-gate24-data/1","seed":20261123,"counts":{k:len(v) for k,v in data.items()},"sha256":hashlib.sha256(json.dumps(body,ensure_ascii=False,sort_keys=True).encode()).hexdigest(),"sealed_pairs":len({x.pair_id for x in data["sealed"] if x.pair_id}),"semantic_group_disjoint":not({x.group_id for x in data["train"]}&{x.group_id for x in data["sealed"]})}

def stratified_label_permutation(items, seed=20261124):
    rng=random.Random(seed); out=list(items); changed=0; mapping=[]; strata={}
    for i,x in enumerate(items): strata.setdefault(x.kind,[]).append(i)
    for key,idxs in strata.items():
        vals=[items[i].answer for i in idxs]
        if len(set(vals))<2: raise ValueError(f"control infeasible: {key}")
        unique=list(dict.fromkeys(vals)); rng.shuffle(unique)
        found=None
        for shift in range(1,len(unique)):
            value_map={v:unique[(j+shift)%len(unique)] for j,v in enumerate(unique)}
            candidate=[value_map[v] for v in vals]
            if all(a!=b for a,b in zip(vals,candidate)): found=candidate; break
        if found is None: raise ValueError(f"no derangement: {key}")
        for i,answer in zip(idxs,found):
            changed += answer != items[i].answer
            out[i]=Gate24Item(**{**asdict(items[i]),"answer":answer})
            mapping.append({"item_id":items[i].item_id,"original":items[i].answer,"permuted":answer,"stratum":key})
    return out,{"seed":seed,"changed_fraction":changed/max(1,len(items)),"mapping":mapping,"fixed_points":len(items)-changed,"strata":{k:len(v) for k,v in strata.items()}}

def context_blind(items):
    return [Gate24Item(**{**asdict(x),"prompt":f"已知两个对象。问题：{x.queried}在哪里？"}) for x in items]

