from __future__ import annotations
from dataclasses import dataclass
import hashlib, json, random

@dataclass(frozen=True)
class BindingCase:
    text: str; question: str; answer: str; pair_id: str; split: str

def build_counterfactual_pairs(n: int = 80, seed: int = 20261122) -> list[BindingCase]:
    rng=random.Random(seed); out=[]
    for i in range(n):
        a,b=rng.sample(["小明","小红","小刚","小丽","阿强","老王","小芳","大伟"],2)
        pa,pb=rng.sample(["一号位置","二号位置","三号位置","四号位置","五号位置","六号位置"],2)
        # Paired order reversal prevents a last-statement shortcut from being
        # mistaken for entity binding.
        first=f"{a}在{pa}。{b}在{pb}。"; second=f"{b}在{pb}。{a}在{pa}。"
        q=f"{a}在哪里？"; out += [BindingCase(first,q,pa,f"pair-{i:03d}","counterfactual"),BindingCase(second,q,pa,f"pair-{i:03d}","counterfactual")]
    return out

def manifest(cases: list[BindingCase]) -> dict:
    raw=json.dumps([c.__dict__ for c in cases],ensure_ascii=False,sort_keys=True).encode()
    return {"version":"language-v2-data-1","count":len(cases),"sha256":hashlib.sha256(raw).hexdigest(),"pair_count":len({c.pair_id for c in cases}),"source":"code-generated independent oracle"}
