from __future__ import annotations
from dataclasses import dataclass,asdict
import hashlib,json,random
@dataclass(frozen=True)
class MiniItem:
    item_id:str; group_id:str; split:str; prompt:str; answer:str; entity:str; queried:str; pos_a:str; pos_b:str; order:str; pair_id:str
ENTITIES=("小明","小红"); PLACES=("一号位置","三号位置")
def build_train16():
    out=[]
    for i in range(16):
        a,b=ENTITIES if i%2==0 else ENTITIES[::-1]; pa,pb=PLACES if (i//2)%2==0 else PLACES[::-1]
        q=a if (i//4)%2==0 else b; ans=pa if q==a else pb; order="ab" if i%2==0 else "ba"
        s=f"{a}在{pa}。{b}在{pb}。" if order=="ab" else f"{b}在{pb}。{a}在{pa}。"
        out.append(MiniItem(f"tr-{i:02d}",f"train-{i:02d}","train",s+f"问题：{q}在哪里？",ans,q,q,pa,pb,order,f"pair-{i//2:02d}"))
    return out
def build_dev():
    return [MiniItem("dev-00","dev-group","dev","小刚在一号位置。小芳在三号位置。问题：小刚在哪里？","一号位置","小刚","小刚","一号位置","三号位置","ab","dev-pair")]
def build_sealed():
    return [MiniItem("se-00","sealed-group","sealed","小刚在三号位置。小芳在一号位置。问题：小刚在哪里？","三号位置","小刚","小刚","三号位置","一号位置","ab","sealed-pair")]
def independent_labels(items,seed=20261130):
    rng=random.Random(seed); pool=[x.answer for x in items]; rng.shuffle(pool)
    return [MiniItem(**{**asdict(x),"answer":pool[i]}) for i,x in enumerate(items)]
def manifest():
    d={"train":build_train16(),"dev":build_dev(),"sealed":build_sealed()}
    body={k:[asdict(x) for x in v] for k,v in d.items()}
    return {"schema_version":"minlearn2/1","seed":20261130,"counts":{k:len(v) for k,v in d.items()},"sha256":hashlib.sha256(json.dumps(body,ensure_ascii=False,sort_keys=True).encode()).hexdigest(),"group_disjoint":True,"optimizer_steps":0}
def condition_table(items):
    return {"answer":{a:sum(x.answer==a for x in items) for a in sorted({x.answer for x in items})},"query":{a:sum(x.queried==a for x in items) for a in sorted({x.queried for x in items})},"order":{a:sum(x.order==a for x in items) for a in sorted({x.order for x in items})}}

