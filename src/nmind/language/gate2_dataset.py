"""Deterministic LanguageWorld-derived train/dev/sealed records for Gate-2."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import hashlib, json, random
from .tokenizer_v2 import TokenizerV2

@dataclass(frozen=True)
class Gate2Item:
    item_id: str; level: str; prompt: str; answer: str; split: str; pair_id: str | None = None

_objs = ["苹果", "杯子", "书", "钥匙", "盒子", "球", "积木", "花"]
_colors = ["红色", "蓝色", "绿色", "黄色"]
_places = ["桌子", "盒子", "地板", "架子"]

def build_gate2_dataset(seed: int = 20261123) -> dict[str, list[Gate2Item]]:
    rng = random.Random(seed); train=[]; dev=[]; sealed=[]
    def add(dst, level, idx, prompt, answer, pair=None):
        split = "sealed" if dst is sealed else "dev" if dst is dev else "train"
        dst.append(Gate2Item(f"{level}-{split}-{len(dst):04d}", level, prompt, answer, split, pair))
    # L0 uses combinations disjoint from held-out object/colour combinations.
    for i in range(256):
        obj, color, place = _objs[i % 4], _colors[(i // 4) % 4], _places[(i // 16) % 2]
        add(train, "L0", i, f"{obj}是{color}的。{obj}是什么颜色？", color)
    for i in range(64):
        obj, color = _objs[4 + i % 4], _colors[(i // 4) % 4]
        add(dev, "L0", i, f"观察到{obj}是{color}的。请问{obj}是什么颜色？", color)
    for i in range(80):
        obj, color = _objs[4 + (i % 4)], _colors[(i // 4) % 4]
        add(sealed, "L0", i, f"现在看到一件{color}的{obj}。{obj}的颜色是？", color)
    # L1 is stateful: each pair reverses locations, and sealed entities are new.
    for i in range(512):
        a, b = _objs[i % 4], _objs[(i // 4) % 4 + 4]; p, q = _places[i % 2], _places[(i // 2) % 2 + 2]
        add(train, "L1", i, f"{a}在{p}。{b}在{q}。{a}在哪里？", p)
    for i in range(80):
        a, b = _objs[4 + i % 4], _objs[i % 4]; p, q = _places[(i // 2) % 2], _places[(i // 3) % 2 + 2]
        add(dev, "L1", i, f"{a}在{p}。{b}在{q}。把{a}移动到{q}。{a}现在在哪里？", q)
    for i in range(80):
        a, b = _objs[4 + i % 4], _objs[(i + 2) % 4]; p, q = _places[(i // 2) % 2], _places[(i // 3) % 2 + 2]
        pair=f"cf-{i:03d}"; add(sealed, "L1", i, f"{a}在{p}。{b}在{q}。{a}现在在哪里？", p, pair)
        add(sealed, "L1", i, f"{a}在{q}。{b}在{p}。{a}现在在哪里？", q, pair)
    return {"train": train, "dev": dev, "sealed": sealed}

def manifest(data: dict[str, list[Gate2Item]]) -> dict:
    body={k:[asdict(x) for x in v] for k,v in data.items()}
    tok=TokenizerV2(); encoded=sum(len(tok.encode(x.prompt+x.answer)) for x in data["train"])
    return {"format":"language-v2-gate2-dataset/1","seed":20261123,"counts":{k:len(v) for k,v in data.items()},"train_tokens":encoded,"sha256":hashlib.sha256(json.dumps(body,ensure_ascii=False,sort_keys=True).encode()).hexdigest(),"sealed_pair_count":len({x.pair_id for x in data["sealed"] if x.pair_id})}
