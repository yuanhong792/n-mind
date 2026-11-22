"""Deterministic LanguageWorld-derived train/dev/sealed records for Gate-2."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import hashlib, json
from .tokenizer_v2 import TokenizerV2

@dataclass(frozen=True)
class Gate2Item:
    item_id: str; level: str; prompt: str; answer: str; split: str; pair_id: str | None = None

_objs = ["\u82f9\u679c", "\u676f\u5b50", "\u4e66", "\u94a5\u5319", "\u76d2\u5b50", "\u7403", "\u79ef\u6728", "\u82b1"]
_colors = ["\u7ea2\u8272", "\u84dd\u8272", "\u7eff\u8272", "\u9ec4\u8272"]
_places = ["\u684c\u5b50", "\u76d2\u5b50", "\u5730\u677f", "\u67b6\u5b50"]

def build_gate2_dataset(seed: int = 20261123) -> dict[str, list[Gate2Item]]:
    train=[]; dev=[]; sealed=[]
    def add(dst, level, prompt, answer, pair=None):
        split = "sealed" if dst is sealed else "dev" if dst is dev else "train"
        dst.append(Gate2Item(f"{level}-{split}-{len(dst):04d}", level, prompt, answer, split, pair))
    for i in range(256):
        obj, color = _objs[i % 4], _colors[(i // 4) % 4]
        add(train, "L0", f"{obj}\u662f{color}\u7684\u3002{obj}\u662f\u4ec0\u4e48\u989c\u8272\uff1f", color)
    for i in range(64):
        obj, color = _objs[4 + i % 4], _colors[(i // 4) % 4]
        add(dev, "L0", f"\u89c2\u5bdf\u5230{obj}\u662f{color}\u7684\u3002\u8bf7\u95ee{obj}\u662f\u4ec0\u4e48\u989c\u8272\uff1f", color)
    for i in range(80):
        obj, color = _objs[4 + i % 4], _colors[(i // 4) % 4]
        add(sealed, "L0", f"\u73b0\u5728\u770b\u5230\u4e00\u4ef6{color}\u7684{obj}\u3002{obj}\u7684\u989c\u8272\u662f\uff1f", color)
    for i in range(512):
        a, b = _objs[i % 4], _objs[(i // 4) % 4 + 4]; p, q = _places[i % 2], _places[(i // 2) % 2 + 2]
        add(train, "L1", f"{a}\u5728{p}\u3002{b}\u5728{q}\u3002{a}\u5728\u54ea\u91cc\uff1f", p)
    for i in range(80):
        a, b = _objs[4 + i % 4], _objs[i % 4]; p, q = _places[(i // 2) % 2], _places[(i // 3) % 2 + 2]
        add(dev, "L1", f"{a}\u5728{p}\u3002{b}\u5728{q}\u3002\u628a{a}\u79fb\u52a8\u5230{q}\u3002{a}\u73b0\u5728\u5728\u54ea\u91cc\uff1f", q)
    for i in range(80):
        a, b = _objs[4 + i % 4], _objs[(i + 2) % 4]; p, q = _places[(i // 2) % 2], _places[(i // 3) % 2 + 2]
        pair=f"cf-{i:03d}"; add(sealed, "L1", f"{a}\u5728{p}\u3002{b}\u5728{q}\u3002{a}\u73b0\u5728\u5728\u54ea\u91cc\uff1f", p, pair)
        add(sealed, "L1", f"{a}\u5728{q}\u3002{b}\u5728{p}\u3002{a}\u73b0\u5728\u5728\u54ea\u91cc\uff1f", q, pair)
    return {"train": train, "dev": dev, "sealed": sealed}

def manifest(data: dict[str, list[Gate2Item]]) -> dict:
    body={k:[asdict(x) for x in v] for k,v in data.items()}
    tok=TokenizerV2(); encoded=sum(len(tok.encode(x.prompt+x.answer)) for x in data["train"])
    return {"format":"language-v2-gate2-dataset/1","seed":20261123,"counts":{k:len(v) for k,v in data.items()},"train_tokens":encoded,"sha256":hashlib.sha256(json.dumps(body,ensure_ascii=False,sort_keys=True).encode()).hexdigest(),"sealed_pair_count":len({x.pair_id for x in data["sealed"] if x.pair_id})}
