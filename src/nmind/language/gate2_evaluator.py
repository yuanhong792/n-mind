from __future__ import annotations
import re
from pathlib import Path
import torch
from .gate2_dataset import Gate2Item
from .neural_language_v2 import load_language_v2
from .tokenizer_v2 import TokenizerV2

def _norm(x: str) -> str: return re.sub(r"\s+", "", x)

def evaluate_checkpoint(path: str | Path, items: list[Gate2Item], *, max_new_tokens=24) -> dict:
    tok=TokenizerV2(); model,_=load_language_v2(path,tok); model.eval(); rows=[]
    for item in items:
        prompt_ids=tok.encode(item.prompt); state=None; generated=[]
        with torch.no_grad():
            logits,state=model(torch.tensor([prompt_ids],dtype=torch.long),state)
            for _ in range(max_new_tokens):
                nxt=int(logits[:,-1].argmax()); generated.append(nxt)
                if nxt == 2: break
                logits,state=model(torch.tensor([[nxt]],dtype=torch.long),state)
        try: text=tok.decode(generated)
        except UnicodeDecodeError: text="<invalid-byte>"
        rows.append({"item_id":item.item_id,"pair_id":item.pair_id,"expected":item.answer,"prediction":text,"exact_match":_norm(text)==_norm(item.answer),"source":"neural_only"})
    pairs={}
    for row in rows:
        if row["pair_id"]: pairs.setdefault(row["pair_id"],[]).append(row["exact_match"])
    return {"items":rows,"exact_match":sum(x["exact_match"] for x in rows)/max(1,len(rows)),"counterfactual_pair_pass":sum(all(v) for v in pairs.values())/max(1,len(pairs)),"counterfactual_pairs":len(pairs),"response_source":"neural"}
