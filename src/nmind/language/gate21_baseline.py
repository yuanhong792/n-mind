from __future__ import annotations
import torch
from .gate2_dataset import Gate2Item
from .neural_language_v2 import LanguageV2Config, LanguageV2Model
from .tokenizer_v2 import TokenizerV2
from .gate2_evaluator import _norm

def zero_step_gate2_baseline(items: list[Gate2Item], seed: int = 20261123, limit: int | None = None) -> dict:
    torch.manual_seed(seed); tok=TokenizerV2(); model=LanguageV2Model(LanguageV2Config(d_model=24,n_blocks=2,block_hidden=24,workspace_dim=12,workspace_slots=2)); model.eval(); rows=[]
    for item in (items if limit is None else items[:limit]):
        with torch.no_grad(): logits,state=model(torch.tensor([tok.encode(item.prompt)],dtype=torch.long)); generated=[]
        for _ in range(24):
            nxt=int(logits[:,-1].argmax()); generated.append(nxt)
            if nxt == 2: break
            with torch.no_grad(): logits,state=model(torch.tensor([[nxt]],dtype=torch.long),state)
        try: pred=tok.decode(generated)
        except UnicodeDecodeError: pred="<invalid-byte>"
        rows.append({"item_id":item.item_id,"pair_id":item.pair_id,"expected":item.answer,"prediction":pred,"exact_match":_norm(pred)==_norm(item.answer)})
    pairs={}
    for row in rows:
        if row["pair_id"]: pairs.setdefault(row["pair_id"],[]).append(row["exact_match"])
    return {"status":"ZERO_TRAINING_BASELINE","seed":seed,"optimizer_steps":0,"items":len(rows),"exact_match":sum(x["exact_match"] for x in rows)/max(1,len(rows)),"counterfactual_pair_pass":sum(all(x) for x in pairs.values())/max(1,len(pairs)),"counterfactual_pairs":len(pairs),"response_source":"neural_only_random_initialization"}
