from __future__ import annotations
from typing import Any
import torch
from .neural_language_v2 import LanguageV2Config, LanguageV2Model
from .tokenizer_v2 import TokenizerV2

def dry_run(*, seed=20261122, steps=0, device="cpu") -> dict[str, Any]:
    torch.manual_seed(seed); tok=TokenizerV2(); model=LanguageV2Model(LanguageV2Config())
    model.eval()
    text="小明在一号位置。小红在二号位置。"
    ids=torch.tensor([tok.encode(text,add_eos=True)],dtype=torch.long,device=device)
    with torch.no_grad(): logits,state=model(ids)
    return {"status":"DRY_RUN","attempted_steps":int(steps),"optimizer_steps_executed":0,"vocab_size":tok.vocab_size,"params":model.n_params(),"logits_shape":list(logits.shape),"device":str(device),"loss":None}
