"""Bounded candidate trainer. It is only called by the explicit Gate-2 runner."""
from __future__ import annotations
import hashlib, json, time
from pathlib import Path
import torch
from torch.nn import functional as F
from .gate2_dataset import Gate2Item
from .neural_language_v2 import LanguageV2Config, LanguageV2Model, save_language_v2
from .tokenizer_v2 import TokenizerV2

def _state_hash(model):
    return hashlib.sha256(b"".join(v.detach().cpu().numpy().tobytes() for v in model.parameters())).hexdigest()

def train_arm(items: list[Gate2Item], out: str | Path, *, steps: int, seed: int, lr: float, permute: bool=False) -> dict:
    torch.manual_seed(seed); tok=TokenizerV2(); model=LanguageV2Model(LanguageV2Config(d_model=24,n_blocks=2,block_hidden=24,workspace_dim=12,workspace_slots=2)); model.train()
    before=_state_hash(model); opt=torch.optim.AdamW(model.parameters(), lr=lr); out=Path(out); out.mkdir(parents=True,exist_ok=True)
    rows=list(items[:max(1, min(len(items), 256))]); if_answer=[x.answer for x in rows]
    if permute:
        g=torch.Generator().manual_seed(seed+77); order=torch.randperm(len(if_answer),generator=g).tolist(); if_answer=[if_answer[i] for i in order]
    losses=[]; start=time.perf_counter(); actual=0; tokens=0
    for step in range(steps):
        item=rows[step % len(rows)]; text=item.prompt + if_answer[step % len(if_answer)]
        ids=torch.tensor([tok.encode(text, add_eos=True)],dtype=torch.long); inp, target=ids[:,:-1],ids[:,1:]
        opt.zero_grad(set_to_none=True); logits,_=model(inp); loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),target.reshape(-1))
        if not torch.isfinite(loss): raise FloatingPointError("non-finite loss")
        loss.backward(); opt.step()
        actual += 1; tokens += int(target.numel()); losses.append(float(loss.detach()))
    after=_state_hash(model); ckpt=out/"candidate.pt"; save_language_v2(ckpt,model,tok,steps=actual,seed=seed,manifest_hash="gate2")
    return {"arm":out.name,"seed":seed,"requested_steps":steps,"optimizer_steps_executed":actual,"lr":lr,"permute":permute,"loss_first":losses[0],"loss_last":losses[-1],"tokens":tokens,"wall_s":time.perf_counter()-start,"param_sha_before":before,"param_sha_after":after,"parameters_changed":before!=after,"checkpoint":str(ckpt),"device":"cuda" if torch.cuda.is_available() else "cpu"}
