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

def balanced_rows(items: list[Gate2Item]) -> list[Gate2Item]:
    """Interleave levels so a bounded run cannot consume only the first level."""
    groups = {}
    for item in items: groups.setdefault(item.level, []).append(item)
    out=[]
    for i in range(max(map(len, groups.values()))):
        for level in sorted(groups):
            if i < len(groups[level]): out.append(groups[level][i])
    return out

def train_arm(items: list[Gate2Item], out: str | Path, *, steps: int, seed: int, lr: float, permute: bool=False) -> dict:
    torch.manual_seed(seed); tok=TokenizerV2(); model=LanguageV2Model(LanguageV2Config(d_model=24,n_blocks=2,block_hidden=24,workspace_dim=12,workspace_slots=2)); model.train()
    before=_state_hash(model); opt=torch.optim.AdamW(model.parameters(), lr=lr); out=Path(out); out.mkdir(parents=True,exist_ok=True)
    rows=balanced_rows(list(items)); rows=rows[:max(1, min(len(rows), 512))]
    if_answer=[x.answer for x in rows]
    if permute:
        g=torch.Generator().manual_seed(seed+77); order=torch.randperm(len(if_answer),generator=g).tolist(); if_answer=[if_answer[i] for i in order]
    losses=[]; answer_losses=[]; start=time.perf_counter(); actual=0; tokens=0
    for step in range(steps):
        item=rows[step % len(rows)]; text=item.prompt + if_answer[step % len(if_answer)]
        ids=torch.tensor([tok.encode(text, add_eos=True)],dtype=torch.long); inp, target=ids[:,:-1],ids[:,1:]
        prompt_len=len(tok.encode(item.prompt)); answer_start=max(0,prompt_len-1)
        opt.zero_grad(set_to_none=True); logits,_=model(inp)
        # Only answer bytes (and EOS) are supervised. Prompt remains context.
        answer_logits=logits[:,answer_start:,:]; answer_target=target[:,answer_start:]
        loss=F.cross_entropy(answer_logits.reshape(-1,answer_logits.size(-1)),answer_target.reshape(-1))
        if not torch.isfinite(loss): raise FloatingPointError("non-finite loss")
        loss.backward(); opt.step()
        actual += 1; tokens += int(answer_target.numel()); losses.append(float(loss.detach())); answer_losses.append(float(loss.detach()))
    after=_state_hash(model); ckpt=out/"candidate.pt"; save_language_v2(ckpt,model,tok,steps=actual,seed=seed,manifest_hash="gate2")
    return {"arm":out.name,"seed":seed,"requested_steps":steps,"optimizer_steps_executed":actual,"lr":lr,"permute":permute,"loss_first":losses[0],"loss_last":losses[-1],"answer_loss_first":answer_losses[0],"answer_loss_last":answer_losses[-1],"tokens":tokens,"rows_used":len(rows),"levels_used":sorted({x.level for x in rows}),"loss_mask":"answer_span_plus_eos","wall_s":time.perf_counter()-start,"param_sha_before":before,"param_sha_after":after,"parameters_changed":before!=after,"checkpoint":str(ckpt),"device":"cuda" if torch.cuda.is_available() else "cpu"}
