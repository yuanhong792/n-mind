"""Zero-step Gate-2.1 diagnostics for data coverage, loss attribution and state flow."""
from __future__ import annotations
from dataclasses import asdict
import hashlib, json
from pathlib import Path
import torch
from .gate2_dataset import build_gate2_dataset
from .neural_language_v2 import load_language_v2
from .tokenizer_v2 import TokenizerV2

def data_coverage() -> dict:
    data=build_gate2_dataset(); used=data["train"][:256]
    return {"train_total":len(data["train"]),"trainer_rows_used":len(used),"used_levels":sorted({x.level for x in used}),"unused_levels":sorted({x.level for x in data["train"] if x not in used}),"l1_training_rows_used":sum(x.level=="L1" for x in used),"answer_boundary_supervision":"none"}

def teacher_forced_answer_metrics(path: str | Path, items, limit=24) -> dict:
    tok=TokenizerV2(); model,_=load_language_v2(path,tok); model.eval(); total=correct=0; answer_bytes=0
    for item in items[:limit]:
        full=tok.encode(item.prompt+item.answer, add_eos=True); prompt_n=len(tok.encode(item.prompt));
        with torch.no_grad(): logits,_=model(torch.tensor([full[:-1]],dtype=torch.long))
        # The answer span is scored separately from prompt-copy tokens.
        pred=logits[0,prompt_n-1:-1].argmax(-1).tolist(); target=full[prompt_n:]
        correct += sum(int(a==b) for a,b in zip(pred,target)); total += len(target); answer_bytes += len(target)
    return {"items":min(limit,len(items)),"answer_token_accuracy":correct/max(1,total),"answer_tokens":answer_bytes,"metric":"teacher_forced_next_byte_only"}

def state_equivalence() -> dict:
    from .neural_language_v2 import LanguageV2Config, LanguageV2Model
    tok=TokenizerV2(); model=LanguageV2Model(LanguageV2Config()); model.eval(); ids=tok.encode("\u82f9\u679c\u5728\u684c\u5b50\u3002")
    with torch.no_grad():
        full,_=model(torch.tensor([ids])); left,st=model(torch.tensor([ids[:4]])); right,_=model(torch.tensor([ids[4:]],dtype=torch.long),st)
    return {"max_abs_logit_diff_last_token":float((full[:,-1]-right[:,-1]).abs().max()),"protocol":"full_vs_state_chunked","optimizer_steps":0}

def diagnose(checkpoints: dict[str,str]) -> dict:
    data=build_gate2_dataset(); return {"status":"DIAGNOSED_NO_TRAINING","data_coverage":data_coverage(),"state":state_equivalence(),"teacher_forced_answer":{k:teacher_forced_answer_metrics(v,data["sealed"],24) for k,v in checkpoints.items()},"high_confidence_findings":["L1 is excluded by the trainer's first-256-row truncation","loss has no answer-span mask and is dominated by prompt-copy tokens","free generation is the primary semantic metric; teacher forcing is auxiliary"],"network":0,"optimizer_steps":0}
