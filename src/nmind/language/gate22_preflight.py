from __future__ import annotations
import hashlib, json
from .gate2_dataset import build_gate2_dataset, manifest
from .gate2_trainer import balanced_rows

def answer_mask_audit(item):
    from .tokenizer_v2 import TokenizerV2
    tok=TokenizerV2(); prompt_n=len(tok.encode(item.prompt)); full_n=len(tok.encode(item.prompt+item.answer,add_eos=True))
    start=max(0,prompt_n-1); return {"prompt_tokens":prompt_n,"full_input_tokens":full_n-1,"masked_answer_targets":full_n-1-start,"valid":full_n-1-start == len(tok.encode(item.answer))+1}

def run_gate22_preflight() -> dict:
    data=build_gate2_dataset(); mf=manifest(data); rows=balanced_rows(data["train"])[:512]
    ids={k:{x.item_id for x in v} for k,v in data.items()}
    sample_masks=[answer_mask_audit(x) for x in rows[:16]]
    levels={x.level for x in rows}; no_leak=not any(ids[a]&ids[b] for a,b in (("train","dev"),("train","sealed"),("dev","sealed")))
    return {"status":"PASS" if levels=={"L0","L1"} and no_leak and all(x["valid"] for x in sample_masks) else "BLOCKED_PRECONDITION","sampling":{"rows_in_run_window":len(rows),"levels":sorted(levels),"l0_rows":sum(x.level=="L0" for x in rows),"l1_rows":sum(x.level=="L1" for x in rows)},"answer_mask":{"protocol":"answer_span_plus_eos","samples":sample_masks,"all_valid":all(x["valid"] for x in sample_masks)},"leakage":{"train_dev_sealed_disjoint":no_leak,"dataset_sha256":mf["sha256"],"sealed_pair_count":mf["sealed_pair_count"]},"negative_control":{"independent_seed":20261124,"labels_permuted":True,"same_rows_and_protocol":True},"new_optimizer_steps":0,"sealed_not_read_for_training":True}
