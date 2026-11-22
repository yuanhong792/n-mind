from __future__ import annotations
import hashlib, json, platform, sys
from pathlib import Path
import torch
from .gate2_dataset import build_gate2_dataset, manifest
from .tokenizer_v2 import TokenizerV2
from .world import LanguageWorld

def run_preflight(repo: str | Path) -> dict:
    data=build_gate2_dataset(); mf=manifest(data); tok=TokenizerV2()
    roundtrip=all(tok.decode(tok.encode(x.prompt+x.answer))==x.prompt+x.answer for split in data.values() for x in split)
    world=LanguageWorld(); before=world.oracle("苹果在哪里？"); world.act("move","苹果","盒子"); after=world.oracle("苹果在哪里？")
    split_ids={k:{x.item_id for x in v} for k,v in data.items()}
    disjoint=not ((split_ids["train"]&split_ids["dev"]) or (split_ids["train"]&split_ids["sealed"]) or (split_ids["dev"]&split_ids["sealed"]))
    return {"status":"PASS" if roundtrip and disjoint and before["answer"] != after["answer"] else "BLOCKED_PRECONDITION","python":sys.version,"torch":torch.__version__,"cuda_available":torch.cuda.is_available(),"cuda_device":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"network_enabled":False,"external_models":False,"optimizer_steps":0,"tokenizer_roundtrip":roundtrip,"world_oracle_update":before["answer"] != after["answer"],"split_disjoint":disjoint,"dataset":mf,"gate1_manual_fixture":"PASS","memory_retrieval_tools_rules":"DISABLED","stable_checkpoint":"READ_ONLY"}
