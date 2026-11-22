from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import hashlib, json, time, torch
from .neural_dialogue import NMindLM, LMConfig
from .tokenizer_v2 import TokenizerV2

@dataclass
class LanguageV2Config:
    vocab_size: int = 260
    d_model: int = 64
    n_blocks: int = 4
    block_hidden: int = 64
    workspace_dim: int = 32
    workspace_slots: int = 4
    top_k: int = 2
    def to_lm(self): return LMConfig(self.vocab_size,self.d_model,self.n_blocks,self.block_hidden,self.workspace_dim,self.workspace_slots,self.top_k)

class LanguageV2Model(NMindLM):
    """V2 envelope around the existing N-Cell/Router/Workspace sequence core."""
    format = "nmind-language-v2/1"
    def __init__(self, cfg: LanguageV2Config): super().__init__(cfg.to_lm()); self.v2_config=cfg

def save_language_v2(path: str | Path, model: LanguageV2Model, tokenizer: TokenizerV2, *, steps=0, seed=0, manifest_hash="") -> None:
    blob={"format":LanguageV2Model.format,"config":model.v2_config.__dict__,"tokenizer_hash":tokenizer.vocab_hash(),"steps":steps,"seed":seed,"manifest_hash":manifest_hash,"state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},"saved_at":time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); torch.save(blob,p)

def load_language_v2(path: str | Path, tokenizer: TokenizerV2):
    blob=torch.load(Path(path),map_location="cpu",weights_only=True)
    if blob.get("format") != LanguageV2Model.format or blob.get("tokenizer_hash") != tokenizer.vocab_hash(): raise ValueError("Language V2 checkpoint/tokenizer mismatch")
    model=LanguageV2Model(LanguageV2Config(**blob["config"])); model.load_state_dict(blob["state_dict"],strict=True); model.eval(); return model, {k:blob.get(k) for k in ("steps","seed","manifest_hash","saved_at")}
