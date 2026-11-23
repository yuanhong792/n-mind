from __future__ import annotations
import json,sys,importlib.util
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[3]; OUT=ROOT/"experiments/language_v2/minlearn"; OUT.mkdir(parents=True,exist_ok=True); sys.path.insert(0,str(ROOT/"src"))
from nmind.language.neural_language_v2 import LanguageV2Config,LanguageV2Model
from nmind.language.minlearn_diagnostics import gradient_audit,SelectiveBlockRead,FullBlockRead,TinyGRUBaseline,candidate_registry
from nmind.language.tokenizer_v2 import TokenizerV2
tok=TokenizerV2(); torch.manual_seed(20261130); m=LanguageV2Model(LanguageV2Config(vocab_size=260,d_model=24,n_blocks=2,block_hidden=24,workspace_dim=12,workspace_slots=2))
texts=["小明在一号位置。小刚在二号位置。问题：小明在哪里？","小明在二号位置。小刚在一号位置。问题：小明在哪里？"]*8
ids=[tok.encode(x,add_eos=True) for x in texts]; n=max(map(len,ids)); x=torch.tensor([a+[0]*(n-len(a)) for a in ids]); y=x.clone(); audit=gradient_audit(m,x,y)
summary={"status":"ZERO_TRAINING_DIAGNOSTIC","optimizer_steps":0,"sample_count":16,"tokenizer_reversible":all(tok.decode(tok.encode(t))==t for t in texts),"grad_norms":audit.grad_norms,"loss":audit.loss,"parameter_changed":audit.parameter_changed,"router_grad_norm":sum(v for k,v in audit.grad_norms.items() if "router" in k),"head_grad_norm":sum(v for k,v in audit.grad_norms.items() if "head" in k),"block_to_ws_grad_norm":sum(v for k,v in audit.grad_norms.items() if "block_to_ws" in k),"workspace_grad_norm":sum(v for k,v in audit.grad_norms.items() if "workspace" in k),"candidate_registry":candidate_registry(),"baselines":{"original":"NCell sparse + h.mean","selective_block_workspace":"implemented shape test","full_block_workspace":"implemented shape test","tiny_gru":"implemented shape test"}}
(OUT/"MINLEARN_ZERO_STEP_AUDIT.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
(OUT/"MINLEARN_PREREGISTRATION.json").write_text(json.dumps({"status":"PENDING_USER_APPROVAL","optimizer_steps":0,"sample_count":16,"future_arms":[{"id":"original","steps":128},{"id":"selective_block_workspace","steps":128},{"id":"full_block_workspace","steps":128},{"id":"tiny_gru","steps":128}],"metrics":["train_full_answer_fit","unseen_composition","counterfactual_follow_rate","per_chinese_character_accuracy","eos_accuracy","router_grad_norm","parameter_update_audit"],"major_variable":"one readout/architecture change per arm","no_gate24_budget_reuse":True},ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps({"optimizer_steps":0,"loss":audit.loss,"router_grad_norm":summary["router_grad_norm"],"head_grad_norm":summary["head_grad_norm"]},ensure_ascii=False))

