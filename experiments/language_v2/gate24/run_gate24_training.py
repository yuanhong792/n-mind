from __future__ import annotations
import json, hashlib, time, sys, importlib.util
from pathlib import Path
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[3]; OUT=ROOT/"experiments/language_v2/gate24"; sys.path.insert(0,str(ROOT/"src"))
spec=importlib.util.spec_from_file_location("gate24_dataset",ROOT/"src/nmind/language/gate24_dataset.py"); mod=importlib.util.module_from_spec(spec); sys.modules["gate24_dataset"]=mod; spec.loader.exec_module(mod)
from nmind.language.neural_language_v2 import LanguageV2Config,LanguageV2Model,save_language_v2
from nmind.language.tokenizer_v2 import TokenizerV2
MAX_SECONDS=5400; START=time.perf_counter()
def state_hash(m):
    return hashlib.sha256(b"".join(v.detach().cpu().numpy().tobytes() for v in m.parameters())).hexdigest()
def make_items():
    d=mod.build_gate24(); perm,meta=mod.stratified_label_permutation(d["train"])
    blind=mod.context_blind(d["train"])
    return d,perm,blind,meta
def train_arm(name,items,steps,seed,lr):
    if time.perf_counter()-START>=MAX_SECONDS: raise RuntimeError("WALL_CLOCK_BUDGET")
    torch.manual_seed(seed); tok=TokenizerV2(); cfg=LanguageV2Config(d_model=24,n_blocks=2,block_hidden=24,workspace_dim=12,workspace_slots=2); model=LanguageV2Model(cfg); model.train()
    before=state_hash(model); opt=torch.optim.AdamW(model.parameters(),lr=lr); losses=[]; tokens=0; t0=time.perf_counter()
    for step in range(steps):
        if time.perf_counter()-START>=MAX_SECONDS: raise RuntimeError("WALL_CLOCK_BUDGET")
        item=items[step%len(items)]; ids=torch.tensor([tok.encode(item.prompt+item.answer,add_eos=True)],dtype=torch.long); inp,target=ids[:,:-1],ids[:,1:]
        start=max(0,len(tok.encode(item.prompt))-1); logits,_=model(inp); al=logits[:,start:,:]; at=target[:,start:]
        loss=F.cross_entropy(al.reshape(-1,al.size(-1)),at.reshape(-1)); opt.zero_grad(set_to_none=True)
        if lr>0: loss.backward(); opt.step()
        losses.append(float(loss.detach())); tokens+=int(at.numel())
    model.eval(); after=state_hash(model); path=OUT/"training_runs"/name; path.mkdir(parents=True,exist_ok=True); ck=path/"candidate.pt"; save_language_v2(ck,model,tok,steps=steps,seed=seed,manifest_hash="gate24")
    return {"arm":name,"steps_requested":steps,"optimizer_steps":steps if lr>0 else 0,"lr":lr,"loss_first":losses[0],"loss_last":losses[-1],"tokens":tokens,"wall_s":time.perf_counter()-t0,"state_sha_before":before,"state_sha_after":after,"parameters_changed":before!=after,"checkpoint":str(ck),"device":"cuda" if torch.cuda.is_available() else "cpu"}
def main():
    d,perm,blind,meta=make_items(); seed=20261124; results=[]
    for name,items,steps,lr in [("A_MAIN_CONTRASTIVE",d["train"],384,1e-3),("B_LABEL_BROKEN",perm,384,1e-3),("C_CONTEXT_BLIND",blind,384,1e-3),("D_ZERO_UPDATE",d["train"],32,0.0)]:
        results.append(train_arm(name,items,steps,seed,lr))
    total_steps=sum(x["optimizer_steps"] for x in results); elapsed=time.perf_counter()-START
    out={"status":"COMPLETED_AUTHORIZED_TRAINING","authorization":"user Gate-2.4 approval","arms":results,"total_optimizer_steps":total_steps,"wall_s":elapsed,"wall_limit_s":MAX_SECONDS,"network_requests":0,"stable_modified":False,"sealed_used_for_training":False,"gate24_status":"PENDING_USER_APPROVAL_FOR_FURTHER_TRAINING"}
    (OUT/"GATE24_TRAINING_RESULTS.json").write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT/"GATE24_TRAINING_BUDGET_AUDIT.json").write_text(json.dumps({"requested_total":1184,"actual_optimizer_steps":total_steps,"wall_s":elapsed,"within_step_budget":total_steps<=1184,"within_wall_budget":elapsed<=MAX_SECONDS,"network_requests":0,"stable_modified":False},ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(out,ensure_ascii=False))
if __name__=="__main__": main()

