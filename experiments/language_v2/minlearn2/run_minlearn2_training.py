from __future__ import annotations
import json,sys,time,hashlib,importlib.util
from pathlib import Path
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[3]; OUT=ROOT/"experiments/language_v2/minlearn2"; sys.path.insert(0,str(ROOT/"src"))
spec=importlib.util.spec_from_file_location("minlearn2_data",ROOT/"src/nmind/language/minlearn2.py"); data=importlib.util.module_from_spec(spec); sys.modules["minlearn2_data"]=data; spec.loader.exec_module(data)
from nmind.language.neural_language_v2 import LanguageV2Config,LanguageV2Model,save_language_v2
from nmind.language.tokenizer_v2 import TokenizerV2
MAX_WALL=5400; START=time.perf_counter(); SEED=20261130
def sh(m): return hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes() for p in m.parameters())).hexdigest()
def soft_forward(m,ids):
    B,T=ids.shape; st=m.initial_state(B,ids.device); rows=[]; err=torch.full((B,1),.5); unc=torch.full((B,1),.5)
    for t in range(T):
        z=m.tok_emb(ids[:,t]); rd=m.workspace.read(st.ws,z); raw,probs=m.router.scores(z,st.h,rd.value,err,unc)
        active=torch.ones(B,m.cfg.n_blocks); h,g,mask,p=m.blocks(z,st.h,rd.value,active,probs)
        payload=m.block_to_ws(h).view(B,m.cfg.n_blocks,-1); wr=m.workspace.write(st.ws,(payload*probs.unsqueeze(-1)).sum(1,keepdim=True))
        st=type(st)(h=h,ws={"content":wr.state["content"],"occupied":wr.occupied,"utility":wr.utility}); rows.append(m.head(h.mean(1)))
    return torch.stack(rows,1)
def forward(m,ids,soft):
    return soft_forward(m,ids) if soft else m(ids)[0]
def train_arm(name,items,soft,freeze_router,seed):
    torch.manual_seed(seed); tok=TokenizerV2(); m=LanguageV2Model(LanguageV2Config(d_model=24,n_blocks=2,block_hidden=24,workspace_dim=12,workspace_slots=2)); m.train()
    if freeze_router:
        for p in m.router.parameters(): p.requires_grad=False
    before=sh(m); opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-3); losses=[]; t0=time.perf_counter()
    for step in range(128):
        if time.perf_counter()-START>=MAX_WALL: raise RuntimeError("WALL_CLOCK_BUDGET")
        x=items[step%len(items)]; ids=torch.tensor([tok.encode(x.prompt+x.answer,add_eos=True)])
        inp,target=ids[:,:-1],ids[:,1:]; start=max(0,len(tok.encode(x.prompt))-1); logits=forward(m,inp,soft); al=logits[:,start:,:]; at=target[:,start:]
        loss=F.cross_entropy(al.reshape(-1,al.size(-1)),at.reshape(-1)); opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); losses.append(float(loss.detach()))
    m.eval(); path=OUT/"training_runs"/name; path.mkdir(parents=True,exist_ok=True); ck=path/"candidate.pt"; save_language_v2(ck,m,tok,steps=128,seed=seed,manifest_hash="minlearn2")
    return {"arm":name,"steps":128,"soft_router":soft,"router_frozen":freeze_router,"loss_first":losses[0],"loss_last":losses[-1],"wall_s":time.perf_counter()-t0,"state_sha_before":before,"state_sha_after":sh(m),"parameters_changed":before!=sh(m),"checkpoint":str(ck)}
def eval_arm(name,items):
    tok=TokenizerV2(); path=OUT/"training_runs"/name/"candidate.pt"; blob=torch.load(path,map_location="cpu",weights_only=True); m=LanguageV2Model(__import__("nmind.language.neural_language_v2",fromlist=["LanguageV2Config"]).LanguageV2Config(**blob["config"])); m.load_state_dict(blob["state_dict"]); m.eval(); rows=[]; exact=chars=eos=0; totalchars=0
    def safe_decode(v):
        try: return tok.decode(v),False
        except UnicodeDecodeError: return "<invalid_utf8>",True
    for x in items:
        ids=torch.tensor([tok.encode(x.prompt+x.answer,add_eos=True)]); prompt=tok.encode(x.prompt); target=ids[:,len(prompt):]; logits,_=m(ids[:,:-1]); start=max(0,len(prompt)-1); pred=logits[:,start:start+target.shape[1],:].argmax(-1); exact+=int(torch.equal(pred,target)); gold,gold_bad=safe_decode(target[0].tolist()); guess,bad=safe_decode(pred[0].tolist()); chars+=sum(a==b for a,b in zip(gold,guess)); totalchars+=len(gold); eos+=int(pred[0,-1].item()==target[0,-1].item()); rows.append({"item_id":x.item_id,"exact":bool(torch.equal(pred,target)),"gold":gold,"pred":guess,"invalid_utf8":bad})
    return {"arm":name,"n":len(items),"exact":exact/len(items),"unicode_char_accuracy":chars/max(1,totalchars),"eos_accuracy":eos/len(items),"invalid_utf8":sum(r["invalid_utf8"] for r in rows)/len(rows),"rows":rows}
def main():
    train=data.build_train16(); dev=data.build_dev(); sealed=data.build_sealed(); independent=data.independent_labels(train)
    results=[]; results.append(train_arm("A_HARD_TOPK_ORIGINAL",train,False,False,SEED)); results.append(train_arm("B_SOFT_MIXTURE_TRAINABLE",train,True,False,SEED)); results.append(train_arm("C_SOFT_MIXTURE_ROUTER_FROZEN",train,True,True,SEED)); results.append(train_arm("D_SOFT_MIXTURE_LABEL_INDEPENDENT",independent,True,False,SEED))
    ev=[eval_arm(x,train) | {"split":"train"} for x in ("A_HARD_TOPK_ORIGINAL","B_SOFT_MIXTURE_TRAINABLE","C_SOFT_MIXTURE_ROUTER_FROZEN","D_SOFT_MIXTURE_LABEL_INDEPENDENT")]
    ev += [eval_arm(x,dev) | {"split":"dev"} for x in ("A_HARD_TOPK_ORIGINAL","B_SOFT_MIXTURE_TRAINABLE","C_SOFT_MIXTURE_ROUTER_FROZEN","D_SOFT_MIXTURE_LABEL_INDEPENDENT")]
    ev += [eval_arm(x,sealed) | {"split":"sealed"} for x in ("A_HARD_TOPK_ORIGINAL","B_SOFT_MIXTURE_TRAINABLE","C_SOFT_MIXTURE_ROUTER_FROZEN","D_SOFT_MIXTURE_LABEL_INDEPENDENT")]
    out={"schema_version":"minlearn2/1","status":"COMPLETED_AUTHORIZED_TRAINING","arms":results,"evaluations":ev,"actual_optimizer_steps":512,"wall_s":time.perf_counter()-START,"wall_limit_s":MAX_WALL,"network_requests":0,"stable_modified":False,"sealed_used_for_training":False}
    (OUT/"MINLEARN2_TRAINING_RESULTS.json").write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT/"MINLEARN2_EVAL.json").write_text(json.dumps(ev,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"status":out["status"],"steps":512,"wall_s":out["wall_s"],"eval_summary":[{"arm":x["arm"],"split":x["split"],"exact":x["exact"]} for x in ev]},ensure_ascii=False))
if __name__=="__main__": main()

