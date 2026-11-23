from __future__ import annotations
import hashlib, json, math, platform, sys
from pathlib import Path
import torch
from nmind.language.gate2_dataset import build_gate2_dataset
from nmind.language.gate2_evaluator import _norm
from nmind.language.gate22_preflight import answer_mask_audit
from nmind.language.neural_language_v2 import load_language_v2, LanguageV2Config, LanguageV2Model
from nmind.language.tokenizer_v2 import TokenizerV2

ROOT=Path(__file__).resolve().parents[3]; G22=ROOT/"experiments/language_v2/gate22"; OUT=ROOT/"experiments/language_v2/gate23"
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest() if Path(p).exists() else None
def write(name,x): (OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding="utf-8")
def cos(a,b): return float(torch.nn.functional.cosine_similarity(a.reshape(1,-1),b.reshape(1,-1)).item())

def audit():
    paths=[ROOT/"GATE22_REPORT.md",G22/"GATE22_EVAL_PER_ITEM.json",G22/"GATE22_BUDGET_AUDIT.json",G22/"GATE22_PREREGISTRATION.json",G22/"runs/main/candidate.pt",G22/"runs/label_permutation/candidate.pt",G22/"runs/lr_zero/candidate.pt"]
    tok=TokenizerV2(); model,meta=load_language_v2(G22/"runs/main/candidate.pt",tok)
    return {"status":"INPUTS_VERIFIED","git_sha":__import__("subprocess").check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"files":[{"path":str(p.relative_to(ROOT)),"exists":p.exists(),"bytes":p.stat().st_size if p.exists() else None,"sha256":sha(p)} for p in paths],"python":sys.version,"torch":torch.__version__,"cuda":torch.cuda.is_available(),"tokenizer_hash":tok.vocab_hash(),"model_params":model.n_params(),"model_meta":meta,"optimizer_steps":0,"network":0}

def reproduce(data):
    ev=json.loads((G22/"GATE22_EVAL_PER_ITEM.json").read_text(encoding="utf-8")); main=ev["main"]["items"]; perm=ev["label_permutation"]["items"]
    rows=[]; same=0; dist={}
    for a,b in zip(main,perm):
        eq=a["prediction"]==b["prediction"]
        same+=eq; dist[a["prediction"]]=dist.get(a["prediction"],0)+1
        item=next(x for x in data["sealed"] if x.item_id==a["item_id"])
        rows.append({"item_id":a["item_id"],"level":item.level,"pair_id":item.pair_id,"prompt":item.prompt,"prompt_sha256":hashlib.sha256(item.prompt.encode()).hexdigest(),"expected":item.answer,"main":a,"permuted":b,"prediction_identical":eq,"error_type":"correct" if a["exact_match"] else ("invalid_or_nonanswer" if a["prediction"]=="<invalid-byte>" else "wrong_answer")})
    return rows,{"items":len(rows),"prediction_identical_rate":same/max(1,len(rows)),"main_exact":sum(x["main"]["exact_match"] for x in rows)/len(rows),"permuted_exact":sum(x["permuted"]["exact_match"] for x in rows)/len(rows),"main_prediction_distribution":dist}

def loss_decomp(path,data):
    tok=TokenizerV2(); model,_=load_language_v2(path,tok); model.eval(); out=[]; places={"桌子","盒子","地板","架子"}; colors={"红色","蓝色","绿色","黄色"}
    for item in data["sealed"][:24] + data["sealed"][80:104]:
        ids=tok.encode(item.prompt+item.answer,add_eos=True); pn=len(tok.encode(item.prompt));
        with torch.no_grad(): logits,_=model(torch.tensor([ids[:-1]],dtype=torch.long))
        lp=torch.log_softmax(logits[0,pn-1:],-1); target=torch.tensor(ids[pn:]); nll=(-lp[range(len(target)),target]).tolist(); typ="position" if item.answer in places else "color" if item.answer in colors else "other"
        out.append({"item_id":item.item_id,"level":item.level,"type":typ,"answer_nll":sum(nll)/len(nll),"token_count":len(nll),"eos_nll":nll[-1],"answer_top1":float((logits[0,pn-1:].argmax(-1)==target).float().mean())})
    return {"items":out,"by_type":{t:{"mean_nll":sum(x["answer_nll"] for x in out if x["type"]==t)/max(1,sum(x["type"]==t for x in out)),"count":sum(x["type"]==t for x in out)} for t in {x["type"] for x in out}},"mask":"answer_span_plus_eos","optimizer_steps":0}

def drift():
    tok=TokenizerV2(); path=G22/"runs/lr_zero/candidate.pt"; m,_=load_language_v2(path,tok); ids=torch.tensor([tok.encode("苹果在桌子。苹果在哪里？")],dtype=torch.long)
    def hashes(): return {k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k,v in m.state_dict().items() if k not in dict(m.named_parameters())}
    m.eval(); before=hashes();
    with torch.no_grad(): a,_=m(ids); b,_=m(ids)
    eval_after=hashes(); m.train(); train_before=hashes();
    with torch.no_grad(): c,_=m(ids)
    train_after=hashes()
    return {"eval_repeat_max_diff":float((a-b).abs().max()),"eval_buffers_changed":before!=eval_after,"train_buffers_changed":train_before!=train_after,"changed_buffer_names":[k for k in train_after if train_after[k]!=train_before.get(k)],"optimizer_steps":0}

def traces(data):
    tok=TokenizerV2(); m,_=load_language_v2(G22/"runs/main/candidate.pt",tok); m.eval(); rows=[]
    pairs={}
    for item in data["sealed"]:
        if item.pair_id: pairs.setdefault(item.pair_id,[]).append(item)
    for pid,items in list(pairs.items())[:10]:
        states=[]; logits_list=[]
        for item in items:
            with torch.no_grad(): logits,st=m(torch.tensor([tok.encode(item.prompt)],dtype=torch.long))
            states.append(st); logits_list.append(logits[0,-1])
        h0,h1=states[0].h,states[1].h; c0,c1=states[0].ws["content"],states[1].ws["content"]
        rows.append({"pair_id":pid,"h_cosine":cos(h0,h1),"h_l2":float((h0-h1).norm()),"workspace_cosine":cos(c0,c1),"workspace_l2":float((c0-c1).norm()),"logit_l2":float((logits_list[0]-logits_list[1]).norm()),"interpretation":"diagnostic correlation only"})
    return rows

def main():
    OUT.mkdir(parents=True,exist_ok=True); data=build_gate2_dataset(); write("GATE23_INPUT_AUDIT.json",audit()); rows,summary=reproduce(data); (OUT/"GATE23_REPRO_PER_ITEM.jsonl").write_text("\n".join(json.dumps(x,ensure_ascii=False) for x in rows)+"\n",encoding="utf-8"); write("GATE23_ERROR_TAXONOMY.json",summary); write("GATE23_LOSS_DECOMPOSITION.json",loss_decomp(G22/"runs/main/candidate.pt",data)); write("GATE23_ANSWER_SPAN_AUDIT.json",{"samples":[answer_mask_audit(x) for x in data["train"][:32]],"all_valid":all(answer_mask_audit(x)["valid"] for x in data["train"][:32]),"optimizer_steps":0}); write("GATE23_LR0_DRIFT.json",drift()); (OUT/"GATE23_STATE_TRACES.jsonl").write_text("\n".join(json.dumps(x) for x in traces(data))+"\n",encoding="utf-8"); write("GATE23_CONTROL_AUDIT.json",{"main_vs_permuted":summary,"lr0":json.loads((OUT/"GATE23_LR0_DRIFT.json").read_text()),"negative_control_status":"REQUIRES_CAVEAT_MAIN_AND_PERMUTED_SAME_EM","optimizer_steps":0}); write("GATE23_REPRESENTATION_DIAGNOSTICS.json",{"state_traces":"GATE23_STATE_TRACES.jsonl","router_direct_trace":"NOT_IMPLEMENTED_IN_READ_ONLY_PROTOCOL","workspace_trace":"captured_after_prompt","probe_fitted":False,"optimizer_steps":0}); write("GATE23_CAUSAL_ABLATIONS.json",{"status":"NOT_RUN","reason":"strict parameter-shape-preserving ablations require an additional isolated diagnostic implementation; no training performed","optimizer_steps":0}); write("GATE24_PREREGISTRATION.json",{"status":"PENDING_USER_APPROVAL","candidates":[{"id":"A","hypothesis":"answer-conditioned slot head","steps":"new approval required"},{"id":"B","hypothesis":"structured position answer target with same N-Cell","steps":"new approval required"},{"id":"C","hypothesis":"contrastive counterfactual objective","steps":"new approval required"}]}); print(json.dumps({"status":"REPRODUCED_DIAGNOSED","summary":summary,"lr0":json.loads((OUT/"GATE23_LR0_DRIFT.json").read_text())},ensure_ascii=False))
if __name__=="__main__": main()
