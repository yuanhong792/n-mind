from __future__ import annotations
import hashlib, json, random, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
import torch
from nmind.language.tokenizer_v2 import ByteBPETokenizer
from nmind.language.neural_language_v2 import LanguageV2Config, LanguageV2Model, load_language_v2, save_language_v2
from nmind.language.bpe_decoder import effective_token_ids, generate_bpe

ROOT=Path(__file__).resolve().parents[3]; SRC=ROOT/"experiments/language_v1"; OUT=SRC/"v15"; PARENT=SRC/"candidate_v14_main.pt"
LEGACY=[("用户：你好\n助手：","你好！"),("用户：你叫什么？\n助手：","我叫N-MIND。"),("用户：谢谢\n助手：","不客气！"),("用户：小明有红色苹果\n助手：","小明的苹果是红色的。"),("用户：小红有绿色苹果\n助手：","小红的苹果是绿色的。"),("用户：小刚有蓝色苹果\n助手：","小刚的苹果是蓝色的。"),("用户：小丽有黄色苹果\n助手：","小丽的苹果是黄色的。"),("用户：我叫小明\n助手：","你好，小明！"),("用户：我叫什么？\n助手：","你叫小明。")]
TRAIN=[("用户：小明有红色苹果，小红有绿色苹果。\n用户：小明的苹果是什么颜色？\n助手：","红色"),("用户：小明有红色苹果，小红有绿色苹果。\n用户：小红的苹果是什么颜色？\n助手：","绿色"),("用户：小明有绿色苹果，小红有红色苹果。\n用户：小明的苹果是什么颜色？\n助手：","绿色"),("用户：小明有绿色苹果，小红有红色苹果。\n用户：小红的苹果是什么颜色？\n助手：","红色")]
SEALED=[("用户：小刚有蓝色书包，小丽有黄色书包。\n用户：小丽的书包是什么颜色？\n助手：","黄色"),("用户：小刚有黄色杯子，小丽有蓝色杯子。\n用户：小刚的杯子是什么颜色？\n助手：","黄色"),("用户：小丽有绿色苹果，小明有蓝色苹果。\n用户：小明的苹果是什么颜色？\n助手：","蓝色"),("用户：小红有黄色书包，小刚有绿色书包。\n用户：小红的书包是什么颜色？\n助手：","黄色")]
COUNTER=[("用户：小明有红色苹果，小红有绿色苹果。\n用户：小明的苹果是什么颜色？\n助手：","红色"),("用户：小明有红色苹果，小红有绿色苹果。\n用户：小红的苹果是什么颜色？\n助手：","绿色"),("用户：小明有绿色苹果，小红有红色苹果。\n用户：小明的苹果是什么颜色？\n助手：","绿色"),("用户：小明有绿色苹果，小红有红色苹果。\n用户：小红的苹果是什么颜色？\n助手：","红色")]
def sha_model(m):
 h=hashlib.sha256()
 for n,v in sorted(m.state_dict().items()): h.update(n.encode()); h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def sha_file(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def train_arm(parent,tok,query,seed,name):
 torch.manual_seed(seed); random.seed(seed); cfg=LanguageV2Config(**{**parent.v2_config.__dict__,"query_readout":query}); m=LanguageV2Model(cfg)
 missing,unexpected=m.load_state_dict(parent.state_dict(),strict=False) if query else (m.load_state_dict(parent.state_dict(),strict=True),[])
 if query and (set(unexpected) or set(missing)!={"query_readout_proj.weight","query_readout_proj.bias"}): raise RuntimeError((missing,unexpected))
 m.train(); m.use_router_load_update=False; opt=torch.optim.AdamW(m.parameters(),lr=.0005); valid=torch.tensor(effective_token_ids(tok)); lookup=torch.full((tok.vocab_size,),-1,dtype=torch.long); lookup[valid]=torch.arange(valid.numel()); losses=[]; route_before=m.router.load_bias.detach().clone(); t=time.perf_counter()
 for step in range(256):
  batch=[TRAIN[(step*2+i)%len(TRAIN)] for i in range(2)]; seq=[tok.encode(p+a,add_eos=True) for p,a in batch]; n=max(map(len,seq)); x=torch.zeros((2,n),dtype=torch.long); mask=torch.zeros((2,n-1),dtype=torch.bool)
  for i,(p,a) in enumerate(batch):
   x[i,:len(seq[i])]=torch.tensor(seq[i]); start=len(tok.encode(p))-1; mask[i,start:len(seq[i])-1]=True
  logits,_=m(x[:,:-1]); z=logits.index_select(-1,valid); target=x[:,1:]; loss=torch.nn.functional.cross_entropy(z[mask],lookup[target[mask]])
  if not torch.isfinite(loss): raise RuntimeError(f"non-finite loss at {step}")
  loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step(); opt.zero_grad(set_to_none=True); losses.append(float(loss.detach()))
 m.eval(); m.use_router_load_update=False
 return m,{"name":name,"steps":len(losses),"loss_first":losses[0],"loss_last":losses[-1],"losses":losses,"wall_seconds":time.perf_counter()-t,"param_sha256":sha_model(m),"param_delta_l1":float(sum((a.detach()-b.detach()).abs().sum() for (a,b) in zip(parent.parameters(),m.parameters()))),"router_load_bias_delta":float((m.router.load_bias.detach()-route_before).abs().sum())}
def score(m,tok,p,a):
 ids=tok.encode(p); aids=tok.encode(a); valid=torch.tensor(effective_token_ids(tok)); vals=[]
 with torch.inference_mode():
  logits,st=m(torch.tensor([ids]))
  for tid in aids:
   lp=torch.log_softmax(logits[0,-1].index_select(0,valid),-1); pos=int((valid==tid).nonzero(as_tuple=True)[0][0]); vals.append(float(lp[pos])); logits,st=m(torch.tensor([[tid]]),st)
 return {"token_ids":aids,"token_logprobs":vals,"logprob_sum":sum(vals),"mean_logprob":sum(vals)/len(vals)}
def evaluate(m,tok,rows,label):
 out=[]
 for p,a in rows:
  scores={c:score(m,tok,p,c) for c in ("红色","绿色","蓝色","黄色")}; pred=max(scores,key=lambda c:scores[c]["mean_logprob"]); g=generate_bpe(m,tok,p,max_new_tokens=16)
  out.append({"arm":label,"prompt":p,"expected":a,"predicted":pred,"correct":pred==a,"generated":g.text,"token_ids":g.token_ids,"finish_reason":g.finish_reason,"invalid_candidate_ids":g.invalid_candidate_ids,"scores":scores})
 return out
def main():
 torch.set_num_threads(1); tok=ByteBPETokenizer.train([p+a for p,a in LEGACY],vocab_size=4096); parent,meta=load_language_v2(PARENT,tok); parent.use_router_load_update=False; parent.eval(); parent_sha=sha_model(parent); start=time.perf_counter()
 base,rb=train_arm(parent,tok,False,20261150,"baseline"); var,rv=train_arm(parent,tok,True,20261151,"query_attention"); total=time.perf_counter()-start
 rows=[]
 for label,m in (("parent",parent),("baseline",base),("query_attention",var)):
  rows += evaluate(m,tok,SEALED,label)+evaluate(m,tok,COUNTER,label)
 OUT.mkdir(parents=True,exist_ok=True); save_language_v2(OUT/"candidate_v15_baseline.pt",base,tok,steps=256,seed=20261150,manifest_hash="v15-preregistered"); save_language_v2(OUT/"candidate_v15_query_attention.pt",var,tok,steps=256,seed=20261151,manifest_hash="v15-preregistered")
 result={"status":"COMPLETED_AUTHORIZED_TRAINING","parent_sha256":sha_file(PARENT),"parent_model_sha256":parent_sha,"tokenizer_hash":tok.vocab_hash(),"total_optimizer_steps":rb["steps"]+rv["steps"],"wall_seconds":total,"stable_modified":False,"network_requests":0,"external_models":0,"arms":[rb,rv],"results_summary":{"parent_accuracy":sum(x["correct"] for x in rows if x["arm"]=="parent")/8,"baseline_accuracy":sum(x["correct"] for x in rows if x["arm"]=="baseline")/8,"variant_accuracy":sum(x["correct"] for x in rows if x["arm"]=="query_attention")/8}}
 (OUT/"V15_TRAINING_RESULTS.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8"); (OUT/"V15_FREE_GENERATION.jsonl").write_text("\n".join(json.dumps(x,ensure_ascii=False) for x in rows)+"\n",encoding="utf-8")
 comp={"status":"COMPLETED","rows":rows,"fixed_red_baseline_accuracy":sum(x["expected"]=="红色" for x in rows)/len(rows),"recent_attribute_baseline":"NOT_APPLICABLE_FOR_MULTIENTITY","arms":{"parent":{},"baseline":{},"query_attention":{}}}
 for arm in comp["arms"]: rr=[x for x in rows if x["arm"]==arm]; comp["arms"][arm]={"accuracy":sum(x["correct"] for x in rr)/len(rr),"fixed_red_rate":sum(x["predicted"]=="红色" for x in rr)/len(rr),"eos_rate":sum(x["finish_reason"]=="eos" for x in rr)/len(rr),"utf8_errors":sum(x["finish_reason"]=="byte_decode_error" for x in rr)}
 (OUT/"V15_AB_COMPARISON.json").write_text(json.dumps(comp,ensure_ascii=False,indent=2),encoding="utf-8"); (OUT/"V15_COUNTERFACTUAL_EVAL.json").write_text(json.dumps({"status":"COMPLETED","rows":[x for x in rows if x["prompt"] in [p for p,a in COUNTER]]},ensure_ascii=False,indent=2),encoding="utf-8")
 (OUT/"V15_RETENTION_AUDIT.json").write_text(json.dumps({"status":"RETENTION_INCONCLUSIVE","reason":"No pre-registered V1.4 sealed old-skill set was available for same-protocol comparison.","parent_and_candidates_evaluated":False},ensure_ascii=False,indent=2),encoding="utf-8")
 (OUT/"V15_PARAMETER_AUDIT.json").write_text(json.dumps({"parent_params":parent.n_params(),"baseline_params":base.n_params(),"variant_params":var.n_params(),"baseline":rb,"variant":rv,"stable_modified":False,"stable_sha256":sha_file(ROOT/"results/bot_r1/stable_model.pt")},ensure_ascii=False,indent=2),encoding="utf-8")
 (OUT/"V15_TRAINING_REPORT.md").write_text(f"# V1.5 Training Report\n\n状态：COMPLETED_AUTHORIZED_TRAINING。两组各执行 {rb['steps']} 步，总计 {rb['steps']+rv['steps']} 步；总耗时 {total:.2f}s。\n\nBaseline loss: {rb['loss_first']:.6f} -> {rb['loss_last']:.6f}\n\nQuery-Attention loss: {rv['loss_first']:.6f} -> {rv['loss_last']:.6f}\n\n自由生成原文见 `V15_FREE_GENERATION.jsonl`。旧技能保持：`RETENTION_INCONCLUSIVE`。Stable 未修改。\n",encoding="utf-8")
 print(json.dumps(result["results_summary"],ensure_ascii=False))
if __name__=="__main__": main()
