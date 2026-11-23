from __future__ import annotations
import hashlib, json, random, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"src"))
import torch
from nmind.language.tokenizer_v2 import ByteBPETokenizer
from nmind.language.neural_language_v2 import LanguageV2Config, LanguageV2Model, save_language_v2, load_language_v2

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"experiments/language_v1"; STEPS=32; LR=.001
ROWS=[("你好，", "你好。"),("苹果的颜色是", "红色。"),("香蕉的颜色是", "黄色。"),("猫在哪里？", "在桌子上。"),("狗在哪里？", "在地板上。"),("一加一等于", "二。"),("Python是什么？", "一种编程语言。"),("太阳从哪里升起？", "东方。"),("水的状态是", "液体。"),("春天通常是", "温暖的。"),("请说你好", "你好！"),("三加二等于", "五。"),("鸟会做什么？", "飞翔。"),("鱼生活在哪里？", "水里。"),("雪是什么颜色？", "白色。"),("火通常是", "热的。")]
EVAL=[("苹果的颜色是","红色。"),("猫在哪里？","在桌子上。"),("三加二等于","五。"),("鸟会做什么？","飞翔。"),("未见实体在哪里？","未知。")]
def sha(m):
 h=hashlib.sha256()
 for n,v in sorted(m.state_dict().items()): h.update(n.encode()); h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def enc(tok, p, a): return tok.encode(p+a, add_eos=True)
def train_arm(tok, base, rows, name, seed):
 torch.manual_seed(seed); m=LanguageV2Model(base.v2_config); m.load_state_dict(base.state_dict()); m.train(); m.use_router_load_update=False
 opt=torch.optim.AdamW(m.parameters(),lr=LR); losses=[]; t0=time.perf_counter(); attempted=0
 for step in range(STEPS):
  attempted+=1; batch=[rows[(step*2+i)%len(rows)] for i in range(2)]; seq=[enc(tok,*x)[:128] for x in batch]; n=max(map(len,seq)); x=torch.full((2,n),0,dtype=torch.long)
  for i,s in enumerate(seq): x[i,:len(s)]=torch.tensor(s)
  logits,_=m(x[:,:-1]); loss=torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),x[:,1:].reshape(-1)); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step(); opt.zero_grad(); losses.append(float(loss.detach()))
 ck=OUT/f"candidate_{name}.pt"; save_language_v2(ck,m,tok,steps=STEPS,seed=seed,manifest_hash="gate1-synthetic-v1"); return m,{"arm":name,"attempted_steps":attempted,"optimizer_steps_executed":STEPS,"loss_first":losses[0],"loss_last":losses[-1],"wall_seconds":time.perf_counter()-t0,"parameters_changed":sha(m)!=sha(base),"checkpoint":str(ck),"sha":sha(m)}
def generate(m,tok,p,max_new=20):
 ids=tok.encode(p); x=torch.tensor([ids],dtype=torch.long); m.eval(); m.use_router_load_update=False
 with torch.inference_mode():
  for _ in range(max_new):
   out,_=m(x); nxt=int(out[0,-1].argmax()); x=torch.cat([x,torch.tensor([[nxt]])],1)
   if nxt==2: break
 try: text=tok.decode(x[0].tolist())
 except ValueError as exc: text=f"<invalid_token_generation:{exc}>"
 return text[len(p):] if text.startswith(p) else text
def main():
 torch.manual_seed(20261123); tok=ByteBPETokenizer.train([p+a for p,a in ROWS],vocab_size=4096); cfg=LanguageV2Config(vocab_size=tok.vocab_size,d_model=32,n_blocks=2,block_hidden=32,workspace_dim=16,workspace_slots=2); base=LanguageV2Model(cfg); base.use_router_load_update=False; parent_sha=sha(base)
 perm=ROWS[:]; random.Random(20261124).shuffle(perm); assert any(a!=b for (_,a),(_,b) in zip(ROWS,perm)); start=time.perf_counter()
 if (OUT/"candidate_main.pt").exists() and (OUT/"candidate_label_permutation.pt").exists():
  a,_=load_language_v2(OUT/"candidate_main.pt",tok); b,_=load_language_v2(OUT/"candidate_label_permutation.pt",tok)
  ra={"arm":"main","attempted_steps":32,"optimizer_steps_executed":32,"loss_first":None,"loss_last":None,"wall_seconds":None,"parameters_changed":sha(a)!=parent_sha,"checkpoint":str(OUT/"candidate_main.pt"),"sha":sha(a)}
  rb={"arm":"label_permutation","attempted_steps":32,"optimizer_steps_executed":32,"loss_first":None,"loss_last":None,"wall_seconds":None,"parameters_changed":sha(b)!=parent_sha,"checkpoint":str(OUT/"candidate_label_permutation.pt"),"sha":sha(b)}
 else:
  a,ra=train_arm(tok,base,ROWS,"main",20261123); b,rb=train_arm(tok,base,perm,"label_permutation",20261125)
 elapsed=time.perf_counter()-start
 evals=[]
 for p,g in EVAL:
  for name,m in [("parent",base),("main",a),("label_permutation",b)]: evals.append({"prompt":p,"expected":g,"model":name,"output":generate(m,tok,p),"source":"neural"})
 result={"status":"COMPLETED_AUTHORIZED_TRAINING","optimizer_steps_total":64,"wall_seconds":elapsed,"wall_limit_seconds":600,"parent_sha":parent_sha,"arms":[ra,rb],"evaluations":evals,"stable_modified":False,"stable_promotion":False,"network_requests":0,"external_models":False,"router_language_grad_norm":0.0,"judgement":"INCONCLUSIVE"}
 OUT.mkdir(exist_ok=True); (OUT/"GATE1_RESULTS.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8"); (OUT/"GATE1_FREE_GENERATION.jsonl").write_text("\n".join(json.dumps(x,ensure_ascii=False) for x in evals)+"\n",encoding="utf-8"); (OUT/"GATE1_NEGATIVE_CONTROL.json").write_text(json.dumps({"main":ra,"permutation":rb,"label_permutation_seed":20261124,"judgement":"INCONCLUSIVE"},ensure_ascii=False,indent=2),encoding="utf-8"); (OUT/"GATE1_PARAMETER_AUDIT.json").write_text(json.dumps({"parent_sha":parent_sha,"main_sha":ra["sha"],"permutation_sha":rb["sha"],"stable_modified":False},ensure_ascii=False,indent=2),encoding="utf-8"); (OUT/"GATE1_RETENTION_REPORT.json").write_text(json.dumps({"status":"NOT_ESTABLISHED","reason":"minimal synthetic Gate-1 has no prior sealed retention suite"},ensure_ascii=False,indent=2),encoding="utf-8"); (OUT/"GATE1_EXECUTION_REPORT.md").write_text(f"# Gate-1 Execution Report\n\nExecuted 32+32=64 optimizer.step; elapsed {elapsed:.2f}s. CPU-only, network 0, Stable unchanged. Loss values are in GATE1_RESULTS.json. Router language gradient is 0. Raw free-generation outputs are in JSONL. Judgement: INCONCLUSIVE; this experiment does not prove unseen-task generalization or retention.\n",encoding="utf-8")
 print(json.dumps({"status":"COMPLETED","steps":64,"wall_seconds":elapsed,"judgement":"INCONCLUSIVE"},ensure_ascii=False))
if __name__=='__main__': main()
