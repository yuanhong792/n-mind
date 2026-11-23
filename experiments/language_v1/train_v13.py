from __future__ import annotations
import hashlib, json, sys, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src"))
import torch
from nmind.language.tokenizer_v2 import ByteBPETokenizer
from nmind.language.neural_language_v2 import LanguageV2Config, LanguageV2Model, save_language_v2
from nmind.language.bpe_decoder import effective_token_ids
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"experiments/language_v1"; MAX_STEPS=256; LIMIT=1800
TRAIN=[("用户：你好\n助手：","你好！"),("用户：你叫什么？\n助手：","我叫N-MIND。"),("用户：谢谢\n助手：","不客气！"),("用户：小明有红色苹果\n助手：","小明的苹果是红色的。"),("用户：小红有绿色苹果\n助手：","小红的苹果是绿色的。")]
L2=[("用户：小刚有蓝色苹果\n助手：","小刚的苹果是蓝色的。"),("用户：小丽有黄色苹果\n助手：","小丽的苹果是黄色的。")]
L3=[("用户：我叫小明\n助手：","你好，小明！"),("用户：我叫什么？\n助手：","你叫小明。")]
def sha(m):
 h=hashlib.sha256()
 for n,v in sorted(m.state_dict().items()): h.update(n.encode()); h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def answer_loss(m,tok,rows):
 vals=[]; valid=torch.tensor(effective_token_ids(tok)); lookup=torch.full((tok.vocab_size,),-1,dtype=torch.long); lookup[valid]=torch.arange(len(valid))
 with torch.inference_mode():
  for p,a in rows:
   ids=tok.encode(p+a,add_eos=True); x=torch.tensor([ids]); out,_=m(x[:,:-1]); target=x[:,1:]; mask=torch.zeros_like(target,dtype=torch.bool); mask[:,len(tok.encode(p))-1:]=True; z=out.index_select(-1,valid); vals.append(float(torch.nn.functional.cross_entropy(z[mask],lookup[target[mask]])))
 return sum(vals)/len(vals)
def generate(m,tok,p,n=40):
 active=effective_token_ids(tok); x=torch.tensor([tok.encode(p)]); m.eval();
 with torch.inference_mode():
  out,st=m(x)
  for _ in range(n):
   z=out[0,-1]; mask=torch.full_like(z,float('-inf')); idx=torch.tensor(active); mask[idx]=z[idx]; t=int(mask.argmax())
   if t==2: break
   x=torch.cat([x,torch.tensor([[t]])],1); out,st=m(torch.tensor([[t]]),st)
 try: return tok.decode(x[0].tolist())[len(p):]
 except Exception as e: return f"<decode_error:{e}>"
def main():
 torch.manual_seed(20261123); texts=[p+a for p,a in TRAIN+L2+L3]; tok=ByteBPETokenizer.train(texts,vocab_size=4096); cfg=LanguageV2Config(vocab_size=4096,d_model=32,n_blocks=2,block_hidden=32,workspace_dim=16,workspace_slots=2); m=LanguageV2Model(cfg); m.train(); m.use_router_load_update=False; before=sha(m); opt=torch.optim.AdamW(m.parameters(),lr=.001); losses=[]; start=time.perf_counter(); steps=0
 for step in range(MAX_STEPS):
  if time.perf_counter()-start>=LIMIT: break
  batch=[TRAIN[(step*2+i)%len(TRAIN)] for i in range(2)]; seq=[tok.encode(p+a,add_eos=True) for p,a in batch]; n=max(map(len,seq)); x=torch.zeros((2,n),dtype=torch.long); masks=torch.zeros((2,n-1),dtype=torch.bool)
  for i,(p,a) in enumerate(batch):
   s=seq[i]; x[i,:len(s)]=torch.tensor(s); masks[i,max(0,len(tok.encode(p))-1):len(s)-1]=True
  out,_=m(x[:,:-1]); valid=torch.tensor(effective_token_ids(tok)); lookup=torch.full((tok.vocab_size,),-1,dtype=torch.long); lookup[valid]=torch.arange(len(valid)); z=out.index_select(-1,valid); target=x[:,1:]; loss=torch.nn.functional.cross_entropy(z[masks],lookup[target[masks]])
  if not torch.isfinite(loss): raise RuntimeError('non-finite loss')
  loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step(); opt.zero_grad(); losses.append(float(loss)); steps+=1
 m.eval(); ck=OUT/"candidate_v13_main.pt"; save_language_v2(ck,m,tok,steps=steps,seed=20261123,manifest_hash="language-v13-synthetic"); eval_rows=[]
 for level,rows in [("L1",TRAIN),("L2",L2),("L3",L3)]:
  for p,a in rows: eval_rows.append({"level":level,"prompt":p,"expected":a,"generated":generate(m,tok,p),"source":"neural"})
 result={"status":"COMPLETED_AUTHORIZED_TRAINING","steps":steps,"optimizer_steps":steps,"wall_seconds":time.perf_counter()-start,"wall_limit":LIMIT,"loss_first":losses[0],"loss_last":losses[-1],"train_eval_loss":answer_loss(m,tok,TRAIN),"l2_eval_loss":answer_loss(m,tok,L2),"l3_eval_loss":answer_loss(m,tok,L3),"parent_sha":before,"candidate_sha":sha(m),"parameters_changed":before!=sha(m),"candidate":str(ck),"stable_modified":False,"network_requests":0,"device":"cpu","effective_token_count":len(effective_token_ids(tok)),"masked_token_count":4096-len(effective_token_ids(tok)),"judgement":"SENTENCE_MEMORIZED_PENDING_GENERATION_REVIEW","optimizer_steps_authorized":256}
 OUT.mkdir(exist_ok=True); (OUT/"LANGUAGE_V1_3_EVAL.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8'); (OUT/"LANGUAGE_V1_3_GENERATION.jsonl").write_text("\n".join(json.dumps(x,ensure_ascii=False) for x in eval_rows)+"\n",encoding='utf-8'); (OUT/"LANGUAGE_V1_3_REPORT.md").write_text(f"# Language V1.3 训练报告\n\n实际执行 {steps} 次 optimizer.step，耗时 {result['wall_seconds']:.2f}s。CPU-only，Stable未修改，网络0。Loss {losses[0]:.6f}->{losses[-1]:.6f}。L1/L2/L3 原始生成见 LANGUAGE_V1_3_GENERATION.jsonl。当前只允许根据生成结果判定记忆或泛化，不能将 Loss 下降直接称为语言理解。\n",encoding='utf-8')
 print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__': main()
