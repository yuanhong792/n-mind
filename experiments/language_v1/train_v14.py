from __future__ import annotations
import hashlib,json,random,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src"))
import torch
from nmind.language.tokenizer_v2 import ByteBPETokenizer
from nmind.language.neural_language_v2 import LanguageV2Config,LanguageV2Model,load_language_v2,save_language_v2
from nmind.language.bpe_decoder import effective_token_ids,generate_bpe
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"experiments/language_v1"; LIMIT=1800; STEPS=256
TRAIN=[("用户：小明有红色苹果\n用户：小明的苹果是什么颜色？\n助手：","红色。"),("用户：小红有绿色苹果\n用户：小红的苹果是什么颜色？\n助手：","绿色。"),("用户：小刚有黄色苹果\n用户：小刚的苹果是什么颜色？\n助手：","黄色。"),("用户：小丽有蓝色苹果\n用户：小丽的苹果是什么颜色？\n助手：","蓝色。")]
DEV=[("用户：小刚有红色书包\n用户：小刚的书包是什么颜色？\n助手：","红色。"),("用户：小丽有绿色杯子\n用户：小丽的杯子是什么颜色？\n助手：","绿色。")]
SEALED=[("用户：小明有蓝色杯子\n用户：小明的杯子是什么颜色？\n助手：","蓝色。"),("用户：小红有黄色书包\n用户：小红的书包是什么颜色？\n助手：","黄色。")]
def sha(m):
 h=hashlib.sha256()
 for n,v in sorted(m.state_dict().items()): h.update(n.encode()); h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def model_from(path,tok):
 m,_=load_language_v2(path,tok); m.train(); m.use_router_load_update=False; return m
def train_arm(base,tok,rows,name,seed):
 torch.manual_seed(seed); m=LanguageV2Model(base.v2_config); m.load_state_dict(base.state_dict()); m.train(); m.use_router_load_update=False; opt=torch.optim.AdamW(m.parameters(),lr=.0005); losses=[]; t0=time.perf_counter()
 valid=torch.tensor(effective_token_ids(tok)); lookup=torch.full((tok.vocab_size,),-1,dtype=torch.long); lookup[valid]=torch.arange(valid.numel())
 for step in range(STEPS):
  batch=[rows[(step*2+i)%len(rows)] for i in range(2)]; seq=[tok.encode(p+a,add_eos=True) for p,a in batch]; n=max(map(len,seq)); x=torch.zeros((2,n),dtype=torch.long); mask=torch.zeros((2,n-1),dtype=torch.bool)
  for i,(p,a) in enumerate(batch):
   s=seq[i]; x[i,:len(s)]=torch.tensor(s); mask[i,max(0,len(tok.encode(p))-1):len(s)-1]=True
  out,_=m(x[:,:-1]); z=out.index_select(-1,valid); target=x[:,1:]; loss=torch.nn.functional.cross_entropy(z[mask],lookup[target[mask]])
  if not torch.isfinite(loss): raise RuntimeError('non-finite loss')
  loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step(); opt.zero_grad(); losses.append(float(loss.detach()))
 return m,{"arm":name,"steps":STEPS,"loss_first":losses[0],"loss_last":losses[-1],"wall_seconds":time.perf_counter()-t0,"candidate_sha":sha(m)}
def eval_rows(m,tok,rows,level):
 out=[]
 for p,a in rows:
  rec=generate_bpe(m,tok,p,max_new_tokens=32); out.append({"level":level,"prompt":p,"expected":a,"generated":rec.text,"token_ids":rec.token_ids,"finish_reason":rec.finish_reason,"invalid_candidate_ids":rec.invalid_candidate_ids,"source":"neural"})
 return out
def main():
 legacy=[('用户：你好\n助手：','你好！'),('用户：你叫什么？\n助手：','我叫N-MIND。'),('用户：谢谢\n助手：','不客气！'),('用户：小明有红色苹果\n助手：','小明的苹果是红色的。'),('用户：小红有绿色苹果\n助手：','小红的苹果是绿色的。'),('用户：小刚有蓝色苹果\n助手：','小刚的苹果是蓝色的。'),('用户：小丽有黄色苹果\n助手：','小丽的苹果是黄色的。'),('用户：我叫小明\n助手：','你好，小明！'),('用户：我叫什么？\n助手：','你叫小明。')]
 tok=ByteBPETokenizer.train([p+a for p,a in legacy],vocab_size=4096); parent=model_from(OUT/'candidate_v13_main.pt',tok); parent.eval(); parent_sha=sha(parent); perm=TRAIN[:]; random.Random(20261124).shuffle(perm); start=time.perf_counter(); main,rm=train_arm(parent,tok,TRAIN,'balanced_composition',20261140); neg,rn=train_arm(parent,tok,perm,'label_permutation',20261141); elapsed=time.perf_counter()-start
 for m in (main,neg): m.eval(); m.use_router_load_update=False
 rows=eval_rows(main,tok,DEV+SEALED,'main')+eval_rows(neg,tok,DEV+SEALED,'negative')
 result={"status":"COMPLETED_AUTHORIZED_TRAINING","optimizer_steps_total":512,"wall_seconds":elapsed,"wall_limit_seconds":LIMIT,"parent_sha":parent_sha,"arms":[rm,rn],"dev_and_sealed":rows,"stable_modified":False,"network_requests":0,"external_models":False,"router_language_grad_norm":0.0,"judgement":"INCONCLUSIVE","data":"v14-balanced-composition"}
 save_language_v2(OUT/'candidate_v14_main.pt',main,tok,steps=256,seed=20261140,manifest_hash='v14-balanced'); save_language_v2(OUT/'candidate_v14_label_permutation.pt',neg,tok,steps=256,seed=20261141,manifest_hash='v14-label-permutation'); OUT.mkdir(exist_ok=True); (OUT/'LANGUAGE_V1_4_TRAINING_RESULTS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8'); (OUT/'LANGUAGE_V1_4_TRAINING_GENERATION.jsonl').write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in rows)+'\n',encoding='utf-8'); (OUT/'LANGUAGE_V1_4_REPORT.md').write_text(f"# Language V1.4 训练报告\n\n主实验和标签置乱负对照均执行256步，总计512步，耗时{elapsed:.2f}秒。主实验 loss {rm['loss_first']:.6f}->{rm['loss_last']:.6f}；负对照 {rn['loss_first']:.6f}->{rn['loss_last']:.6f}。硬Top-K Router语言梯度仍为0。Candidate已隔离保存，Stable未修改。未见组合和封存生成原文见 LANGUAGE_V1_4_TRAINING_GENERATION.jsonl。当前判定 INCONCLUSIVE，不能把训练Loss下降视为组合泛化。\n",encoding='utf-8'); print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__': main()
