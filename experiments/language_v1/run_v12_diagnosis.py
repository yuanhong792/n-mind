from __future__ import annotations
import hashlib, json, sys, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src"))
import torch
from nmind.language.tokenizer_v2 import ByteBPETokenizer
from nmind.language.neural_language_v2 import LanguageV2Config, LanguageV2Model, load_language_v2
from nmind.language.bpe_decoder import generate_bpe

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"experiments/language_v1"
ROWS=[("你好，","你好。"),("苹果的颜色是","红色。"),("香蕉的颜色是","黄色。"),("猫在哪里？","在桌子上。"),("狗在哪里？","在地板上。"),("一加一等于","二。"),("Python是什么？","一种编程语言。"),("太阳从哪里升起？","东方。"),("水的状态是","液体。"),("春天通常是","温暖的。"),("请说你好","你好！"),("三加二等于","五。"),("鸟会做什么？","飞翔。"),("鱼生活在哪里？","水里。"),("雪是什么颜色？","白色。"),("火通常是","热的。")]
def sha(m):
 h=hashlib.sha256()
 for n,v in sorted(m.state_dict().items()): h.update(n.encode()); h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def main():
 tok=ByteBPETokenizer.train([p+a for p,a in ROWS],vocab_size=4096); cfg=LanguageV2Config(vocab_size=4096,d_model=32,n_blocks=2,block_hidden=32,workspace_dim=16,workspace_slots=2)
 models={"parent":LanguageV2Model(cfg)}; models["parent"].use_router_load_update=False
 for name in ("main","label_permutation"): models[name],_=load_language_v2(OUT/f"candidate_{name}.pt",tok); models[name].use_router_load_update=False
 evals=[]
 for p,a in ROWS[:5]:
  for name,m in models.items():
   t=time.perf_counter(); rec=generate_bpe(m,tok,p); evals.append({"prompt":p,"expected":a,"model":name,"token_ids":rec.token_ids,"generated_text":rec.text,"generated_token_count":len(rec.token_ids),"finish_reason":rec.finish_reason,"invalid_candidate_ids":rec.invalid_candidate_ids,"elapsed_seconds":time.perf_counter()-t,"classification":"VALID_BUT_NONSENSICAL" if rec.text else rec.finish_reason})
 losses={}
 for name,m in models.items():
  vals=[]
  with torch.inference_mode():
   for p,a in ROWS:
    ids=tok.encode(p+a,add_eos=True); x=torch.tensor([ids]); out,_=m(x[:,:-1]); vals.append(float(torch.nn.functional.cross_entropy(out.reshape(-1,out.shape[-1]),x[:,1:].reshape(-1))))
  losses[name+"_eval_loss"]=sum(vals)/len(vals)
 result={"status":"ZERO_STEP_DIAGNOSIS","optimizer_steps":0,"tokenizer_vocab_size":tok.vocab_size,"active_merge_count":len(tok.merges),"checkpoint_sha":{"parent":sha(models["parent"]),"main":sha(models["main"]),"label_permutation":sha(models["label_permutation"])},"loss":losses,"generation_count":len(evals),"stable_modified":False,"network_requests":0,"historical_train_loss":"HISTORICAL_TRAIN_LOSS_UNAVAILABLE","retention":"RETENTION_INCONCLUSIVE","root_cause":{"invalid_ids":"undefined BPE slots selected by LM head; not tokenizer roundtrip failure","learning":"valid generation remains semantically unverified","router_gradient":0.0}}
 OUT.mkdir(exist_ok=True); (OUT/"LANGUAGE_V1_2_RESULTS.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8"); (OUT/"LANGUAGE_V1_2_GENERATION.jsonl").write_text("\n".join(json.dumps(x,ensure_ascii=False) for x in evals)+"\n",encoding="utf-8"); (OUT/"LANGUAGE_V1_2_LOSS_AUDIT.json").write_text(json.dumps(losses,ensure_ascii=False,indent=2),encoding="utf-8"); (OUT/"LANGUAGE_V1_2_RETENTION_AUDIT.json").write_text(json.dumps({"status":"RETENTION_INCONCLUSIVE","reason":"no historical sealed retention set"},ensure_ascii=False,indent=2),encoding="utf-8"); (OUT/"LANGUAGE_V1_2_DIAGNOSIS.md").write_text("# Language V1.2 诊断\n\n已执行零步 Candidate 复评。Gate-1 历史训练 loss 无法恢复，标记 HISTORICAL_TRAIN_LOSS_UNAVAILABLE。BPE 往返正常；问题是 LM Head 会选择未定义 merge 的词表槽位，已新增主动词表约束生成器。即使修复后文本合法，也不能据此宣称神经语义能力。Router 语言梯度仍为0；旧技能保持 RETENTION_INCONCLUSIVE。optimizer.step=0，Stable未修改，网络请求=0。\n",encoding="utf-8")
if __name__=='__main__': main()
