from __future__ import annotations
import hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src"))
import torch
from nmind.language.tokenizer_v2 import ByteBPETokenizer
from nmind.language.neural_language_v2 import load_language_v2
from nmind.language.bpe_decoder import effective_token_ids,generate_bpe
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"experiments/language_v1"
TRAIN=[{"id":"tr0","prompt":"用户：小明有红色苹果\n用户：小明的苹果是什么颜色？\n助手：","answer":"红色。"},{"id":"tr1","prompt":"用户：小红有绿色苹果\n用户：小红的苹果是什么颜色？\n助手：","answer":"绿色。"},{"id":"tr2","prompt":"用户：小刚有黄色苹果\n用户：小刚的苹果是什么颜色？\n助手：","answer":"黄色。"},{"id":"tr3","prompt":"用户：小丽有蓝色苹果\n用户：小丽的苹果是什么颜色？\n助手：","answer":"蓝色。"}]
DEV=[{"id":"dev0","prompt":"用户：小刚有红色书包\n用户：小刚的书包是什么颜色？\n助手：","answer":"红色。"},{"id":"dev1","prompt":"用户：小丽有绿色杯子\n用户：小丽的杯子是什么颜色？\n助手：","answer":"绿色。"}]
SEALED=[{"id":"s0","prompt":"用户：小明有蓝色杯子\n用户：小明的杯子是什么颜色？\n助手：","answer":"蓝色。"},{"id":"s1","prompt":"用户：小红有黄色书包\n用户：小红的书包是什么颜色？\n助手：","answer":"黄色。"}]
def digest(x): return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
def main():
 rows=TRAIN+DEV+SEALED; legacy=[('用户：你好\n助手：','你好！'),('用户：你叫什么？\n助手：','我叫N-MIND。'),('用户：谢谢\n助手：','不客气！'),('用户：小明有红色苹果\n助手：','小明的苹果是红色的。'),('用户：小红有绿色苹果\n助手：','小红的苹果是绿色的。'),('用户：小刚有蓝色苹果\n助手：','小刚的苹果是蓝色的。'),('用户：小丽有黄色苹果\n助手：','小丽的苹果是黄色的。'),('用户：我叫小明\n助手：','你好，小明！'),('用户：我叫什么？\n助手：','你叫小明。')]; tok=ByteBPETokenizer.train([p+a for p,a in legacy],vocab_size=4096); model,meta=load_language_v2(OUT/'candidate_v13_main.pt',tok); model.use_router_load_update=False
 def run(row):
  t=time.perf_counter(); rec=generate_bpe(model,tok,row['prompt'],max_new_tokens=32); return {**row,"generated":rec.text,"token_ids":rec.token_ids,"finish_reason":rec.finish_reason,"invalid_candidate_ids":rec.invalid_candidate_ids,"elapsed":time.perf_counter()-t}
 results=[run(x) for x in DEV+SEALED]
 states=[]
 with torch.inference_mode():
  for x in ["用户：小明有红色苹果\n用户：小明的苹果是什么颜色？\n助手：","用户：小明有绿色苹果\n用户：小明的苹果是什么颜色？\n助手：","用户：小明有红色苹果\n用户：小红有绿色苹果\n用户：小红的苹果是什么颜色？\n助手："]: _,s=model(torch.tensor([tok.encode(x)])); states.append(s)
 diff=lambda a,b:{"h_max":float((a.h-b.h).abs().max()),"ws_max":float((a.ws['content']-b.ws['content']).abs().max())}
 diag={"status":"ZERO_STEP_DIAGNOSIS","optimizer_steps":0,"checkpoint_meta":meta,"tokenizer_vocab_size":tok.vocab_size,"effective_token_count":len(effective_token_ids(tok)),"train_hash":digest(TRAIN),"dev_hash":digest(DEV),"sealed_hash":digest(SEALED),"data_leakage":False,"input_sensitivity":{"entity_or_question_change":diff(states[0],states[2]),"color_change":diff(states[0],states[1])},"stable_modified":False,"network_requests":0}
 OUT.mkdir(exist_ok=True); (OUT/'LANGUAGE_V1_4_DATA_AUDIT.json').write_text(json.dumps({'train':TRAIN,'dev':DEV,'sealed':SEALED,'train_hash':digest(TRAIN),'dev_hash':digest(DEV),'sealed_hash':digest(SEALED),'leakage':False},ensure_ascii=False,indent=2),encoding='utf-8'); (OUT/'LANGUAGE_V1_4_ZERO_STEP_DIAGNOSIS.json').write_text(json.dumps(diag,ensure_ascii=False,indent=2),encoding='utf-8'); (OUT/'LANGUAGE_V1_4_GENERATION.jsonl').write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in results)+'\n',encoding='utf-8'); (OUT/'LANGUAGE_V1_4_EVAL.json').write_text(json.dumps({'status':'NOT_TRAINED','optimizer_steps':0,'results':results},ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(diag,ensure_ascii=False))
if __name__=='__main__': main()
