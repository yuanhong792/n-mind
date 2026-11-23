from __future__ import annotations
import json, platform, sys, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src"))
import torch
from nmind.language.tokenizer_v2 import ByteBPETokenizer
from nmind.language.neural_language_v2 import LanguageV2Config, LanguageV2Model
from nmind.language.bpe_decoder import effective_token_ids, constrained_causal_loss
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"experiments/language_v1"
ROWS=[("用户：你好\n助手：","你好！"),("用户：你叫什么？\n助手：","我叫N-MIND。"),("用户：谢谢\n助手：","不客气！"),("用户：小明有红色苹果\n助手：","小明的苹果是红色的。"),("用户：我叫小明\n助手：","你好，小明！")]
def main():
 tok=ByteBPETokenizer.train([p+a for p,a in ROWS],vocab_size=4096); cfg=LanguageV2Config(vocab_size=tok.vocab_size,d_model=32,n_blocks=2,block_hidden=32,workspace_dim=16,workspace_slots=2); m=LanguageV2Model(cfg); m.use_router_load_update=False; m.eval()
 sample=tok.encode(ROWS[0][0]+ROWS[0][1],add_eos=True); x=torch.tensor([sample]);
 with torch.inference_mode(): out,_=m(x[:,:-1])
 loss=constrained_causal_loss(out,x[:,1:],tok)
 result={"status":"READY_NO_TRAINING","optimizer_steps":0,"device":"cuda" if torch.cuda.is_available() else "cpu","torch":torch.__version__,"cuda":torch.cuda.is_available(),"tokenizer_vocab_size":tok.vocab_size,"effective_token_count":len(effective_token_ids(tok)),"masked_token_count":tok.vocab_size-len(effective_token_ids(tok)),"model_parameters":m.n_params(),"forward_loss":float(loss),"levels":{"L1":"READY","L2":"PRE_REGISTERED","L3":"PRE_REGISTERED"},"new_training_authorization":"PENDING_USER_APPROVAL"}
 OUT.mkdir(exist_ok=True); (OUT/"LANGUAGE_V1_3_EVAL.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__': main()
