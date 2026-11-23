from __future__ import annotations
import json,sys,importlib.util
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[3]; OUT=ROOT/"experiments/language_v2/gate24"; sys.path.insert(0,str(ROOT/"src"))
spec=importlib.util.spec_from_file_location("gate24_dataset",ROOT/"src/nmind/language/gate24_dataset.py"); mod=importlib.util.module_from_spec(spec); sys.modules["gate24_dataset"]=mod; spec.loader.exec_module(mod)
from nmind.language.tokenizer_v2 import TokenizerV2
from nmind.language.neural_language_v2 import load_language_v2
def eval_arm(name,items):
    tok=TokenizerV2(); m,_=load_language_v2(OUT/"training_runs"/name/"candidate.pt",tok); m.eval(); rows=[]; tf=0; first=0
    for x in items:
        full=tok.encode(x.prompt+x.answer,add_eos=True); prompt=tok.encode(x.prompt); target=full[len(prompt):]; inp=torch.tensor([full[:-1]],dtype=torch.long)
        with torch.inference_mode(): logits,_=m(inp)
        start=max(0,len(prompt)-1); pred=logits[0,start:start+len(target)].argmax(-1).tolist()
        ok=pred==target; tf+=ok; first+=int(pred[:1]==target[:1])
        rows.append({"item_id":x.item_id,"kind":x.kind,"tf_exact":ok,"first_token":bool(pred[:1]==target[:1]),"answer_len":len(target)})
    return {"arm":name,"sealed_items":len(items),"tf_exact":tf/len(items),"first_token_accuracy":first/len(items),"rows":rows}
def main():
    d=mod.build_gate24(); results=[eval_arm(n,d["sealed"]) for n in ("A_MAIN_CONTRASTIVE","B_LABEL_BROKEN","C_CONTEXT_BLIND","D_ZERO_UPDATE")]
    out={"status":"COMPLETED_NEURAL_ONLY_EVAL","source":"sealed evaluation only; no sealed labels used for training","optimizer_steps":1152,"arms":[{"arm":x["arm"],"sealed_items":x["sealed_items"],"tf_exact":x["tf_exact"],"first_token_accuracy":x["first_token_accuracy"]} for x in results],"detail":results}
    (OUT/"GATE24_NEURAL_EVAL.json").write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(out,ensure_ascii=False))
if __name__=="__main__": main()

