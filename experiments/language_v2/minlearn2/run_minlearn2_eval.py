import json,importlib.util,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]; sys.path.insert(0,str(ROOT/"src")); OUT=ROOT/"experiments/language_v2/minlearn2"
spec=importlib.util.spec_from_file_location("tr",OUT/"run_minlearn2_training.py"); m=importlib.util.module_from_spec(spec); sys.modules["tr"]=m; spec.loader.exec_module(m)
d=m.data
arms=("A_HARD_TOPK_ORIGINAL","B_SOFT_MIXTURE_TRAINABLE","C_SOFT_MIXTURE_ROUTER_FROZEN","D_SOFT_MIXTURE_LABEL_INDEPENDENT")
ev=[]
for split,items in (("train",d.build_train16()),("dev",d.build_dev()),("sealed",d.build_sealed())):
    for arm in arms: ev.append(m.eval_arm(arm,items)|{"split":split})
(OUT/"MINLEARN2_EVAL.json").write_text(json.dumps(ev,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps([{"arm":x["arm"],"split":x["split"],"exact":x["exact"],"chars":x["unicode_char_accuracy"],"eos":x["eos_accuracy"],"invalid_utf8":x["invalid_utf8"]} for x in ev],ensure_ascii=False))

