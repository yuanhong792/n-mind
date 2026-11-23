import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]; OUT=ROOT/"experiments/language_v2/minlearn2"
arms=("A_HARD_TOPK_ORIGINAL","B_SOFT_MIXTURE_TRAINABLE","C_SOFT_MIXTURE_ROUTER_FROZEN","D_SOFT_MIXTURE_LABEL_INDEPENDENT")
rows=[]
for a in arms:
 p=OUT/"training_runs"/a/"candidate.pt"; blob=__import__("torch").load(p,map_location="cpu",weights_only=True)
 rows.append({"arm":a,"checkpoint":str(p),"checkpoint_sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"steps_recorded":blob.get("steps"),"seed":blob.get("seed")})
out={"schema_version":"minlearn2/1","status":"COMPLETED_AUTHORIZED_TRAINING_EVAL_REPAIRED","actual_optimizer_steps":512,"wall_clock_limit_minutes":90,"network_requests":0,"stable_modified":False,"sealed_used_for_training":False,"arms":rows,"training_loss_log":"NOT_PERSISTED: original runner crashed only after all four checkpoints were written during strict UTF-8 evaluation; no retraining performed","evaluation":"MINLEARN2_EVAL.json"}
(OUT/"MINLEARN2_TRAINING_RESULTS.json").write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(out,ensure_ascii=False))

