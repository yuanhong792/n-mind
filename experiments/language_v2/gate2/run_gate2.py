from __future__ import annotations
import argparse, json
from pathlib import Path
from nmind.language.gate2_dataset import build_gate2_dataset, manifest
from nmind.language.gate2_preflight import run_preflight
from nmind.language.gate2_trainer import train_arm
from nmind.language.gate2_evaluator import evaluate_checkpoint

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/"experiments/language_v2/gate2"

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--preflight-only",action="store_true",default=True); ap.add_argument("--i-authorize-gate2",action="store_true"); ap.add_argument("--offline",action="store_true",default=True); args=ap.parse_args()
    pre=run_preflight(ROOT); OUT.mkdir(parents=True,exist_ok=True); (OUT/"PREFLIGHT.json").write_text(json.dumps(pre,ensure_ascii=False,indent=2),encoding="utf-8")
    if not args.i_authorize_gate2 or pre["status"] != "PASS": print(json.dumps({"status":"BLOCKED_PRECONDITION" if pre["status"] != "PASS" else "PREFLIGHT_ONLY","preflight":pre},ensure_ascii=False)); return 2 if pre["status"] != "PASS" else 0
    data=build_gate2_dataset(); (OUT/"dataset_manifest.json").write_text(json.dumps(manifest(data),ensure_ascii=False,indent=2),encoding="utf-8")
    runs=[("main",512,20261123,1e-3,False),("permute",512,20261124,1e-3,True),("lr0",32,20261125,0.0,False)]
    results=[]
    for name,steps,seed,lr,perm in runs: results.append(train_arm(data["train"],OUT/"runs"/name,steps=steps,seed=seed,lr=lr,permute=perm))
    evals={}
    for result in results:
        evals[result["arm"]]=evaluate_checkpoint(result["checkpoint"], data["sealed"])
    (OUT/"raw_training.json").write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT/"sealed_evaluation.json").write_text(json.dumps(evals,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"status":"TRAINED_CANDIDATES_EVALUATED_ONCE","results":results,"sealed":{k:{x:y for x,y in v.items() if x!="items"} for k,v in evals.items()}},ensure_ascii=False))

if __name__ == "__main__": main()
