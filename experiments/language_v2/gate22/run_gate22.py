from __future__ import annotations
import json, time
from pathlib import Path
from nmind.language.gate22_preflight import run_gate22_preflight
from nmind.language.gate2_dataset import build_gate2_dataset, manifest
from nmind.language.gate2_trainer import train_arm
from nmind.language.gate2_evaluator import evaluate_checkpoint
from nmind.language.gate21_baseline import zero_step_gate2_baseline

ROOT=Path(__file__).resolve().parents[3]; OUT=ROOT/"experiments/language_v2/gate22"

def write(name, value):
    (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

def main():
    OUT.mkdir(parents=True, exist_ok=True); pre=run_gate22_preflight(); write("GATE22_PREFLIGHT_VERIFIED.json", pre)
    if pre["status"] != "PASS": raise SystemExit("BLOCKED_PREFLIGHT")
    data=build_gate2_dataset(); write("dataset_manifest.json", manifest(data))
    prereg={"format":"language-v2-gate22/1","status":"AUTHORIZED_SINGLE_ROUND","init_seed":20261123,"permutation_seed":20261124,"arms":{"main":512,"label_permutation":512,"lr_zero":32},"total_budget":1056,"wall_limit_s":5400,"loss":"answer_span_plus_eos","evaluation":"neural-only greedy, 240 sealed + 80 pairs","memory_retrieval_tools_rules":False,"stable":"read-only"}; write("GATE22_PREREGISTRATION.json",prereg)
    zero=zero_step_gate2_baseline(data["sealed"],seed=20261123); write("GATE22_ZERO_BASELINE.json",zero)
    start=time.perf_counter(); results=[]; budget=0
    specs=[("main",512,20261123,1e-3,False,None),("label_permutation",512,20261124,1e-3,True,20261124),("lr_zero",32,20261125,0.0,False,None)]
    for name,steps,seed,lr,perm,pseed in specs:
        if budget+steps>1056 or time.perf_counter()-start>5400: raise SystemExit("BLOCKED_RESOURCE")
        result=train_arm(data["train"],OUT/"runs"/name,steps=steps,seed=seed,lr=lr,permute=perm,init_seed=20261123,permutation_seed=pseed); results.append(result); budget += result["optimizer_steps_executed"]
        write({"main":"GATE22_TRAIN_MAIN.json","label_permutation":"GATE22_TRAIN_PERMUTED.json","lr_zero":"GATE22_TRAIN_LR0.json"}[name],result)
    evals={"zero":zero}
    for result in results: evals[result["arm"]]=evaluate_checkpoint(result["checkpoint"],data["sealed"])
    write("GATE22_EVAL_PER_ITEM.json",evals)
    write("GATE22_BUDGET_AUDIT.json",{"status":"COMPLETED","steps_by_arm":{r["arm"]:r["optimizer_steps_executed"] for r in results},"optimizer_steps_total":budget,"wall_s":time.perf_counter()-start,"within_budget":budget<=1056 and time.perf_counter()-start<=5400,"stable":"read-only"})
    print(json.dumps({"status":"COMPLETED","steps":budget,"wall_s":time.perf_counter()-start,"metrics":{k:{x:v[x] for x in ["exact_match","counterfactual_pair_pass"]} for k,v in evals.items()}},ensure_ascii=False))

if __name__ == "__main__": main()
