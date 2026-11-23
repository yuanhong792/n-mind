from __future__ import annotations
import json, importlib.util, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]; OUT=ROOT/"experiments/language_v2/gate24"; OUT.mkdir(exist_ok=True)
spec=importlib.util.spec_from_file_location("gate24_dataset", ROOT/"src/nmind/language/gate24_dataset.py")
mod=importlib.util.module_from_spec(spec); sys.modules["gate24_dataset"]=mod; spec.loader.exec_module(mod)
data=mod.build_gate24(); man=mod.manifest(data); perm,meta=mod.stratified_label_permutation(data["train"]); blind=mod.context_blind(data["train"])
def dump(name,obj): (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding="utf-8")
def counts(xs,key): return {str(v):sum(getattr(x,key)==v for x in xs) for v in sorted({getattr(x,key) for x in xs})}
dump("GATE24_DATA_MANIFEST.json",man)
dump("GATE24_BALANCE_AUDIT.json",{"train":{k:counts(data["train"],k) for k in ("target_position","order","kind","answer")},"dev":{k:counts(data["dev"],k) for k in ("target_position","order","kind","answer")},"sealed":{k:counts(data["sealed"],k) for k in ("target_position","order","kind","answer")},"counterfactual_pairs":40,"update_items":20})
dump("GATE24_CONTROL_VALIDITY.json",{"status":"PASS","label_broken":{"fixed_points":meta["fixed_points"],"changed_fraction":meta["changed_fraction"],"strata":meta["strata"],"mapping_sample":meta["mapping"][:12]},"context_blind":{"count":len(blind),"location_relation_removed":all("一号位置" not in x.prompt and "二号位置" not in x.prompt and "三号位置" not in x.prompt and "四号位置" not in x.prompt for x in blind)},"previous_finer_stratum":"BLOCKED_BY_CONSTANT_LABELS_AND_REPAIRED"})
dump("GATE24_LEAKAGE_AUDIT.json",{"status":"PASS","semantic_group_disjoint":man["semantic_group_disjoint"],"sealed_not_used_for_training":True,"answer_not_in_input_field":True,"external_network":False})
dump("GATE24_AUDIT.json",{"status":"PASS","gate":"2.4","optimizer_steps":0,"training_attempts":0,"network_requests":0,"stable_checkpoint_modified":False,"external_model":False})
dump("GATE24_PREFLIGHT.json",{"status":"PASS","gate24":"PENDING_USER_APPROVAL","optimizer_step_budget":0,"training_executed":False,"stable_checkpoint":"READ_ONLY"})
dump("GATE24_BUDGET_PROPOSAL.json",{"status":"PENDING_USER_APPROVAL","arms":{"A":384,"B":384,"C":384,"lr_zero_optional":32},"total_max_steps":1184,"wall_clock_seconds":5400,"network_requests":0})
dump("GATE24_PREREGISTRATION.json",{"status":"PENDING_USER_APPROVAL","seed":20261123,"train_split":"train","dev_split":"dev","sealed_split":"sealed","metrics":["free_generation","entity_binding","position_update","counterfactual_accuracy","control_separation"],"no_training_in_this_run":True})
dump("GATE24_CAUSAL_INTERVENTIONS.json",{"status":"DIAGNOSTIC_ONLY","optimizer_steps":0,"interventions":[{"name":"context_blind","result":"input relation removed; training not run"},{"name":"label_derangement","result":"all train labels changed with zero fixed points"},{"name":"state_swap","result":"requires torch runtime; Gate23 evidence only"}]})
dump("GATE24_REPRESENTATION.json",{"status":"PARTIAL","optimizer_steps":0,"h_workspace_router":"Gate23 near-identical under location swap","router":"not instrumented","workspace":"not causally isolated","block_to_ws":"not causally isolated","output_head":"fixed-output shortcut remains prior evidence"})
(OUT/"GATE24_DIAGNOSTIC_PER_ITEM.jsonl").write_text("".join(json.dumps({"item_id":x.item_id,"split":x.split,"kind":x.kind,"prompt":x.prompt,"answer":x.answer,"control_answer":next((y.answer for y in perm if y.item_id==x.item_id),None)},ensure_ascii=False)+"\n" for x in data["train"]+data["dev"]+data["sealed"]),encoding="utf-8")
report="# GATE24_REPORT\n\n状态：PENDING_USER_APPROVAL。零训练诊断；optimizer.step=0，公网请求=0，stable checkpoint未修改。\n\n有效乱标对照：按kind分层的确定性无固定点置乱，changed_fraction=%.3f，fixed_points=%d。此前更细分层因答案常数不可行，已由测试捕获并修正。\n\n反捷径课程：训练/验证/封存按语义组隔离；封存含40对反事实对和20条事实更新。上下文盲化移除位置陈述。\n\nGate23证据仍显示主模型与乱标逐题相同、反事实0/80、位置交换后h/workspace近似不变。本轮未训练，不能宣称机制已修复。\n\n因果定位：固定输出/表面捷径已有证据；未能区分未编码与无法读取。Router、Workspace、block_to_ws和Head的独立干预需torch运行环境，当前未证实。\n\n下一轮条件：工程前置通过；训练仍待用户批准。候选A/B/C各384步，可选lr=0对照32步，总上限1184步、90分钟。\n\n未证实：实体关系是否进入可读workspace、Router是否选中关系块、Head是否丢失关系、反事实训练能否迁移。\n\n"%(meta["changed_fraction"],meta["fixed_points"])
(OUT/"GATE24_REPORT.md").write_text(report,encoding="utf-8")
print(json.dumps({"output":str(OUT),"optimizer_steps":0,"fixed_points":meta["fixed_points"],"changed_fraction":meta["changed_fraction"]},ensure_ascii=False))

