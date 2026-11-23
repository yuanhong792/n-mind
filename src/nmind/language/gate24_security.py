from __future__ import annotations
from .gate24_dataset import build_gate24, manifest, stratified_label_permutation, context_blind

def audit_controls():
    data=build_gate24(); perm,meta=stratified_label_permutation(data["train"])
    blind=context_blind(data["train"])
    train_ids={x.group_id for x in data["train"]}; sealed_ids={x.group_id for x in data["sealed"]}
    return {"status":"PASS" if meta["fixed_points"]==0 and meta["changed_fraction"]>=.9 and not train_ids&sealed_ids else "BLOCKED_INVALID_CONTROL","manifest":manifest(data),"label_broken":{k:v for k,v in meta.items() if k!="mapping"},"label_mapping_sample":meta["mapping"][:20],"context_blind":{"count":len(blind),"contains_location_relation":any("在一号位置" in x.prompt or "在二号位置" in x.prompt or "在三号位置" in x.prompt or "在四号位置" in x.prompt for x in blind)},"training_sealed_group_overlap":len(train_ids&sealed_ids)}
