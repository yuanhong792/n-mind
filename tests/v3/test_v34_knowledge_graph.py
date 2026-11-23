from nmind.v3.facade import V3Facade

def test_entities_assertions_conflict_and_revoke(tmp_path):
    f=V3Facade(tmp_path); a=f.v34_add_entity({"canonical_name":"蓝盒项目","kind":"project","aliases":["蓝盒"]}); b=f.v34_add_entity({"canonical_name":"小林","kind":"person"}); ev=f.v34_add_evidence({"source_kind":"user","locator":"user://test","excerpt":"授权事实"})
    x=f.v34_add_assertion({"subject_id":a["entity_id"],"predicate":"负责人","object_value":"小林","object_entity_id":b["entity_id"],"evidence_ids":[ev["evidence_id"]],"verification_status":"USER_CONFIRMED"})
    y=f.v34_add_assertion({"subject_id":a["entity_id"],"predicate":"负责人","object_value":"小赵","evidence_ids":[ev["evidence_id"]]})
    assert f.graph.conflicts() and f.graph.query("蓝盒")["assertions"]
    f.graph.revoke_assertion(x["assertion_id"]); assert all(z["assertion_id"]!=x["assertion_id"] for z in f.graph.query("蓝盒")["assertions"])

def test_experience_is_scoped_and_persistent(tmp_path):
    f=V3Facade(tmp_path); rec=f.graph.add_experience({"task_family":"json_extract","verification_status":"VERIFIED_WITHIN_SCOPE","action_or_method":"whitelisted extractor"}); assert f.graph.experiences("json_extract")[0]["experience_id"]==rec["experience_id"]; f.close(); g=V3Facade(tmp_path); assert g.graph.experiences()
