from nmind.evaluation.phase2_validity import task_validity_check
def test_validity_gate_rejects_shortcut_task():
    r=task_validity_check({'name':'s0','oracle_em':1,'random_level':.25,'shortcut_ems':{'recent':1.0},'effective_samples':16,'combination_overlap':16,'counterfactual_pairs_present':False,'unk_rate':0})
    assert not r['valid'] and 'shortcuts_le_random_plus_005' in r['failed']
def test_validity_gate_accepts_control_shape():
    r=task_validity_check({'name':'large','oracle_em':1,'random_level':.05,'shortcut_ems':{'recent':.07,'majority':.05,'fixed':.05,'random':.05},'effective_samples':500,'combination_overlap':0,'counterfactual_pairs_present':True,'unk_rate':0})
    assert r['valid']
