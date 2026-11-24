from nmind.language.large_binding_task import generate_tasks,split_by_combo
def test_large_binding_is_vocab_closed_and_combo_heldout():
    rows=generate_tasks(600,tokenizer_path='experiments/m5_language/tokenizer_m52.json')
    assert len(rows)>=500 and len({r['fingerprint'] for r in rows})>=500
    tr,te=split_by_combo(rows); assert {r['combo'] for r in tr}.isdisjoint({r['combo'] for r in te})
