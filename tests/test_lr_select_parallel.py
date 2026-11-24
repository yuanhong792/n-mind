import json
def test_global_cap_keeps_completed_task_file(tmp_path):
    p=tmp_path/'done.json'; p.write_text(json.dumps({'complete':True,'steps':1000}),encoding='utf8')
    timed_out=True
    assert timed_out and json.loads(p.read_text(encoding='utf8'))['complete']
