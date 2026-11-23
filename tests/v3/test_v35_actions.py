from nmind.planning.v35_store import V35PlanStore
from nmind.planning.safe_actions import SafeActionRegistry

def test_plan_store_requires_explicit_approval(tmp_path):
    s=V35PlanStore(tmp_path/'p.db'); s.save('p',{'plan':[]}); assert s.get('p')['approved']==0; assert s.set_status('p','APPROVED',True)['approved']==1

def test_safe_action_rejects_escape_and_reads_only_workspace(tmp_path):
    (tmp_path/'note.txt').write_text('hello',encoding='utf-8'); a=SafeActionRegistry(workspace=tmp_path)
    assert a.read_file('读取 note.txt',{})['verified'] is True
    try: a.read_file('读取 ../secret.txt',{})
    except PermissionError: pass
    else: raise AssertionError('path escape must be rejected')
