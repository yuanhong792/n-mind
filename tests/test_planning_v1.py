from nmind.planning import GoalSpec, PlanExecutor, PlanningStore, RulePlanner
from nmind.planning.tool_adapter import build_tools

def test_rule_planner_builds_acyclic_bounded_plan(tmp_path):
    g=GoalSpec.create("比较列表和元组", budget={"max_steps":8,"max_tool_calls":12})
    p=RulePlanner().plan(g)
    assert p.source == "rule" and not p.validation_errors
    assert len(p.steps) <= 8

def test_executor_dependency_order_and_persistence(tmp_path):
    g=GoalSpec.create("计算 2 + 3", budget={"max_steps":8,"max_tool_calls":12})
    st=PlanningStore(tmp_path/"planning.db")
    out=PlanExecutor(st,build_tools()).run(g,RulePlanner().plan(g))
    assert out["status"] == "SUCCEEDED"
    events=st.events(g.goal_id)
    assert any(e["kind"]=="step_result" for e in events)

def test_executor_rejects_unsupported_tool(tmp_path):
    g=GoalSpec.create("查询", budget={"max_steps":8,"max_tool_calls":12})
    p=RulePlanner().plan(g)
    bad=p.__class__(p.task_id,p.goal_id,p.version,p.source,tuple([p.steps[0].__class__(**{**p.steps[0].__dict__,"tool_id":"shell"})]))
    out=PlanExecutor(PlanningStore(tmp_path/"p.db"),{}).run(g,bad)
    assert out["status"] == "BLOCKED"
