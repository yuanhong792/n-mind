import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from models import GRUBaseline,CausalTransformerBaseline,DeltaRuleLinearBaseline,parameter_count,assert_reference_match
def test_baselines_construct_and_count():
    for cls in (GRUBaseline,CausalTransformerBaseline,DeltaRuleLinearBaseline):
        assert parameter_count(cls(32)) > 0
    counts=assert_reference_match({'gru':GRUBaseline(32),'transformer':CausalTransformerBaseline(32),'delta':DeltaRuleLinearBaseline(32)})
    assert all(412825 <= n <= 456280 for n in counts.values())
