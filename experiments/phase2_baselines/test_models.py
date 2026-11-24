import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from models import GRUBaseline,CausalTransformerBaseline,DeltaRuleLinearBaseline,parameter_count
def test_baselines_construct_and_count():
    for cls in (GRUBaseline,CausalTransformerBaseline,DeltaRuleLinearBaseline):
        assert parameter_count(cls(32)) > 0
