"""N-Runtime：硬件探测、计算治理、纵向闭环引擎。"""

from .engine import GRID, NMindEngine, Outcome, SequenceResult
from .governor import BudgetDecision, ComputeGovernor, GovernorReport
from .hardware import HardwareReport, cpu_features, detect

__all__ = ["NMindEngine", "Outcome", "SequenceResult", "ComputeGovernor", "BudgetDecision",
           "GovernorReport", "HardwareReport", "detect", "cpu_features", "GRID"]
