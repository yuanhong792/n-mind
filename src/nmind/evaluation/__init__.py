"""评测层：任务套件、基线、CPU 基准、消融。

这一层只负责"量"，不负责"说"：所有结论文字来自实测数字，
且必须与 docs/LIMITATIONS.md 的口径一致（能力上限、能耗未测量、不外推到 GPU）。
"""

from .ablations import (
    DEFAULT_VARIANTS,
    Variant,
    make_episode_memory_provider,
    run_ablations,
    run_cloud_ablation,
    run_memory_ablation,
    run_structural_ablations,
    write_reports,
)
from .baselines import GruBackbone, SharedTrunkCell, build_dense, run_baselines
from .benchmarks import cpu_benchmark, measure
from .tasks import TaskSuite, budget_difficulty_report, decision_metrics, memory_span_curve, readout_probe

__all__ = [
    "TaskSuite", "readout_probe", "memory_span_curve", "decision_metrics",
    "budget_difficulty_report",
    "GruBackbone", "SharedTrunkCell", "build_dense", "run_baselines",
    "cpu_benchmark", "measure",
    "Variant", "DEFAULT_VARIANTS", "make_episode_memory_provider",
    "run_structural_ablations", "run_memory_ablation", "run_cloud_ablation",
    "run_ablations", "write_reports",
]
