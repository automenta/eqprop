"""Analysis module for per-iteration model validation."""

from .trajectory import TrajectoryRecorder, Trajectory
from .metrics import IterationMetrics, compute_iteration_metrics
from .iteration_analyzer import IterationAnalyzer, AnalysisReport
from .synthetic_tasks import (
    SyntheticTask, IdentityTask, LinearTask, XORTask, 
    MemorizationTask, AttractorTask, get_all_tasks
)
from .theoretical import TheoreticalValidator

__all__ = [
    "TrajectoryRecorder", "Trajectory",
    "IterationMetrics", "compute_iteration_metrics",
    "IterationAnalyzer", "AnalysisReport",
    "SyntheticTask", "IdentityTask", "LinearTask", "XORTask",
    "MemorizationTask", "AttractorTask", "get_all_tasks",
    "TheoreticalValidator"
]
