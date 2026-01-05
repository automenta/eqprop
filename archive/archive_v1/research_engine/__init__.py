"""
TorEqProp Unified Research Engine

A complete, transparent autonomous research system that consolidates
all experiment infrastructure into a single, coherent package.

Components:
    - dashboard: Live TUI with full parameter transparency
    - runner: Time-aware experiment execution
    - collector: Structured result storage
    - analyzer: Parameter sensitivity & ANOVA analysis
    - scheduler: Progressive validation with intelligent selection
    - reporter: Publication-ready report generation
    - visualizations: Heatmaps and parameter space plots
"""

from .config import ResearchConfig, ExperimentPreset, ValidationTier, TIERS, TIER_ORDER
from .collector import ResultCollector, Trial
from .runner import TimeAwareRunner
from .scheduler import ProgressiveScheduler
from .analyzer import ParameterAnalyzer
from .reporter import ResearchReporter
from .dashboard import ResearchDashboard

__all__ = [
    "ResearchConfig",
    "ExperimentPreset", 
    "ValidationTier",
    "TIERS",
    "TIER_ORDER",
    "ResultCollector",
    "Trial",
    "TimeAwareRunner",
    "ProgressiveScheduler",
    "ParameterAnalyzer",
    "ResearchReporter",
    "ResearchDashboard",
]

__version__ = "1.0.0"
