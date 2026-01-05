"""
TEP Experiment: Rigorous evaluation of Toroidal Equilibrium Propagation.

This package implements the experiment specification for fair, systematic
comparison between TEP and Backpropagation (BP).

Key Features:
- Identical treatment: TEP and BP receive same optimization budget
- Multi-objective: Pareto fronts over accuracy, time, params, convergence
- Phased gating: Automatic go/no-go decisions based on success criteria
- Statistical rigor: Multi-seed evaluation with significance testing

Usage:
    # Quick smoke test
    python -m tep_experiment --smoke-test
    
    # Phase 1: Rapid Signal Detection (6-10 hours)
    python -m tep_experiment --phase 1
    
    # Full pipeline with all phases
    python -m tep_experiment --full

See README.md for detailed documentation.
"""

from .config import (
    SharedSearchSpace,
    TEPSearchSpace,
    BPSearchSpace,
    PhaseConfig,
    TrialResult,
    PHASE_CONFIGS,
)
from .engine import TEPExperimentEngine
from .analysis import ParetoFront, ParetoAnalyzer

__version__ = "0.1.0"
__all__ = [
    "SharedSearchSpace",
    "TEPSearchSpace", 
    "BPSearchSpace",
    "PhaseConfig",
    "TrialResult",
    "PHASE_CONFIGS",
    "TEPExperimentEngine",
    "ParetoFront",
    "ParetoAnalyzer",
]
