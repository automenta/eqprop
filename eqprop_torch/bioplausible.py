"""
Bio-Plausible Learning Algorithms

Direct imports of research algorithms as first-class PyTorch models.
All algorithms inherit from BaseAlgorithm (nn.Module) and provide both
standard forward() and custom train_step() methods.
"""

import sys
from pathlib import Path

# Ensure algorithms/ is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    # Import all algorithm classes directly
    from algorithms import (
        BaseAlgorithm,
        BackpropBaseline,
        StandardEqProp,
        StandardFA,
        EquilibriumAlignment,
        AdaptiveFeedbackAlignment,
        ContrastiveFeedbackAlignment,
        LayerwiseEquilibriumFA,
        PredictiveCodingHybrid,
        EnergyGuidedFA,
        SparseEquilibrium,
        MomentumEquilibrium,
        StochasticFA,
        EnergyMinimizingFA,
        ALGORITHM_REGISTRY,
    )
    HAS_BIOPLAUSIBLE = True
except ImportError as e:
    HAS_BIOPLAUSIBLE = False
    ALGORITHM_REGISTRY = {}


# Re-export all algorithms at module level
__all__ = [
    'BaseAlgorithm',
    'BackpropBaseline', 
    'StandardEqProp',
    'StandardFA',
    'EquilibriumAlignment',
    'AdaptiveFeedbackAlignment',
    'ContrastiveFeedbackAlignment',
    'LayerwiseEquilibriumFA',
    'PredictiveCodingHybrid',
    'EnergyGuidedFA',
    'SparseEquilibrium',
    'MomentumEquilibrium',
    'StochasticFA',
    'EnergyMinimizingFA',
    'HAS_BIOPLAUSIBLE',
    'ALGORITHM_REGISTRY',
]
