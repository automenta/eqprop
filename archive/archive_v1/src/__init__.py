"""TorEqProp: Toroidal Equilibrium Propagation for Transformers.

This package implements equilibrium propagation for transformer architectures,
with support for both standard and symmetric (energy-based) modes.

Main components:
    - models: LoopedTransformerBlock for equilibrium iteration
    - attention: Modular attention mechanisms (softmax, linear, symmetric)
    - ffn: Feed-forward network modules (standard, symmetric)  
    - solver: Fixed-point equilibrium solver
    - trainer: EqProp training loop with configurable update strategies
    - updates: Update strategies (MSE proxy, vector field)
    - utils: Utility functions (linear attention, compute tier detection)
"""

from .models import LoopedTransformerBlock
from .simplified_models import (
    LoopedMLP, ToroidalMLP, HopfieldEqProp, ConvEqProp, ResidualEqProp, GatedEqProp
)
from .attention import SoftmaxAttention, LinearAttention, SymmetricLinearAttention
from .ffn import StandardFFN, SymmetricFFN
from .solver import EquilibriumSolver
from .trainer import EqPropTrainer
from .updates import UpdateStrategy, MSEProxyUpdate, VectorFieldUpdate

__all__ = [
    # Models
    'LoopedTransformerBlock',
    # Attention
    'SoftmaxAttention',
    'LinearAttention',
    'SymmetricLinearAttention',
    # FFN
    'StandardFFN',
    'SymmetricFFN',
    # Training
    'EquilibriumSolver',
    'EqPropTrainer',
    # Update strategies
    'UpdateStrategy',
    'MSEProxyUpdate',
    'VectorFieldUpdate',
    # Simplified Models
    'LoopedMLP',
    'HopfieldEqProp',
    'ConvEqProp',
    'ResidualEqProp',
    'GatedEqProp',
]
