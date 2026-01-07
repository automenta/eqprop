"""
EqProp+SN Evolution System

A genetic algorithm-based system for evolving and evaluating 
Equilibrium Propagation + Spectral Normalization model variants.
"""

from .fitness import FitnessScore, compute_fitness
from .breeder import VariationBreeder, ArchConfig
from .evaluator import VariationEvaluator, EvalTier
from .breakthrough import BreakthroughDetector
from .engine import EvolutionEngine

__all__ = [
    'FitnessScore', 'compute_fitness',
    'VariationBreeder', 'ArchConfig',
    'VariationEvaluator', 'EvalTier',
    'BreakthroughDetector',
    'EvolutionEngine',
]
