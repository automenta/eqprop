"""TorEqProp Release Models Package"""

from .looped_mlp import LoopedMLP, BackpropMLP
from .ternary import TernaryEqProp
from .neural_cube import NeuralCube
from .lazy_eqprop import LazyEqProp, LazyStats
from .feedback_alignment import FeedbackAlignmentEqProp, FeedbackAlignmentLayer
from .kernel import EqPropKernel, compare_memory_autograd_vs_kernel

# Newly ported models
from .temporal_resonance import TemporalResonanceEqProp
from .homeostatic import HomeostaticEqProp
from .conv_eqprop import ConvEqProp
from .modern_conv_eqprop import ModernConvEqProp, SimpleConvEqProp
from .transformer import TransformerEqProp, EqPropAttention
from .causal_transformer_eqprop import CausalTransformerEqProp
from .eqprop_diffusion import EqPropDiffusion

__all__ = [
    'LoopedMLP', 'BackpropMLP',
    'TernaryEqProp',
    'NeuralCube',
    'LazyEqProp', 'LazyStats',
    'FeedbackAlignmentEqProp', 'FeedbackAlignmentLayer',
    'EqPropKernel', 'compare_memory_autograd_vs_kernel',
    'TemporalResonanceEqProp',
    'HomeostaticEqProp',
    'ConvEqProp',
    'ModernConvEqProp', 'SimpleConvEqProp',
    'TransformerEqProp', 'EqPropAttention',
    'CausalTransformerEqProp',
    'EqPropDiffusion',
]
