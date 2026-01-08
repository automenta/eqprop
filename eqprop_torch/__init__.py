"""
EqProp-Torch: Equilibrium Propagation for PyTorch

A production-grade library for training neural networks with Equilibrium Propagation,
featuring spectral normalization for stability and torch.compile for acceleration.
"""

from .core import EqPropTrainer
from .models import (
    LoopedMLP,
    BackpropMLP,
    ConvEqProp,
    TransformerEqProp,
)
from .kernel import EqPropKernel, HAS_CUPY
from .acceleration import compile_model, get_optimal_backend, check_cupy_available
from .datasets import get_vision_dataset, get_lm_dataset, CharDataset, create_data_loaders
from .utils import (
    export_to_onnx,
    count_parameters,
    verify_spectral_norm,
    create_model_preset,
    ModelRegistry,
)

# Language model variants (optional import - fails gracefully if dependencies missing)
try:
    from .lm_models import (
        get_eqprop_lm,
        list_eqprop_lm_variants,
        create_eqprop_lm,
        EQPROP_LM_REGISTRY,
    )
    HAS_LM_VARIANTS = True
except ImportError:
    HAS_LM_VARIANTS = False
    get_eqprop_lm = None
    list_eqprop_lm_variants = None

__version__ = "0.1.0"
__all__ = [
    # Trainer
    "EqPropTrainer",
    # Models
    "LoopedMLP",
    "BackpropMLP", 
    "ConvEqProp",
    "TransformerEqProp",
    # Kernel
    "EqPropKernel",
    "HAS_CUPY",
    # Utils
    "compile_model",
    "get_optimal_backend",
    "check_cupy_available",
    # Datasets
    "get_vision_dataset",
    "get_lm_dataset",
    "CharDataset",
    "create_data_loaders",
    # Utils
    "export_to_onnx",
    "count_parameters",
    "verify_spectral_norm",
    "create_model_preset",
    "ModelRegistry",
    # LM variants (if available)
    "get_eqprop_lm",
    "list_eqprop_lm_variants",
    "create_eqprop_lm",
    "HAS_LM_VARIANTS",
]


