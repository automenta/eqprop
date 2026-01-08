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
from .kernel import EqPropKernel
from .acceleration import compile_model, get_optimal_backend, check_cupy_available
from .datasets import get_vision_dataset, get_lm_dataset, CharDataset, create_data_loaders

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
    # Utils
    "compile_model",
    "get_optimal_backend",
    "check_cupy_available",
    # Datasets
    "get_vision_dataset",
    "get_lm_dataset",
    "CharDataset",
    "create_data_loaders",
]

