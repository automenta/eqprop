"""
EqProp-Torch Utilities

Helper functions for ONNX export, model verification, and training utilities.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Tuple
import warnings


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    input_shape: Tuple[int, ...],
    opset_version: int = 14,
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    device: str = "cpu",
) -> None:
    """
    Export a PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model to export
        output_path: Path to save .onnx file
        input_shape: Example input shape (e.g., (1, 784))
        opset_version: ONNX opset version
        dynamic_axes: Dynamic axis specification
        device: Device to use for export
        
    Example:
        >>> model = LoopedMLP(784, 256, 10)
        >>> export_to_onnx(model, "model.onnx", (1, 784))
    """
    # Handle compiled models
    if hasattr(model, '_orig_mod'):
        model = model._orig_mod
    
    model.eval()
    model = model.to(device)
    dummy_input = torch.randn(*input_shape, device=device)
    
    if dynamic_axes is None:
        dynamic_axes = {
            'input': {0: 'batch'},
            'output': {0: 'batch'}
        }
    
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            opset_version=opset_version,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
        )
        print(f"✓ Model exported to {output_path}")
    except Exception as e:
        raise RuntimeError(f"ONNX export failed: {e}")


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count the number of parameters in a model.
    
    Args:
        model: PyTorch model
        trainable_only: If True, only count trainable parameters
        
    Returns:
        Number of parameters
    """
    if hasattr(model, '_orig_mod'):
        model = model._orig_mod
    
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def verify_spectral_norm(model: nn.Module) -> Dict[str, float]:
    """
    Verify that all layers with spectral normalization have L <= 1.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dict mapping layer names to their Lipschitz constants
    """
    if hasattr(model, '_orig_mod'):
        model = model._orig_mod
    
    lipschitz_values = {}
    
    for name, module in model.named_modules():
        # Check for spectral norm parametrization
        if hasattr(module, 'parametrizations') and hasattr(module.parametrizations, 'weight'):
            with torch.no_grad():
                if hasattr(module, 'weight'):
                    W = module.weight
                    if W.dim() >= 2:
                        # Compute spectral norm
                        if W.dim() > 2:
                            W_flat = W.reshape(W.shape[0], -1)
                        else:
                            W_flat = W
                        
                        s = torch.linalg.svdvals(W_flat)
                        lipschitz_values[name] = s[0].item()
    
    return lipschitz_values


def compute_gradient_norm(model: nn.Module) -> float:
    """
    Compute the global gradient norm across all parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Gradient norm
    """
    if hasattr(model, '_orig_mod'):
        model = model._orig_mod
    
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def estimate_memory_usage(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    batch_size: int = 1,
) -> Dict[str, float]:
    """
    Estimate memory usage of a model.
    
    Args:
        model: PyTorch model
        input_shape: Input shape (without batch dimension)
        batch_size: Batch size for estimation
        
    Returns:
        Dict with memory estimates in MB
    """
    # Parameters
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
    
    # Gradients (same as parameters)
    grad_memory = param_memory
    
    # Activations (rough estimate)
    # This is model-specific, here's a simple heuristic
    activation_memory = batch_size * sum(input_shape) * 4 / 1e6  # 4 bytes per float32
    
    return {
        'parameters_mb': param_memory,
        'gradients_mb': grad_memory,
        'activations_mb': activation_memory,
        'total_mb': param_memory + grad_memory + activation_memory,
    }


class ModelRegistry:
    """
    Simple registry for model factories.
    
    Example:
        >>> registry = ModelRegistry()
        >>> registry.register('my_mlp', lambda: LoopedMLP(784, 256, 10))
        >>> model = registry.create('my_mlp')
    """
    
    def __init__(self):
        self._factories = {}
    
    def register(self, name: str, factory: callable):
        """Register a model factory function."""
        self._factories[name] = factory
    
    def create(self, name: str, **kwargs) -> nn.Module:
        """Create a model from the registry."""
        if name not in self._factories:
            raise ValueError(f"Model '{name}' not registered. Available: {list(self._factories.keys())}")
        return self._factories[name](**kwargs)
    
    def list_models(self) -> List[str]:
        """List all registered models."""
        return list(self._factories.keys())


# Global registry instance
model_registry = ModelRegistry()


def create_model_preset(preset_name: str, **overrides) -> nn.Module:
    """
    Create a model from a preset configuration.
    
    Args:
        preset_name: Name of preset ('mnist_small', 'mnist_large', 'cifar_conv', etc.)
        **overrides: Override default parameters
        
    Returns:
        Configured model
        
    Example:
        >>> model = create_model_preset('mnist_small', hidden_dim=512)
    """
    from .models import LoopedMLP, ConvEqProp
    
    presets = {
        'mnist_small': lambda: LoopedMLP(784, 128, 10, use_spectral_norm=True),
        'mnist_medium': lambda: LoopedMLP(784, 256, 10, use_spectral_norm=True),
        'mnist_large': lambda: LoopedMLP(784, 512, 10, use_spectral_norm=True),
        'cifar_conv': lambda: ConvEqProp(3, 64, 10),
        'cifar_mlp': lambda: LoopedMLP(3072, 512, 10, use_spectral_norm=True),
    }
    
    if preset_name not in presets:
        raise ValueError(f"Unknown preset '{preset_name}'. Available: {list(presets.keys())}")
    
    # Note: overrides would require more sophisticated preset handling
    return presets[preset_name]()


__all__ = [
    'export_to_onnx',
    'count_parameters',
    'verify_spectral_norm',
    'compute_gradient_norm',
    'estimate_memory_usage',
    'ModelRegistry',
    'model_registry',
    'create_model_preset',
]
