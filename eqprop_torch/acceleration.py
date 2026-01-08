"""
EqProp-Torch Acceleration Utilities

Provides torch.compile wrappers for 2-3x speedup and optional CuPy/Triton kernels.
Designed for portability: works on CPU, CUDA, ROCm, and Apple MPS.
"""

import torch
import warnings
from typing import Optional, Tuple


def get_optimal_backend() -> str:
    """
    Detect best available compute backend.
    
    Returns:
        'cuda' | 'mps' | 'cpu'
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def enable_tf32(enable: bool = True):
    """
    Enable TensorFloat-32 (TF32) for significant speedup on Ampere+ GPUs.
    
    TF32 reduces precision slightly (19 bits vs 24 bits significand) 
    but maintains full range, usually providing 2-3x speedup for 
    matmul and convolutions with negligible accuracy loss.
    
    Args:
        enable: Whether to enable TF32
    """
    if torch.cuda.is_available():
        # High precision = TF32 enabled
        # Highest precision = TF32 disabled (slow)
        precision = 'high' if enable else 'highest'
        torch.backends.cuda.matmul.allow_tf32 = enable
        torch.backends.cudnn.allow_tf32 = enable
        torch.set_float32_matmul_precision(precision)


def check_cupy_available() -> Tuple[bool, str]:
    """
    Check if CuPy is available with proper CUDA configuration.
    
    Returns:
        (available: bool, message: str)
    """
    try:
        import cupy as cp
        # Try a simple operation to verify CUDA works
        _ = cp.zeros(10)
        return True, "CuPy available with CUDA"
    except ImportError:
        return False, "CuPy not installed. Install with: pip install cupy-cuda12x"
    except Exception as e:
        return False, f"CuPy installed but CUDA failed: {e}"


def compile_model(
    model: torch.nn.Module, 
    mode: str = "reduce-overhead",
    fullgraph: bool = False,
    dynamic: bool = None,
) -> torch.nn.Module:
    """
    Wrap model with torch.compile for significant speedup.
    
    Works on CPU, CUDA, ROCm, and MPS without modification.
    Falls back gracefully if torch.compile is unavailable.
    
    Args:
        model: PyTorch model to compile
        mode: Compilation mode:
            - 'default': Balanced speed and compile time
            - 'reduce-overhead': Minimize GPU kernel launch overhead
            - 'max-autotune': Maximum speed (longer compile)
        fullgraph: If True, requires entire forward to be capturable
        dynamic: Enable dynamic shapes (None = auto-detect)
    
    Returns:
        Compiled model (or original if compile unavailable)
    
    Example:
        >>> model = LoopedMLP(784, 256, 10)
        >>> model = compile_model(model, mode='reduce-overhead')
    """
    # Check PyTorch version
    if not hasattr(torch, 'compile'):
        warnings.warn(
            "torch.compile not available (requires PyTorch 2.0+). "
            "Using uncompiled model.",
            RuntimeWarning
        )
        return model
    
    try:
        compiled = torch.compile(
            model, 
            mode=mode, 
            fullgraph=fullgraph,
            dynamic=dynamic,
        )
        return compiled
    except Exception as e:
        warnings.warn(
            f"torch.compile failed: {e}. Using uncompiled model.",
            RuntimeWarning
        )
        return model


def compile_settling_loop(settling_fn):
    """
    Decorator to compile the inner settling loop for maximum speed.
    
    Use this on the forward_step method of EqProp models:
    
        @compile_settling_loop
        def forward_step(self, h, x_emb):
            ...
    """
    if not hasattr(torch, 'compile'):
        return settling_fn
    
    try:
        return torch.compile(settling_fn, mode='reduce-overhead')
    except Exception:
        return settling_fn


# =============================================================================
# Optional Triton Kernel Stub
# =============================================================================

TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    pass


def check_triton_available() -> Tuple[bool, str]:
    """Check if Triton is available for custom kernels."""
    if TRITON_AVAILABLE:
        return True, "Triton available"
    return False, "Triton not installed. Install with: pip install triton"


# Triton kernel example (commented - requires Triton and CUDA/ROCm)
# @triton.jit
# def settling_kernel(
#     h_ptr, x_proj_ptr, W_rec_ptr, 
#     hidden_dim: tl.constexpr, 
#     BLOCK_SIZE: tl.constexpr
# ):
#     """Fused settling iteration kernel."""
#     pid = tl.program_id(0)
#     # ... implementation for fused tanh(x_proj + W_rec @ h)


__all__ = [
    'get_optimal_backend',
    'check_cupy_available',
    'check_triton_available',
    'compile_model',
    'compile_settling_loop',
    'TRITON_AVAILABLE',
]
