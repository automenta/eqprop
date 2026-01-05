"""EqProp Kernel Package - Pure NumPy/CuPy Equilibrium Propagation."""

from .eqprop_kernel import (
    EqPropKernel,
    spectral_normalize,
    get_backend,
    to_numpy,
    HAS_CUPY,
)

__all__ = [
    'EqPropKernel',
    'spectral_normalize',
    'get_backend',
    'to_numpy',
    'HAS_CUPY',
]
