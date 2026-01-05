"""
TorEq Dynamic Observatory (TDO)

Real-time visualization of Equilibrium Propagation network dynamics.
Transform training from "fitting a model" to "observing a dynamical system."

Core components:
- SynapseHeatmap: RGB multi-channel neuron state visualization
- ObservatoryRenderer: PyGame-based real-time display
- DynamicsCapture: Hooks for capturing model states during training
"""

from .heatmap import SynapseHeatmap, DynamicsCapture
from .metrics import ObservatoryMetrics
from .renderer import ObservatoryRenderer, HeadlessRenderer, RendererConfig

__all__ = [
    'SynapseHeatmap', 
    'DynamicsCapture', 
    'ObservatoryMetrics',
    'ObservatoryRenderer',
    'HeadlessRenderer',
    'RendererConfig',
]
