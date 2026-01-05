"""
Synapse Heatmap: Multi-channel neuron state visualization.

RGB Channel Mapping:
- Red: Activation magnitude |s| - which neurons are "awake"
- Green: Equilibrium velocity Δs_t - which neurons are "settling"
- Blue: Nudge magnitude (s_nudged - s_free) - credit assignment flow

When Green channel dims → network has reached equilibrium (fixed point).
Blue channel visualizes gradients "bleeding" backward through layers.
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import math


@dataclass
class LayerState:
    """Captured state for a single layer at one timestep."""
    activation: torch.Tensor          # s_t: current activation
    velocity: Optional[torch.Tensor] = None   # Δs = s_t - s_{t-1}
    nudge: Optional[torch.Tensor] = None      # s_nudged - s_free
    lipschitz_violation: Optional[torch.Tensor] = None  # |val| > threshold


@dataclass
class DynamicsCapture:
    """Captures network dynamics during training for visualization.
    
    Attaches hooks to model to record:
    - Activation states at each step
    - Velocity (state change between steps)
    - Nudge magnitudes (difference between free and nudged phases)
    """
    enabled: bool = True
    history: List[Dict[str, LayerState]] = field(default_factory=list)
    max_history: int = 100
    
    # Phase tracking
    free_equilibrium: Optional[Dict[str, torch.Tensor]] = None
    nudged_equilibrium: Optional[Dict[str, torch.Tensor]] = None
    
    def clear(self):
        """Clear captured history."""
        self.history = []
        self.free_equilibrium = None
        self.nudged_equilibrium = None
    
    def record_step(self, layer_name: str, h_new: torch.Tensor, h_old: torch.Tensor):
        """Record a single forward step for a layer."""
        if not self.enabled:
            return
        
        velocity = h_new - h_old
        state = LayerState(
            activation=h_new.detach().clone(),
            velocity=velocity.detach().clone()
        )
        
        if len(self.history) == 0 or layer_name not in self.history[-1]:
            self.history.append({})
        
        self.history[-1][layer_name] = state
        
        # Trim history if needed
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def record_free_equilibrium(self, states: Dict[str, torch.Tensor]):
        """Record the free phase equilibrium state."""
        self.free_equilibrium = {k: v.detach().clone() for k, v in states.items()}
    
    def record_nudged_equilibrium(self, states: Dict[str, torch.Tensor]):
        """Record the nudged phase equilibrium state."""
        self.nudged_equilibrium = {k: v.detach().clone() for k, v in states.items()}
        
        # Compute nudge magnitudes
        if self.free_equilibrium is not None:
            for k in self.nudged_equilibrium:
                if k in self.free_equilibrium:
                    nudge = self.nudged_equilibrium[k] - self.free_equilibrium[k]
                    if self.history and k in self.history[-1]:
                        self.history[-1][k].nudge = nudge


class SynapseHeatmap:
    """Generate RGB heatmaps from captured network dynamics.
    
    Reshapes hidden layers into 2D grids for visualization:
    - 1024 neurons → 32×32 grid
    - 256 neurons → 16×16 grid
    
    RGB channels:
    - R: Activation magnitude (normalized)
    - G: Velocity magnitude (dims as network settles)
    - B: Nudge magnitude (credit assignment visualization)
    """
    
    def __init__(self, 
                 lipschitz_threshold: float = 1.0,
                 velocity_scale: float = 10.0,
                 nudge_scale: float = 5.0):
        """
        Args:
            lipschitz_threshold: Value above which neurons are marked as violating stability
            velocity_scale: Multiplier for velocity channel (green) visibility
            nudge_scale: Multiplier for nudge channel (blue) visibility
        """
        self.lipschitz_threshold = lipschitz_threshold
        self.velocity_scale = velocity_scale
        self.nudge_scale = nudge_scale
    
    def compute_grid_size(self, num_neurons: int) -> Tuple[int, int]:
        """Compute nearest square grid for neuron count."""
        side = int(math.ceil(math.sqrt(num_neurons)))
        return (side, side)
    
    def reshape_to_grid(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reshape [batch, neurons] to [batch, H, W] grid.
        
        Pads with zeros if not a perfect square.
        """
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        
        batch_size, num_neurons = tensor.shape
        h, w = self.compute_grid_size(num_neurons)
        
        # Pad if needed
        total = h * w
        if num_neurons < total:
            padding = torch.zeros(batch_size, total - num_neurons, 
                                 device=tensor.device, dtype=tensor.dtype)
            tensor = torch.cat([tensor, padding], dim=1)
        
        return tensor.view(batch_size, h, w)
    
    def activation_to_red(self, activation: torch.Tensor) -> np.ndarray:
        """Convert activation magnitude to red channel [0, 255]."""
        grid = self.reshape_to_grid(activation)
        # Take mean across batch
        grid = grid.mean(dim=0).abs()
        # Normalize to [0, 1]
        max_val = grid.max() + 1e-8
        normalized = (grid / max_val).detach().cpu().numpy()
        return (normalized * 255).astype(np.uint8)
    
    def velocity_to_green(self, velocity: torch.Tensor) -> np.ndarray:
        """Convert velocity magnitude to green channel [0, 255].
        
        Bright green = still settling
        Dark green = at equilibrium
        """
        grid = self.reshape_to_grid(velocity)
        grid = grid.mean(dim=0).abs() * self.velocity_scale
        # Clamp and normalize
        grid = torch.clamp(grid, 0, 1).detach().cpu().numpy()
        return (grid * 255).astype(np.uint8)
    
    def nudge_to_blue(self, nudge: torch.Tensor) -> np.ndarray:
        """Convert nudge magnitude to blue channel [0, 255].
        
        Bright blue = strong credit assignment signal
        Dark blue = no gradient flow
        """
        grid = self.reshape_to_grid(nudge)
        grid = grid.mean(dim=0).abs() * self.nudge_scale
        grid = torch.clamp(grid, 0, 1).detach().cpu().numpy()
        return (grid * 255).astype(np.uint8)
    
    def compute_stability_overlay(self, activation: torch.Tensor) -> np.ndarray:
        """Compute white overlay where Lipschitz bound is violated.
        
        Returns boolean mask of shape [H, W].
        """
        grid = self.reshape_to_grid(activation)
        grid = grid.mean(dim=0).abs()
        violations = (grid > self.lipschitz_threshold).detach().cpu().numpy()
        return violations
    
    def generate_rgb(self, state: LayerState) -> np.ndarray:
        """Generate RGB image from layer state.
        
        Returns:
            np.ndarray of shape [H, W, 3] with dtype uint8
        """
        # Red: activation
        red = self.activation_to_red(state.activation)
        h, w = red.shape
        
        # Green: velocity (or zeros if not available)
        if state.velocity is not None:
            green = self.velocity_to_green(state.velocity)
        else:
            green = np.zeros((h, w), dtype=np.uint8)
        
        # Blue: nudge (or zeros if not available)
        if state.nudge is not None:
            blue = self.nudge_to_blue(state.nudge)
        else:
            blue = np.zeros((h, w), dtype=np.uint8)
        
        # Stack to RGB
        rgb = np.stack([red, green, blue], axis=-1)
        
        # Apply stability overlay (white for violations)
        violations = self.compute_stability_overlay(state.activation)
        rgb[violations] = [255, 255, 255]
        
        return rgb
    
    def generate_multi_layer(self, 
                             states: Dict[str, LayerState],
                             target_size: int = 128) -> np.ndarray:
        """Generate side-by-side RGB heatmaps for multiple layers.
        
        Args:
            states: Dict mapping layer names to LayerState
            target_size: Target pixel size for each layer's heatmap
        
        Returns:
            np.ndarray of shape [target_size, N*target_size, 3]
        """
        import cv2  # Optional, for resize
        
        layer_images = []
        for name in sorted(states.keys()):
            state = states[name]
            rgb = self.generate_rgb(state)
            
            # Resize to target size
            if rgb.shape[0] != target_size or rgb.shape[1] != target_size:
                rgb = cv2.resize(rgb, (target_size, target_size), 
                                interpolation=cv2.INTER_NEAREST)
            
            layer_images.append(rgb)
        
        if not layer_images:
            return np.zeros((target_size, target_size, 3), dtype=np.uint8)
        
        return np.concatenate(layer_images, axis=1)
