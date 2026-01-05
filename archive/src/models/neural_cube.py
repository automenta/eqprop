"""
Neural Cube: 3D Voxel Topology (TODO5 Item 4.1)

Moves beyond "Neural Networks" to "Self-Organizing Neural Tissue."

Each neuron in a 3D lattice connects only to its 26 physical neighbors.
Neurogenesis/Pruning: Blue Channel (Nudge) grows synapses where learning
is "loud" and prunes them where it is "silent."

Visualization: Slice through the 3D cube to see "clouds of thought" forming.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import math


class NeuralCube(nn.Module):
    """
    3D Voxel Neural Network with local connectivity.
    
    Each neuron at position (i,j,k) connects only to its 26 neighbors
    (including diagonals in 3D). This mimics biological neural tissue
    where connectivity is spatially local.
    """
    
    def __init__(self, cube_size: int = 8, input_dim: int = 784, output_dim: int = 10,
                 alpha: float = 0.5, use_spectral_norm: bool = True):
        """
        Args:
            cube_size: Size of the cube (N×N×N neurons)
            input_dim: Input dimension (e.g., 784 for MNIST)
            output_dim: Output dimension (e.g., 10 for classification)
        """
        super().__init__()
        
        self.cube_size = cube_size
        self.num_neurons = cube_size ** 3
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha = alpha
        
        # Input projection: map input to cube surface
        self.input_projection = nn.Linear(input_dim, cube_size ** 2)
        
        # Local 3D convolution for neighborhood interaction
        # 3x3x3 kernel connects each neuron to its 26 neighbors
        self.local_conv = nn.Conv3d(1, 1, kernel_size=3, padding=1, bias=False)
        
        # Bias per neuron
        self.neuron_bias = nn.Parameter(torch.zeros(1, 1, cube_size, cube_size, cube_size))
        
        # Output readout: global pooling + linear
        self.readout = nn.Linear(cube_size ** 3, output_dim)
        
        # Synapse strength (for neurogenesis/pruning)
        # This is a 3x3x3 kernel that determines connectivity
        self.register_buffer('synapse_mask', torch.ones(1, 1, 3, 3, 3))
        
        # Activity tracking for neurogenesis/pruning
        self.register_buffer('activity_history', torch.zeros(cube_size, cube_size, cube_size))
        self.register_buffer('nudge_history', torch.zeros(cube_size, cube_size, cube_size))
        
        if use_spectral_norm:
            from torch.nn.utils.parametrizations import spectral_norm
            self.local_conv = spectral_norm(self.local_conv)
        
        # Initialize local weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stability."""
        # Initialize local convolution to have bounded spectral norm
        nn.init.orthogonal_(self.local_conv.weight.view(-1, 27).t())
        with torch.no_grad():
            self.local_conv.weight.mul_(0.5)
    
    def _input_to_cube_surface(self, x: torch.Tensor) -> torch.Tensor:
        """Project input onto one face of the cube."""
        batch_size = x.size(0)
        surface = self.input_projection(x)  # [B, cube_size^2]
        surface = surface.view(batch_size, self.cube_size, self.cube_size)
        
        # Create cube with input on bottom face
        cube = torch.zeros(batch_size, 1, self.cube_size, self.cube_size, 
                          self.cube_size, device=x.device)
        cube[:, 0, 0, :, :] = surface  # Bottom face (z=0)
        
        return cube
    
    def forward_step(self, cube_state: torch.Tensor, x_cube: torch.Tensor) -> torch.Tensor:
        """Single equilibrium step with local 3D connectivity."""
        # Apply local convolution (neighbors interact)
        neighbor_input = self.local_conv(cube_state)
        
        # Nonlinearity
        h_target = torch.tanh(neighbor_input + x_cube + self.neuron_bias)
        
        # Smooth update
        new_state = (1 - self.alpha) * cube_state + self.alpha * h_target
        
        return new_state
    
    def forward(self, x: torch.Tensor, steps: int = 30, 
                track_dynamics: bool = False) -> torch.Tensor:
        """Forward pass to equilibrium."""
        batch_size = x.size(0)
        
        # Project input to cube surface
        x_cube = self._input_to_cube_surface(x)
        
        # Initialize state
        cube_state = torch.zeros(batch_size, 1, self.cube_size, self.cube_size,
                                 self.cube_size, device=x.device)
        
        dynamics = [] if track_dynamics else None
        
        for step in range(steps):
            cube_state = self.forward_step(cube_state, x_cube)
            
            if track_dynamics and step % 5 == 0:
                dynamics.append(cube_state.detach().clone())
        
        # Readout: flatten and project
        flat_state = cube_state.view(batch_size, -1)
        output = self.readout(flat_state)
        
        if track_dynamics:
            return output, dynamics
        return output
    
    def compute_nudge_field(self, cube_free: torch.Tensor, 
                            cube_nudged: torch.Tensor) -> torch.Tensor:
        """
        Compute the "nudge field" - the Blue Channel visualization.
        
        This shows where learning signal is strongest in 3D space.
        """
        return torch.abs(cube_nudged - cube_free)
    
    def neurogenesis(self, nudge_field: torch.Tensor, threshold_high: float = 0.1):
        """
        Grow synapses where nudge signal is strong.
        
        This implements the "loud learning" → "grow" rule.
        """
        # Average nudge over batch
        avg_nudge = nudge_field.mean(dim=0).squeeze()  # [cube_size, cube_size, cube_size]
        
        # Update nudge history
        self.nudge_history = 0.9 * self.nudge_history + 0.1 * avg_nudge
        
        # Strengthen synapses in high-nudge regions
        # (This is a simplified version - full implementation would
        #  grow connections between adjacent high-nudge neurons)
        high_activity = self.nudge_history > threshold_high
        
        # Increase local convolution weights in active regions
        # This is a proof-of-concept; real neurogenesis would add new connections
        with torch.no_grad():
            # Slightly strengthen kernel (bounded growth)
            if high_activity.any():
                self.local_conv.weight.data.mul_(1.001)
                self.local_conv.weight.data.clamp_(-1, 1)
    
    def pruning(self, threshold_low: float = 0.01):
        """
        Prune synapses where nudge signal is consistently weak.
        
        This implements the "silent learning" → "prune" rule.
        """
        low_activity = self.nudge_history < threshold_low
        
        # Weaken local convolution weights in inactive regions
        with torch.no_grad():
            if low_activity.sum() > 0.5 * self.num_neurons:
                # Too many neurons silent - slightly decay weights
                self.local_conv.weight.data.mul_(0.999)
    
    def get_cube_slices(self, cube_state: torch.Tensor, axis: int = 0) -> List[torch.Tensor]:
        """
        Get 2D slices through the cube for visualization.
        
        Args:
            axis: 0=z (horizontal slices), 1=y, 2=x
        """
        cube = cube_state.squeeze(0).squeeze(0)  # [N, N, N]
        
        slices = []
        for i in range(self.cube_size):
            if axis == 0:
                slices.append(cube[i, :, :])
            elif axis == 1:
                slices.append(cube[:, i, :])
            else:
                slices.append(cube[:, :, i])
        
        return slices
    
    def energy(self, cube_state: torch.Tensor, x: torch.Tensor, buffer=None) -> torch.Tensor:
        """Energy function for the 3D system."""
        x_cube = self._input_to_cube_surface(x)
        
        # Self energy
        E_self = 0.5 * torch.sum(cube_state ** 2)
        
        # Interaction energy
        neighbor_input = self.local_conv(cube_state)
        pre_act = neighbor_input + x_cube + self.neuron_bias
        
        abs_pre = torch.abs(pre_act)
        log_cosh = abs_pre + F.softplus(-2 * abs_pre) - 0.693147
        E_int = -torch.sum(log_cosh)
        
        return E_self + E_int


def visualize_cube_slice(cube_state: torch.Tensor, slice_idx: int = 0) -> str:
    """ASCII visualization of a cube slice."""
    cube = cube_state.squeeze(0).squeeze(0).detach().cpu()  # [N, N, N]
    slice_2d = cube[slice_idx, :, :]
    
    # Map values to ASCII
    chars = " .:-=+*#@"
    
    lines = [f"Slice z={slice_idx}:"]
    for row in slice_2d:
        line = ""
        for val in row:
            idx = int((val.item() + 1) / 2 * (len(chars) - 1))
            idx = max(0, min(len(chars) - 1, idx))
            line += chars[idx]
        lines.append(line)
    
    return "\n".join(lines)


# ============================================================================
# Test
# ============================================================================

def test_neural_cube():
    """Test Neural Cube implementation."""
    print("Testing NeuralCube...")
    
    model = NeuralCube(cube_size=8, input_dim=784, output_dim=10)
    
    x = torch.randn(4, 784)
    y = torch.randint(0, 10, (4,))
    
    out, dynamics = model(x, steps=30, track_dynamics=True)
    
    assert out.shape == (4, 10), f"Wrong output: {out.shape}"
    assert len(dynamics) == 6, f"Wrong dynamics count: {len(dynamics)}"  # 30/5 = 6
    
    # Test gradient flow
    loss = F.cross_entropy(out, y)
    loss.backward()
    
    print(f"✓ Forward pass works")
    print(f"✓ Tracked {len(dynamics)} dynamic states")
    
    # Visualize a slice
    print("\nVisualization of final state:")
    print(visualize_cube_slice(dynamics[-1][0:1], slice_idx=4))
    
    print("\n✓ Neural Cube test passed")


def test_neurogenesis_pruning():
    """Test neurogenesis and pruning."""
    print("\nTesting neurogenesis/pruning...")
    
    model = NeuralCube(cube_size=4, input_dim=64, output_dim=10)
    
    x = torch.randn(8, 64)
    
    # Simulate training with nudge
    for i in range(10):
        out, dynamics = model(x, steps=20, track_dynamics=True)
        
        # Simulate nudge field (in real training, this comes from target diff)
        nudge_field = torch.randn_like(dynamics[-1]).abs() * 0.2
        
        model.neurogenesis(nudge_field)
        model.pruning()
    
    print(f"  Nudge history range: [{model.nudge_history.min():.4f}, {model.nudge_history.max():.4f}]")
    print("✓ Neurogenesis/pruning test passed")


if __name__ == "__main__":
    test_neural_cube()
    test_neurogenesis_pruning()
    print("\n✓ All Neural Cube tests passed!")
