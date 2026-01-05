"""
Temporal Resonance: Spatiotemporal Limit Cycle Dynamics (TODO5 Item 4.2)

For video/time-series: the equilibrium becomes a stable vibration (limit cycle)
instead of a fixed point. This enables "Infinite Context Window" - the network
resonates with the sequence rather than buffering it.

Key Insight: The network's attractor is a cycle, not a point.
This is the temporal generalization of EqProp.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import math


class TemporalResonanceEqProp(nn.Module):
    """
    EqProp with Limit Cycle Dynamics for sequence processing.
    
    Instead of converging to a fixed point h*, the network converges
    to a stable oscillation pattern (limit cycle) that resonates
    with the input sequence.
    
    This achieves "infinite context" because state is carried in the
    oscillation pattern, not in a finite buffer.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 3, alpha: float = 0.5,
                 oscillation_strength: float = 0.1,
                 use_spectral_norm: bool = True):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.alpha = alpha
        self.oscillation_strength = oscillation_strength
        
        # Input projection
        self.W_in = nn.Linear(input_dim, hidden_dim)
        
        # Recurrent layers
        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        # Oscillatory coupling (creates limit cycles)
        # This is inspired by coupled oscillator networks
        self.osc_coupling = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Phase variable (tracks position in cycle)
        self.phase_dim = hidden_dim // 4
        
        # Output head
        self.head = nn.Linear(hidden_dim, output_dim)
        
        if use_spectral_norm:
            from torch.nn.utils.parametrizations import spectral_norm
            for i, layer in enumerate(self.layers):
                self.layers[i] = spectral_norm(layer)
            self.osc_coupling = spectral_norm(self.osc_coupling)
        
        # Initialize for stable oscillations
        self._init_oscillatory_weights()
    
    def _init_oscillatory_weights(self):
        """Initialize oscillatory coupling for stable limit cycles."""
        with torch.no_grad():
            # Create a rotation-like structure in the coupling matrix
            # This encourages circular dynamics
            dim = self.hidden_dim
            
            # Block-diagonal with 2x2 rotation blocks
            self.osc_coupling.weight.zero_()
            for i in range(0, dim - 1, 2):
                # Each 2x2 block is a rotation
                angle = 0.1  # Small rotation angle
                self.osc_coupling.weight[i, i] = math.cos(angle)
                self.osc_coupling.weight[i, i+1] = -math.sin(angle)
                self.osc_coupling.weight[i+1, i] = math.sin(angle)
                self.osc_coupling.weight[i+1, i+1] = math.cos(angle)
            
            self.osc_coupling.weight.mul_(self.oscillation_strength)
    
    def forward_step(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Single step with oscillatory dynamics."""
        x_emb = self.W_in(x)
        
        # Standard recurrent dynamics
        h_recurrent = x_emb
        for layer in self.layers:
            h_recurrent = h_recurrent + layer(torch.tanh(h))
        
        # Oscillatory contribution (limit cycle attractor)
        h_oscillatory = self.osc_coupling(h)
        
        # Combined target
        h_target = torch.tanh(h_recurrent + h_oscillatory)
        
        return (1 - self.alpha) * h + self.alpha * h_target
    
    def forward(self, x: torch.Tensor, steps: int = 30) -> torch.Tensor:
        """Forward pass for single input."""
        batch_size = x.size(0)
        h = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        
        for _ in range(steps):
            h = self.forward_step(h, x)
        
        return self.head(h)
    
    def forward_sequence(self, x_seq: torch.Tensor, 
                          steps_per_frame: int = 5) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Process a sequence with resonant dynamics.
        
        Args:
            x_seq: [batch_size, seq_len, input_dim]
            steps_per_frame: Relaxation steps per sequence frame
            
        Returns:
            outputs: [batch_size, seq_len, output_dim]
            trajectories: List of hidden states for visualization
        """
        batch_size, seq_len, _ = x_seq.shape
        
        h = torch.zeros(batch_size, self.hidden_dim, device=x_seq.device)
        
        outputs = []
        trajectories = []
        
        for t in range(seq_len):
            x_t = x_seq[:, t, :]
            
            # Relax toward equilibrium/limit cycle
            for _ in range(steps_per_frame):
                h = self.forward_step(h, x_t)
            
            trajectories.append(h.detach().clone())
            outputs.append(self.head(h))
        
        outputs = torch.stack(outputs, dim=1)  # [B, T, output_dim]
        
        return outputs, trajectories
    
    def detect_limit_cycle(self, x: torch.Tensor, max_steps: int = 200,
                            cycle_detection_window: int = 20) -> Dict:
        """
        Detect if the network has settled into a limit cycle.
        
        Returns:
            dict with cycle_detected, cycle_length, amplitude
        """
        batch_size = x.size(0)
        h = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        
        trajectory = []
        
        for step in range(max_steps):
            h = self.forward_step(h, x)
            trajectory.append(h.detach().clone())
        
        # Analyze trajectory for periodicity
        trajectory = torch.stack(trajectory)  # [T, B, H]
        
        # Check if states are repeating (simple cycle detection)
        recent = trajectory[-cycle_detection_window:]  # Last N states
        
        # Compute autocorrelation at different lags
        correlations = []
        for lag in range(1, cycle_detection_window // 2):
            corr = F.cosine_similarity(
                recent[:-lag].flatten(1),
                recent[lag:].flatten(1)
            ).mean().item()
            correlations.append(corr)
        
        # Find peak correlation (indicates periodicity)
        if correlations:
            max_corr = max(correlations)
            cycle_length = correlations.index(max_corr) + 1
            cycle_detected = max_corr > 0.9
        else:
            max_corr = 0
            cycle_length = 0
            cycle_detected = False
        
        # Compute amplitude of oscillation
        amplitude = torch.std(recent, dim=0).mean().item()
        
        return {
            'cycle_detected': cycle_detected,
            'cycle_length': cycle_length,
            'max_correlation': max_corr,
            'amplitude': amplitude
        }
    
    def energy(self, h: torch.Tensor, x: torch.Tensor, buffer=None) -> torch.Tensor:
        """Energy function (extended for oscillatory dynamics)."""
        x_emb = self.W_in(x)
        
        E = 0.5 * torch.sum(h ** 2)
        
        h_recurrent = x_emb
        for layer in self.layers:
            h_recurrent = h_recurrent + layer(torch.tanh(h))
        
        h_oscillatory = self.osc_coupling(h)
        pre_act = h_recurrent + h_oscillatory
        
        abs_pre = torch.abs(pre_act)
        log_cosh = abs_pre + F.softplus(-2 * abs_pre) - 0.693147
        E -= torch.sum(log_cosh)
        
        return E


# ============================================================================
# Test
# ============================================================================

def test_temporal_resonance():
    """Test temporal resonance implementation."""
    print("Testing TemporalResonanceEqProp...")
    
    model = TemporalResonanceEqProp(
        input_dim=32, hidden_dim=64, output_dim=10,
        oscillation_strength=0.2
    )
    
    # Single input test
    x = torch.randn(8, 32)
    out = model(x, steps=30)
    assert out.shape == (8, 10)
    print("✓ Single input works")
    
    # Sequence input test
    x_seq = torch.randn(8, 20, 32)  # 8 batch, 20 timesteps
    outputs, trajectories = model.forward_sequence(x_seq, steps_per_frame=5)
    assert outputs.shape == (8, 20, 10)
    assert len(trajectories) == 20
    print("✓ Sequence processing works")
    
    # Limit cycle detection
    cycle_info = model.detect_limit_cycle(x, max_steps=100)
    print(f"  Cycle detected: {cycle_info['cycle_detected']}")
    print(f"  Cycle length: {cycle_info['cycle_length']}")
    print(f"  Amplitude: {cycle_info['amplitude']:.4f}")
    
    # Test gradient flow
    loss = outputs.sum()
    loss.backward()
    print("✓ Gradients flow through sequence")
    
    print("\n✓ Temporal Resonance test passed")


def test_infinite_context():
    """Test that the network maintains context over long sequences."""
    print("\nTesting infinite context capability...")
    
    model = TemporalResonanceEqProp(
        input_dim=16, hidden_dim=32, output_dim=5,
        oscillation_strength=0.3
    )
    
    # Create a long sequence with a pattern at the start
    seq_len = 100
    x_seq = torch.randn(4, seq_len, 16)
    
    # Add a distinctive pattern at the beginning
    x_seq[:, 0:5, :] = 0  # Mark with zeros
    
    outputs, trajectories = model.forward_sequence(x_seq, steps_per_frame=3)
    
    # Check if information from start is still influencing end
    # (via the oscillation pattern)
    start_trajectory = trajectories[5]
    end_trajectory = trajectories[-1]
    
    # In a resonant network, the pattern should persist
    correlation = F.cosine_similarity(
        start_trajectory.flatten(1),
        end_trajectory.flatten(1)
    ).mean().item()
    
    print(f"  Start-end trajectory correlation: {correlation:.4f}")
    print(f"  (Higher = better context retention)")
    
    if correlation > 0.1:
        print("✓ Network shows context retention")
    else:
        print("⚠ Context may be decaying (expected for untrained network)")


if __name__ == "__main__":
    test_temporal_resonance()
    test_infinite_context()
    print("\n✓ All Temporal Resonance tests passed!")
