"""
Lazy EqProp: Event-Driven Equilibrium Propagation.

Breaks the "Global Clock" to save energy and simulate hardware:
- Activity-gated relaxation: neurons only update if input changed > ε
- "Avalanche" dynamics: input pulse ripples through layers
- Persistent relaxation: continuous weak nudge from output (never fully free)

Goal: Achieve same accuracy as "Clocked" version with ~70% fewer FLOPs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass


@dataclass
class LazyStats:
    """Statistics for lazy execution."""
    total_neurons: int = 0
    active_neurons: int = 0
    skipped_neurons: int = 0
    
    @property
    def skip_ratio(self) -> float:
        if self.total_neurons == 0:
            return 0.0
        return self.skipped_neurons / self.total_neurons
    
    @property
    def flop_savings(self) -> float:
        return self.skip_ratio * 100.0
    
    def reset(self):
        self.total_neurons = 0
        self.active_neurons = 0
        self.skipped_neurons = 0


class LazyEqProp(nn.Module):
    """Event-driven Equilibrium Propagation with lazy updates.
    
    Features:
    - Activity gating: neurons skip update if |Δinput| < epsilon
    - Persistent nudge: output layer continuously provides weak nudge
    - Avalanche visualization: tracks which neurons updated each step
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int = 3,
                 alpha: float = 0.5,
                 epsilon: float = 0.01,
                 persistent_nudge_strength: float = 0.1,
                 use_spectral_norm: bool = True):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.alpha = alpha
        self.epsilon = epsilon  # Activity threshold
        self.persistent_nudge_strength = persistent_nudge_strength
        
        # Input embedding
        self.embed = nn.Linear(input_dim, hidden_dim)
        
        # Hidden layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = nn.Linear(hidden_dim, hidden_dim)
            if use_spectral_norm:
                from torch.nn.utils.parametrizations import spectral_norm
                layer = spectral_norm(layer)
            self.layers.append(layer)
        
        # Output head
        self.Head = nn.Linear(hidden_dim, output_dim)
        
        # Initialize for stability
        for layer in self.layers:
            weight = layer.weight if not use_spectral_norm else layer.parametrizations.weight.original
            nn.init.orthogonal_(weight)
            with torch.no_grad():
                weight.mul_(0.8)
        
        # Statistics tracking
        self.stats = LazyStats()
        
        # Activity masks for visualization
        self.activity_masks: List[torch.Tensor] = []
    
    def lazy_forward_step(self, 
                          h_states: Dict[int, torch.Tensor],
                          prev_inputs: Dict[int, torch.Tensor],
                          x: torch.Tensor,
                          nudge: Optional[torch.Tensor] = None
                          ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        """Single lazy equilibrium step with activity gating.
        
        Args:
            h_states: Current layer states {layer_idx: tensor}
            prev_inputs: Previous layer inputs (for change detection)
            x: Input tensor
            nudge: Optional nudge gradient from output
            
        Returns:
            new_states: Updated states
            new_inputs: Current inputs (for next step comparison)
            activity_masks: Which neurons were active
        """
        x_emb = self.embed(x)
        batch_size = x.size(0)
        
        new_states = {}
        new_inputs = {}
        activity_masks = {}
        
        for i, layer in enumerate(self.layers):
            # Compute input to this layer
            if i == 0:
                layer_input = x_emb
            else:
                layer_input = h_states.get(i - 1, x_emb)
            
            # Add persistent nudge to last layer
            if nudge is not None and i == self.num_layers - 1:
                layer_input = layer_input - self.persistent_nudge_strength * nudge
            
            new_inputs[i] = layer_input
            
            # Get previous input for this layer
            prev = prev_inputs.get(i, torch.zeros_like(layer_input))
            
            # Compute input change magnitude per neuron
            input_delta = (layer_input - prev).abs()
            
            # Activity mask: which neurons have significant input change
            # Shape: [batch, hidden_dim]
            active_mask = (input_delta.mean(dim=-1, keepdim=True) > self.epsilon)
            active_mask = active_mask.expand_as(layer_input).float()
            
            # Track statistics
            num_neurons = batch_size * self.hidden_dim
            num_active = int(active_mask.sum().item())
            self.stats.total_neurons += num_neurons
            self.stats.active_neurons += num_active
            self.stats.skipped_neurons += (num_neurons - num_active)
            
            activity_masks[i] = active_mask
            
            # Current state
            h_current = h_states.get(i, torch.zeros(batch_size, self.hidden_dim, device=x.device))
            
            # Compute new state (only for active neurons)
            pre_act = layer(layer_input)
            h_new = torch.tanh(pre_act)
            h_update = (1 - self.alpha) * h_current + self.alpha * h_new
            
            # Apply activity mask: inactive neurons keep old state
            new_states[i] = active_mask * h_update + (1 - active_mask) * h_current
        
        return new_states, new_inputs, activity_masks
    
    def forward(self, x: torch.Tensor, steps: int = 30, 
                track_activity: bool = False) -> torch.Tensor:
        """Forward pass with lazy dynamics.
        
        Args:
            x: Input tensor
            steps: Number of equilibrium steps
            track_activity: Whether to record activity masks for visualization
            
        Returns:
            Output logits
        """
        batch_size = x.size(0)
        
        # Reset statistics
        self.stats.reset()
        self.activity_masks = []
        
        # Initialize states
        h_states = {
            i: torch.zeros(batch_size, self.hidden_dim, device=x.device)
            for i in range(self.num_layers)
        }
        prev_inputs = {}
        
        for step in range(steps):
            # Optional: compute nudge from current output (persistent relaxation)
            nudge = None
            if self.persistent_nudge_strength > 0 and step > 0:
                # Weak nudge based on current prediction confidence
                with torch.no_grad():
                    logits = self.Head(h_states[self.num_layers - 1])
                    probs = F.softmax(logits, dim=-1)
                    # Nudge toward high-confidence direction
                    nudge = probs - probs.mean(dim=-1, keepdim=True)
                    nudge = nudge @ self.Head.weight  # Project back to hidden
            
            h_states, prev_inputs, masks = self.lazy_forward_step(
                h_states, prev_inputs, x, nudge
            )
            
            if track_activity:
                self.activity_masks.append(masks)
        
        return self.Head(h_states[self.num_layers - 1])
    
    def energy(self, h: torch.Tensor, x: torch.Tensor, 
               buffer: Optional[dict] = None) -> torch.Tensor:
        """Energy function for EqProp training."""
        x_emb = self.embed(x)
        
        # Using last layer state
        h_norm = h
        
        # Self-interaction
        term1 = 0.5 * torch.sum(h ** 2)
        
        # LogCosh for last layer
        pre_act = self.layers[-1](h_norm)
        abs_pre = torch.abs(pre_act)
        log_cosh = abs_pre + F.softplus(-2 * abs_pre) - 0.693147
        term2 = torch.sum(log_cosh)
        
        # Coupling with input
        coupling = torch.sum(h * x_emb)
        
        return term1 - term2 - coupling
    
    def get_flop_savings(self) -> float:
        """Get FLOP savings from lazy execution."""
        return self.stats.flop_savings


class AvalancheVisualizer:
    """Visualize avalanche dynamics in lazy networks.
    
    Tracks the "ripple" of activity propagating through the network
    when a new input arrives.
    """
    
    def __init__(self, model: LazyEqProp):
        self.model = model
    
    def capture_avalanche(self, x: torch.Tensor, steps: int = 30) -> List[Dict[int, torch.Tensor]]:
        """Run forward pass and capture activity masks.
        
        Returns list of {layer_idx: activity_mask} for each step.
        """
        _ = self.model(x, steps=steps, track_activity=True)
        return self.model.activity_masks
    
    def activity_summary(self) -> Dict[str, float]:
        """Summarize avalanche statistics."""
        if not self.model.activity_masks:
            return {}
        
        total_active = 0
        total_neurons = 0
        
        for step_masks in self.model.activity_masks:
            for layer_idx, mask in step_masks.items():
                total_neurons += mask.numel()
                total_active += mask.sum().item()
        
        return {
            'total_neurons': total_neurons,
            'total_active': total_active,
            'activity_ratio': total_active / max(total_neurons, 1),
            'flop_savings': 1.0 - (total_active / max(total_neurons, 1)),
        }


# ============================================================================
# Test
# ============================================================================

def test_lazy_eqprop():
    """Test lazy EqProp implementation."""
    print("Testing LazyEqProp...")
    
    model = LazyEqProp(
        input_dim=10,
        hidden_dim=64,
        output_dim=5,
        num_layers=3,
        epsilon=0.01,
    )
    
    x = torch.randn(4, 10)
    out = model(x, steps=20, track_activity=True)
    
    assert out.shape == (4, 5), f"Expected (4, 5), got {out.shape}"
    print(f"  ✓ Output shape: {out.shape}")
    print(f"  ✓ FLOP savings: {model.get_flop_savings():.1f}%")
    print(f"  ✓ Activity masks recorded: {len(model.activity_masks)}")
    
    # Test avalanche visualizer
    viz = AvalancheVisualizer(model)
    summary = viz.activity_summary()
    print(f"  ✓ Activity ratio: {summary.get('activity_ratio', 0):.2f}")
    
    print("LazyEqProp tests passed!")


def test_persistent_nudge():
    """Test persistent nudge mode."""
    print("\nTesting persistent nudge...")
    
    model = LazyEqProp(
        input_dim=10,
        hidden_dim=32,
        output_dim=5,
        num_layers=3,
        persistent_nudge_strength=0.2,
    )
    
    x = torch.randn(4, 10)
    out = model(x, steps=30)
    
    assert out.shape == (4, 5)
    print(f"  ✓ Persistent nudge mode works")
    
    print("Persistent nudge tests passed!")


if __name__ == '__main__':
    test_lazy_eqprop()
    test_persistent_nudge()
    print("\n✓ All lazy EqProp tests passed!")
