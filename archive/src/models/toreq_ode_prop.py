"""
TorEqODEProp: Toroidal Equilibrium ODE Propagation (Revised)

Models relaxation as continuous-time dynamics with Euler integration.
Fixed version with proper initialization and stable defaults.

Key advantages:
- Infinite effective depth without parameter explosion
- Better for cyclic/periodic data via toroidal constraints
- Continuous flow reduces oscillations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_eqprop import BaseEqProp


class TorEqODEProp(BaseEqProp):
    """Toroidal Equilibrium ODE Propagation.
    
    Uses Euler integration with stable defaults:
    - Small dt (0.1) for stability
    - Moderate damping (0.3) to prevent runaway
    - Optional toroidal projection for periodic data
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, gamma=0.5, dropout=0.0,
                 use_spectral_norm=True, dt=0.1, damping=0.3, toroidal_dims=0):
        # Force spectral norm for ODE stability
        super().__init__(input_dim, hidden_dim, output_dim, gamma, dropout, 
                        use_spectral_norm=True)
        
        self.dt = dt
        self.damping = damping
        self.toroidal_dims = toroidal_dims
        
        # Smaller initialization for ODE stability
        with torch.no_grad():
            self.W1.weight.mul_(0.3)
            self.W2.weight.mul_(0.3)
    
    def _toroidal_project(self, h):
        """Project specified dimensions onto torus (periodic boundaries)."""
        if self.toroidal_dims > 0:
            h = h.clone()
            h[..., :self.toroidal_dims] = torch.remainder(
                h[..., :self.toroidal_dims], 2 * 3.14159
            )
        return h
    
    def forward_step(self, h, x, buffer=None, **kwargs):
        """Stable Euler integration step."""
        x_emb = self.embed(x)
        
        # Compute f_θ(h, x) with normalized hidden state
        h_norm = self.norm(h)
        ffn_out = self.ffn(h)
        
        # ODE: dh/dt = f_θ(h, x) + x_emb - λ*h
        # The damping prevents unbounded growth
        dh_dt = ffn_out + x_emb - self.damping * h
        
        # Euler step with dt
        h_next = h + self.dt * dh_dt
        
        # Apply toroidal projection if enabled
        h_next = self._toroidal_project(h_next)
        
        return h_next, None
    
    def energy(self, h, x, buffer=None):
        """Standard energy."""
        return self.standard_energy(h, x, buffer)
