"""
TCEP: Toroidal Continuous Equilibrium Propagation (Revised)

Continuous-time dynamics with toroidal recirculation for fading memory.
Fixed version with stable integration and proper buffer handling.

Key advantages:
- Continuous dynamics for smooth convergence
- Toroidal recirculation provides temporal context
- Fading memory prevents gradient explosion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_eqprop import BaseEqProp


class TCEP(BaseEqProp):
    """Toroidal Continuous Equilibrium Propagation.
    
    Stable version with:
    - Conservative dt (0.1)
    - Moderate damping (0.3)
    - Gentle recirculation (0.05 strength)
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, gamma=0.5, dropout=0.0,
                 use_spectral_norm=True, dt=0.1, damping=0.3, 
                 recirc_strength=0.05, recirc_decay=0.9):
        # Force spectral norm for stability
        super().__init__(input_dim, hidden_dim, output_dim, gamma, dropout,
                        use_spectral_norm=True)
        
        self.dt = dt
        self.damping = damping
        self.recirc_strength = recirc_strength
        self.recirc_decay = recirc_decay
        
        # Smaller init for stability
        with torch.no_grad():
            self.W1.weight.mul_(0.3)
            self.W2.weight.mul_(0.3)
    
    def forward_step(self, h, x, buffer=None, step=0, max_steps=30, **kwargs):
        """Continuous-time step with toroidal recirculation."""
        x_emb = self.embed(x)
        
        # Initialize buffer if needed
        if buffer is None:
            buffer = torch.zeros_like(h)
        
        # FFN output
        ffn_out = self.ffn(h)
        
        # Toroidal recirculation: decaying influence from buffer
        decay = self.recirc_decay ** (step + 1)
        recirc = self.recirc_strength * decay * buffer
        
        # ODE: dh/dt = f_θ(h, x) + x_emb - λ*h + recirc
        dh_dt = ffn_out + x_emb - self.damping * h + recirc
        
        # Euler integration
        h_next = h + self.dt * dh_dt
        
        # Update buffer with exponential moving average
        buffer_next = self.recirc_decay * buffer + (1 - self.recirc_decay) * h_next.detach()
        
        return h_next, buffer_next
    
    def energy(self, h, x, buffer=None):
        """Standard energy with recirculation regularization."""
        base_energy = self.standard_energy(h, x, buffer)
        
        if buffer is not None:
            recirc_reg = 0.01 * self.recirc_strength * torch.sum((h - buffer) ** 2)
            base_energy = base_energy + recirc_reg
        
        return base_energy
