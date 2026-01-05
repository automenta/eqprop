"""
HTSEP: Hyper-Toroidal Stochastic Equilibrium Propagation (Revised)

Simplified version with optional stochastic spiking and toroidal buffer.
Fixed for stability with conservative defaults.

Key advantages:
- Fading memory via toroidal buffer
- Optional sparsity via stochastic spiking
- Continual learning capability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_eqprop import BaseEqProp


class HTSEP(BaseEqProp):
    """Hyper-Toroidal Stochastic Equilibrium Propagation.
    
    Simplified stable version:
    - Exponential moving average buffer (no complex torus)
    - Optional soft spiking (disabled by default for stability)
    - Conservative blending with buffer
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, gamma=0.5, dropout=0.0,
                 use_spectral_norm=True, buffer_decay=0.9, use_spiking=False,
                 spike_threshold=0.5):
        super().__init__(input_dim, hidden_dim, output_dim, gamma, dropout, use_spectral_norm)
        
        self.buffer_decay = buffer_decay
        self.use_spiking = use_spiking
        self.spike_threshold = spike_threshold
        
        # Learnable buffer blending weight
        self.buffer_weight = nn.Parameter(torch.tensor(0.1))
    
    def _soft_spike(self, h):
        """Soft spiking with straight-through estimator."""
        if not self.use_spiking or not self.training:
            return h
        
        # Soft thresholding
        magnitude = torch.abs(h)
        mask = torch.sigmoid((magnitude - self.spike_threshold) * 10)
        return h * mask
    
    def forward_step(self, h, x, buffer=None, **kwargs):
        """Equilibrium step with toroidal buffer."""
        x_emb = self.embed(x)
        
        # Initialize buffer if needed
        if buffer is None:
            buffer = torch.zeros_like(h)
        
        # Standard FFN
        ffn_out = self.ffn(h)
        
        # Optional soft spiking
        ffn_out = self._soft_spike(ffn_out)
        
        # Target state
        h_target = ffn_out + x_emb
        
        # Damped update
        h_next = (1 - self.gamma) * h + self.gamma * h_target
        
        # Blend with buffer (fading memory effect)
        buffer_w = torch.sigmoid(self.buffer_weight)
        h_next = (1 - buffer_w) * h_next + buffer_w * buffer
        
        # Update buffer with EMA
        buffer_next = self.buffer_decay * buffer + (1 - self.buffer_decay) * h_next.detach()
        
        return h_next, buffer_next
    
    def energy(self, h, x, buffer=None):
        """Standard energy with buffer coherence."""
        base_energy = self.standard_energy(h, x, buffer)
        
        if buffer is not None:
            buffer_w = torch.sigmoid(self.buffer_weight)
            coherence = 0.01 * buffer_w * torch.sum((h - buffer) ** 2)
            base_energy = base_energy + coherence
        
        return base_energy
