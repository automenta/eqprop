"""
MSTEP: Multi-Scale Toroidal Equilibrium Propagation (Revised)

Simplified hierarchical architecture with shared weights across scales.
Fixed version with stable cross-scale coupling.

Key advantages:
- Coarse-to-fine refinement for faster convergence
- Multi-resolution feature extraction
- Shared weights for parameter efficiency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_eqprop import BaseEqProp


class MSTEP(BaseEqProp):
    """Multi-Scale Toroidal Equilibrium Propagation.
    
    Simplified 2-scale version for stability:
    - Coarse scale: pooled representation
    - Fine scale: full resolution
    - Cross-scale coupling via residual addition
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, gamma=0.5, dropout=0.0,
                 use_spectral_norm=True, n_scales=2, coupling_strength=0.1):
        super().__init__(input_dim, hidden_dim, output_dim, gamma, dropout, use_spectral_norm)
        
        self.n_scales = n_scales
        self.coupling_strength = coupling_strength
        
        # Coarse dimension (half of hidden)
        self.coarse_dim = hidden_dim // 2
        
        # Down/up sampling
        self.downsample = nn.Linear(hidden_dim, self.coarse_dim)
        self.upsample = nn.Linear(self.coarse_dim, hidden_dim)
        
        # Coarse-scale FFN (shared structure with main FFN)
        self.coarse_ffn = nn.Sequential(
            nn.LayerNorm(self.coarse_dim),
            nn.Linear(self.coarse_dim, self.coarse_dim * 4),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(self.coarse_dim * 4, self.coarse_dim)
        )
        
        # Initialize for stability
        for m in self.coarse_ffn.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
    
    def forward_step(self, h, x, buffer=None, **kwargs):
        """Multi-scale equilibrium step."""
        x_emb = self.embed(x)
        
        # Initialize coarse state if needed
        if buffer is None:
            buffer = self.downsample(h)
        h_coarse = buffer
        
        # Coarse scale update (faster dynamics)
        coarse_out = self.coarse_ffn(h_coarse)
        coarse_x = self.downsample(x_emb)
        h_coarse_next = (1 - self.gamma * 1.5) * h_coarse + self.gamma * 1.5 * (coarse_out + coarse_x)
        
        # Fine scale update with coarse guidance
        fine_out = self.ffn(h)
        coarse_guidance = self.upsample(h_coarse_next)
        h_target = fine_out + x_emb + self.coupling_strength * (coarse_guidance - h)
        h_next = (1 - self.gamma) * h + self.gamma * h_target
        
        return h_next, h_coarse_next
    
    def energy(self, h, x, buffer=None):
        """Multi-scale energy with coupling term."""
        base_energy = self.standard_energy(h, x, buffer)
        
        if buffer is not None:
            coarse_up = self.upsample(buffer)
            coupling_energy = 0.5 * self.coupling_strength * torch.sum((h - coarse_up) ** 2)
            base_energy = base_energy + coupling_energy
        
        return base_energy
