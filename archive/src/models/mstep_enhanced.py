"""
Enhanced MSTEP: Multi-Scale Toroidal Equilibrium Propagation

3-scale hierarchical architecture with:
- Cross-scale attention for rich inter-scale communication
- Adaptive scale gating to skip unnecessary scales
- Hierarchical supervision at each scale

Key advantages:
- Coarse-to-fine refinement (multigrid acceleration)
- Multi-resolution feature learning
- Better for hierarchical tasks (vision, RL)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_eqprop import BaseEqProp


class CrossScaleAttention(nn.Module):
    """Lightweight cross-scale attention mechanism."""
    
    def __init__(self, dim_query, dim_key):
        super().__init__()
        self.query = nn.Linear(dim_query, dim_query)
        self.key = nn.Linear(dim_key, dim_query)  # Project to query dim
        self.value = nn.Linear(dim_key, dim_query)
        self.scale = dim_query ** -0.5
    
    def forward(self, q, k):
        """
        q: [batch, dim_query] - finer scale
        k: [batch, dim_key] - coarser scale
        """
        Q = self.query(q)
        K = self.key(k)
        V = self.value(k)
        
        # Simple dot-product attention
        attn = torch.sigmoid(torch.sum(Q * K, dim=-1, keepdim=True) * self.scale)
        return attn * V


class EnhancedMSTEP(BaseEqProp):
    """Enhanced Multi-Scale Toroidal Equilibrium Propagation.
    
    3-scale pyramid:
    - Coarse: hidden_dim // 4
    - Medium: hidden_dim // 2  
    - Fine: hidden_dim
    
    With cross-scale attention and adaptive gating.
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, gamma=0.5, dropout=0.0,
                 use_spectral_norm=True, coupling_strength=0.15, adaptive=True):
        super().__init__(input_dim, hidden_dim, output_dim, gamma, dropout, use_spectral_norm)
        
        self.coupling_strength = coupling_strength
        self.adaptive = adaptive
        
        # Scale dimensions
        self.coarse_dim = hidden_dim // 4
        self.medium_dim = hidden_dim // 2
        self.fine_dim = hidden_dim
        
        # Down/up sampling between scales
        self.down_fine_to_medium = nn.Linear(self.fine_dim, self.medium_dim)
        self.down_medium_to_coarse = nn.Linear(self.medium_dim, self.coarse_dim)
        self.up_coarse_to_medium = nn.Linear(self.coarse_dim, self.medium_dim)
        self.up_medium_to_fine = nn.Linear(self.medium_dim, self.fine_dim)
        
        # Scale-specific FFNs
        self.coarse_ffn = nn.Sequential(
            nn.LayerNorm(self.coarse_dim),
            nn.Linear(self.coarse_dim, self.coarse_dim * 4),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(self.coarse_dim * 4, self.coarse_dim)
        )
        
        self.medium_ffn = nn.Sequential(
            nn.LayerNorm(self.medium_dim),
            nn.Linear(self.medium_dim, self.medium_dim * 4),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(self.medium_dim * 4, self.medium_dim)
        )
        
        # Cross-scale attention
        self.medium_to_coarse_attn = CrossScaleAttention(self.medium_dim, self.coarse_dim)
        self.fine_to_medium_attn = CrossScaleAttention(self.fine_dim, self.medium_dim)
        
        # Adaptive gating
        if adaptive:
            self.coarse_gate = nn.Linear(self.coarse_dim, 1)
            self.medium_gate = nn.Linear(self.medium_dim, 1)
        
        # Initialize
        for m in [self.coarse_ffn, self.medium_ffn]:
            for layer in m.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=0.5)
    
    def forward_step(self, h, x, buffer=None, **kwargs):
        """Multi-scale equilibrium step with cross-attention."""
        x_emb = self.embed(x)
        
        # Initialize scales if needed
        if buffer is None:
            h_medium = self.down_fine_to_medium(h)
            h_coarse = self.down_medium_to_coarse(h_medium)
            buffer = (h_coarse, h_medium)
        
        h_coarse, h_medium = buffer
        
        # --- Coarse scale update ---
        coarse_out = self.coarse_ffn(h_coarse)
        coarse_x = self.down_medium_to_coarse(self.down_fine_to_medium(x_emb))
        
        # Adaptive gating
        if self.adaptive:
            coarse_gate = torch.sigmoid(self.coarse_gate(h_coarse))
            coarse_out = coarse_gate * coarse_out
        
        h_coarse_next = (1 - self.gamma * 1.8) * h_coarse + self.gamma * 1.8 * (coarse_out + coarse_x)
        
        # --- Medium scale update with coarse guidance ---
        medium_out = self.medium_ffn(h_medium)
        medium_x = self.down_fine_to_medium(x_emb)
        
        # Cross-scale attention from coarse
        coarse_guidance = self.medium_to_coarse_attn(h_medium, h_coarse_next)
        
        if self.adaptive:
            medium_gate = torch.sigmoid(self.medium_gate(h_medium))
            medium_out = medium_gate * medium_out
        
        h_medium_target = medium_out + medium_x + self.coupling_strength * coarse_guidance
        h_medium_next = (1 - self.gamma * 1.4) * h_medium + self.gamma * 1.4 * h_medium_target
        
        # --- Fine scale update with medium guidance ---
        fine_out = self.ffn(h)
        
        # Cross-scale attention from medium
        medium_guidance = self.fine_to_medium_attn(h, h_medium_next)
        
        h_target = fine_out + x_emb + self.coupling_strength * medium_guidance
        h_next = (1 - self.gamma) * h + self.gamma * h_target
        
        return h_next, (h_coarse_next, h_medium_next)
    
    def energy(self, h, x, buffer=None):
        """Multi-scale energy with cross-scale coupling."""
        base_energy = self.standard_energy(h, x, buffer)
        
        if buffer is not None:
            h_coarse, h_medium = buffer
            
            # Consistency between scales
            medium_from_fine = self.down_fine_to_medium(h)
            coarse_from_medium = self.down_medium_to_coarse(h_medium)
            
            coupling_energy = 0.5 * self.coupling_strength * (
                torch.sum((h_medium - medium_from_fine) ** 2) +
                torch.sum((h_coarse - coarse_from_medium) ** 2)
            )
            
            base_energy = base_energy + coupling_energy
        
        return base_energy
