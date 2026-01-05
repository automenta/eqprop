"""
Base class for all Equilibrium Propagation model variants.

Provides shared infrastructure:
- Common initialization patterns
- Standard energy function template
- Convergence measurement utilities
- Trainer-compatible interface (Head, forward_step, energy)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod


class BaseEqProp(nn.Module, ABC):
    """Abstract base class for Equilibrium Propagation models.
    
    All subclasses must implement:
    - forward_step(h, x, buffer=None, **kwargs) -> (h_next, buffer)
    - energy(h, x, buffer=None) -> scalar
    
    Provides:
    - Common FFN structure with spectral norm option
    - Input embedding
    - Output head (self.Head for trainer compatibility)
    - Standard forward pass to equilibrium
    - Convergence measurement
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, gamma=0.5, dropout=0.0,
                 use_spectral_norm=False, ffn_mult=4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.gamma = gamma
        self.use_spectral_norm = use_spectral_norm
        self.ffn_dim = ffn_mult * hidden_dim
        
        # Input embedding
        self.embed = nn.Linear(input_dim, hidden_dim)
        
        # Standard FFN block
        self.W1 = nn.Linear(hidden_dim, self.ffn_dim)
        self.W2 = nn.Linear(self.ffn_dim, hidden_dim)
        self.dropout_layer = nn.Dropout(dropout)
        
        # Layer norm for stability
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Output head (capital H for trainer compatibility)
        self.Head = nn.Linear(hidden_dim, output_dim)
        
        # Apply spectral normalization
        if use_spectral_norm:
            self._apply_spectral_norm()
        
        # Initialize weights for stability
        self._init_weights()
    
    def _apply_spectral_norm(self):
        """Apply spectral normalization to FFN weights."""
        from torch.nn.utils.parametrizations import spectral_norm
        self.W1 = spectral_norm(self.W1)
        self.W2 = spectral_norm(self.W2)
    
    def _init_weights(self):
        """Initialize weights for stability."""
        nn.init.orthogonal_(self.embed.weight, gain=0.5)
        nn.init.orthogonal_(self.W1.weight, gain=0.5)
        nn.init.orthogonal_(self.W2.weight, gain=0.5)
    
    def ffn(self, h):
        """Standard FFN with tanh activation."""
        h_norm = self.norm(h)
        ffn_hidden = torch.tanh(self.W1(h_norm))
        ffn_hidden = self.dropout_layer(ffn_hidden)
        return self.W2(ffn_hidden)
    
    @abstractmethod
    def forward_step(self, h, x, buffer=None, **kwargs):
        """Single equilibrium step. Must return (h_next, buffer)."""
        pass
    
    @abstractmethod
    def energy(self, h, x, buffer=None):
        """Compute scalar energy. Must return a scalar tensor."""
        pass
    
    def forward(self, x, steps=30, **kwargs):
        """Forward pass to equilibrium."""
        h = self.embed(x)
        buffer = None
        for step in range(steps):
            h, buffer = self.forward_step(h, x, buffer, step=step, max_steps=steps, **kwargs)
        return self.Head(h)
    
    def measure_convergence(self, x, max_steps=50, threshold=1e-4):
        """Measure steps to equilibrium convergence."""
        h = self.embed(x)
        prev_h = h.clone()
        buffer = None
        
        for step in range(max_steps):
            h, buffer = self.forward_step(h, x, buffer, step=step, max_steps=max_steps)
            delta = (h - prev_h).abs().mean().item()
            if delta < threshold:
                return step + 1
            prev_h = h.clone()
        return max_steps
    
    def standard_energy(self, h, x, buffer=None):
        """Standard LogCosh energy for Tanh dynamics."""
        x_emb = self.embed(x)
        h_norm = self.norm(h)
        
        # Self-interaction term
        term1 = 0.5 * torch.sum(h ** 2)
        
        # LogCosh potential
        pre_act = self.W1(h_norm)
        abs_pre = torch.abs(pre_act)
        log_cosh = abs_pre + F.softplus(-2 * abs_pre) - 0.693147
        term2 = torch.sum(log_cosh)
        
        # Coupling term
        ffn_hidden = torch.tanh(pre_act)
        ffn_out = self.W2(ffn_hidden)
        coupling = torch.sum(h * (ffn_out + x_emb))
        
        return term1 - term2 - coupling
