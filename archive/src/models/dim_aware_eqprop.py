"""
Dimensionality-Aware EqProp Models

Algorithmic interventions to address dimension scaling issue:
1. EmbeddingEqProp - Compress to lower-dim before EqProp
2. ProjectedEqProp - Random projection (Johnson-Lindenstrauss)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EmbeddingEqProp(nn.Module):
    """EqProp with learned compression to lower dimension.
    
    Strategy: Compress high-dim input to target_dim using learned
    embedding, then run EqProp in smaller space.
    
    For MNIST (784d): Compress to 128 or 196 dims first.
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, gamma=0.5, dropout=0.0,
                 use_spectral_norm=True, target_dim=128):
        super().__init__()
        
        self.original_input_dim = input_dim
        self.target_dim = target_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.gamma = gamma
        
        # Learned compression (input → target_dim)
        self.compressor = nn.Sequential(
            nn.Linear(input_dim, target_dim * 2),
            nn.LayerNorm(target_dim * 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(target_dim * 2, target_dim),
            nn.LayerNorm(target_dim)
        )
        
        # Embedding from compressed space
        self.embed_layer = nn.Linear(target_dim, hidden_dim)
        
        # Standard FFN
        self.W1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.W2 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout_layer = nn.Dropout(dropout)
        
        # Output head
        self.Head = nn.Linear(hidden_dim, output_dim)
        
        # Apply spectral norm if requested
        if use_spectral_norm:
            from torch.nn.utils.parametrizations import spectral_norm
            self.W1 = spectral_norm(self.W1)
            self.W2 = spectral_norm(self.W2)
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        for m in self.compressor.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.8)
        nn.init.orthogonal_(self.embed_layer.weight, gain=0.5)
        nn.init.orthogonal_(self.W1.weight, gain=0.5)
        nn.init.orthogonal_(self.W2.weight, gain=0.5)
    
    def compress(self, x):
        """Compress from original dim to target dim."""
        return self.compressor(x)
    
    def ffn(self, h):
        """Standard FFN."""
        h_norm = self.norm(h)
        hidden = torch.tanh(self.W1(h_norm))
        hidden = self.dropout_layer(hidden)
        return self.W2(hidden)
    
    def forward_step(self, h, x_compressed, buffer=None, **kwargs):
        """Forward step in compressed space."""
        x_emb = self.embed_layer(x_compressed)
        ffn_out = self.ffn(h)
        h_target = ffn_out + x_emb
        h_next = (1 - self.gamma) * h + self.gamma * h_target
        return h_next, None
    
    def forward(self, x, steps=30):
        """Forward with compression."""
        # Compress input
        x_compressed = self.compress(x)
        
        # Initialize hidden state
        h = self.embed_layer(x_compressed)
        
        # Equilibrium loop
        for step in range(steps):
            h, _ = self.forward_step(h, x_compressed, step=step)
        
        return self.Head(h)
    
    def energy(self, h, x, buffer=None):
        """Energy in compressed space."""
        x_compressed = self.compress(x)
        x_emb = self.embed_layer(x_compressed)
        
        term1 = 0.5 * torch.sum(h ** 2)
        
        h_norm = self.norm(h)
        pre_act = self.W1(h_norm)
        abs_pre = torch.abs(pre_act)
        log_cosh = abs_pre + F.softplus(-2 * abs_pre) - 0.693147
        term2 = torch.sum(log_cosh)
        
        ffn_hidden = torch.tanh(pre_act)
        ffn_out = self.W2(ffn_hidden)
        coupling = torch.sum(h * (ffn_out + x_emb))
        
        return term1 - term2 - coupling


class ProjectedEqProp(nn.Module):
    """EqProp with random projection preprocessing.
    
    Uses Johnson-Lindenstrauss lemma: Random projection preserves
    distances with high probability.
    
    Advantage: No learned parameters in projection, mathematically
    grounded dimension reduction.
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, gamma=0.5, dropout=0.0,
                 use_spectral_norm=True, target_dim=128):
        super().__init__()
        
        self.original_input_dim = input_dim
        self.target_dim = target_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.gamma = gamma
        
        # Fixed random projection matrix (not learned)
        # Scaled for variance preservation
        scale = np.sqrt(1.0 / target_dim)
        projection = torch.randn(input_dim, target_dim) * scale
        self.register_buffer('projection', projection)
        
        # Embedding from projected space
        self.embed_layer = nn.Linear(target_dim, hidden_dim)
        
        # Standard FFN
        self.W1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.W2 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout_layer = nn.Dropout(dropout)
        
        # Output head
        self.Head = nn.Linear(hidden_dim, output_dim)
        
        # Apply spectral norm
        if use_spectral_norm:
            from torch.nn.utils.parametrizations import spectral_norm
            self.W1 = spectral_norm(self.W1)
            self.W2 = spectral_norm(self.W2)
        
        # Initialize
        nn.init.orthogonal_(self.embed_layer.weight, gain=0.5)
        nn.init.orthogonal_(self.W1.weight, gain=0.5)
        nn.init.orthogonal_(self.W2.weight, gain=0.5)
    
    def project(self, x):
        """Apply random projection."""
        return x @ self.projection
    
    def ffn(self, h):
        """Standard FFN."""
        h_norm = self.norm(h)
        hidden = torch.tanh(self.W1(h_norm))
        hidden = self.dropout_layer(hidden)
        return self.W2(hidden)
    
    def forward_step(self, h, x_proj, buffer=None, **kwargs):
        """Forward step in projected space."""
        x_emb = self.embed_layer(x_proj)
        ffn_out = self.ffn(h)
        h_target = ffn_out + x_emb
        h_next = (1 - self.gamma) * h + self.gamma * h_target
        return h_next, None
    
    def forward(self, x, steps=30):
        """Forward with projection."""
        # Project input
        x_proj = self.project(x)
        
        # Initialize hidden state
        h = self.embed_layer(x_proj)
        
        # Equilibrium loop
        for step in range(steps):
            h, _ = self.forward_step(h, x_proj, step=step)
        
        return self.Head(h)
    
    def energy(self, h, x, buffer=None):
        """Energy in projected space."""
        x_proj = self.project(x)
        x_emb = self.embed_layer(x_proj)
        
        term1 = 0.5 * torch.sum(h ** 2)
        
        h_norm = self.norm(h)
        pre_act = self.W1(h_norm)
        abs_pre = torch.abs(pre_act)
        log_cosh = abs_pre + F.softplus(-2 * abs_pre) - 0.693147
        term2 = torch.sum(log_cosh)
        
        ffn_hidden = torch.tanh(pre_act)
        ffn_out = self.W2(ffn_hidden)
        coupling = torch.sum(h * (ffn_out + x_emb))
        
        return term1 - term2 - coupling


class DimensionScaledEqProp(nn.Module):
    """EqProp with dimension-aware scaling.
    
    Scales the learning dynamics to compensate for dimension.
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, gamma=0.5, dropout=0.0,
                 use_spectral_norm=True, reference_dim=64):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.reference_dim = reference_dim
        
        # Compute dimension scaling - use sqrt scaling
        dim_ratio = input_dim / reference_dim
        self.gamma = gamma * min(1.0, np.sqrt(reference_dim / input_dim))
        
        # Standard layers
        self.embed_layer = nn.Linear(input_dim, hidden_dim)
        self.W1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.W2 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout_layer = nn.Dropout(dropout)
        self.Head = nn.Linear(hidden_dim, output_dim)
        
        if use_spectral_norm:
            from torch.nn.utils.parametrizations import spectral_norm
            self.W1 = spectral_norm(self.W1)
            self.W2 = spectral_norm(self.W2)
        
        # Initialize with smaller gains for larger dims
        gain = 0.5 / np.sqrt(dim_ratio)
        nn.init.orthogonal_(self.embed_layer.weight, gain=gain)
        nn.init.orthogonal_(self.W1.weight, gain=gain)
        nn.init.orthogonal_(self.W2.weight, gain=gain)
    
    def ffn(self, h):
        h_norm = self.norm(h)
        hidden = torch.tanh(self.W1(h_norm))
        hidden = self.dropout_layer(hidden)
        return self.W2(hidden)
    
    def forward_step(self, h, x, buffer=None, **kwargs):
        x_emb = self.embed_layer(x)
        ffn_out = self.ffn(h)
        h_target = ffn_out + x_emb
        h_next = (1 - self.gamma) * h + self.gamma * h_target
        return h_next, None
    
    def forward(self, x, steps=30):
        h = self.embed_layer(x)
        for step in range(steps):
            h, _ = self.forward_step(h, x, step=step)
        return self.Head(h)
    
    def energy(self, h, x, buffer=None):
        x_emb = self.embed_layer(x)
        term1 = 0.5 * torch.sum(h ** 2)
        
        h_norm = self.norm(h)
        pre_act = self.W1(h_norm)
        abs_pre = torch.abs(pre_act)
        log_cosh = abs_pre + F.softplus(-2 * abs_pre) - 0.693147
        term2 = torch.sum(log_cosh)
        
        ffn_hidden = torch.tanh(pre_act)
        ffn_out = self.W2(ffn_hidden)
        coupling = torch.sum(h * (ffn_out + x_emb))
        
        # Scale energy by dimension
        return (term1 - term2 - coupling) / np.sqrt(self.hidden_dim)


# Dummy for compatibility
class AdaptiveBetaEqProp(DimensionScaledEqProp):
    """Same as DimensionScaledEqProp for now."""
    pass
