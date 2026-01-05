"""
DiffTorEqProp: Diffusion-Enhanced Toroidal Equilibrium Propagation

Integrates diffusion processes into the equilibrium propagation framework.
Adds controllable Gaussian noise during free-phase relaxation, followed by
energy-guided denoising.

Key advantages:
- Robustness to noisy inputs
- Faster convergence via exploration
- Generative capabilities
- Implicit regularization for better generalization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffTorEqProp(nn.Module):
    """Diffusion-enhanced Equilibrium Propagation.
    
    During equilibrium relaxation, adds controllable noise followed by
    energy-guided denoising pull toward the energy minimum.
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, gamma=0.5, dropout=0.0,
                 use_spectral_norm=False, noise_start=0.1, noise_end=0.01, 
                 denoise_alpha=0.01):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.gamma = gamma
        self.use_spectral_norm = use_spectral_norm
        
        # Diffusion hyperparameters
        self.noise_start = noise_start
        self.noise_end = noise_end
        self.denoise_alpha = denoise_alpha
        
        self.ffn_dim = 4 * hidden_dim
        
        # Input embedding
        self.embed = nn.Linear(input_dim, hidden_dim)
        
        # Weight-tied FFN block with TANH for valid energy function
        self.W1 = nn.Linear(hidden_dim, self.ffn_dim)
        self.W2 = nn.Linear(self.ffn_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Apply spectral normalization for convergence guarantee
        if use_spectral_norm:
            from torch.nn.utils.parametrizations import spectral_norm
            self.W1 = spectral_norm(self.W1)
            self.W2 = spectral_norm(self.W2)
        
        # Layer norm for stability
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Output classifier (capital H for trainer compatibility)
        self.Head = nn.Linear(hidden_dim, output_dim)
        
        # Initialize for stability
        nn.init.orthogonal_(self.W1.weight)
        nn.init.orthogonal_(self.W2.weight)
        with torch.no_grad():
            self.W1.weight.mul_(0.5)
            self.W2.weight.mul_(0.5)

    def _get_noise_level(self, step, max_steps):
        """Linear noise schedule from noise_start to noise_end."""
        if max_steps <= 1:
            return self.noise_end
        progress = step / (max_steps - 1)
        return self.noise_start * (1 - progress) + self.noise_end * progress

    def forward_step(self, h, x, buffer=None, step=0, max_steps=30, add_noise=False):
        """Dynamics with optional diffusion noise injection."""
        h_norm = self.norm(h)
        x_emb = self.embed(x)
        
        # FFN with Tanh (crucial for LogCosh energy)
        ffn_hidden = torch.tanh(self.W1(h_norm))
        ffn_hidden = self.dropout(ffn_hidden)
        ffn_out = self.W2(ffn_hidden)
        
        # Target state includes input injection
        h_target = ffn_out + x_emb 
        
        # Damped update for convergence
        h_next = (1 - self.gamma) * h + self.gamma * h_target
        
        # Add diffusion noise during training (stochastic exploration)
        if add_noise and self.training:
            noise_level = self._get_noise_level(step, max_steps)
            noise = torch.randn_like(h_next) * noise_level
            
            # Energy-guided denoising: pull toward lower energy
            # Approximate energy gradient direction
            with torch.enable_grad():
                h_temp = h_next.detach().requires_grad_(True)
                energy = self._local_energy(h_temp, x)
                energy_grad = torch.autograd.grad(energy, h_temp)[0]
            
            # Add noise and denoise in direction of energy descent
            h_next = h_next + noise - self.denoise_alpha * energy_grad
        
        return h_next, None

    def _local_energy(self, h, x):
        """Compute local energy for gradient-based denoising."""
        h_norm = self.norm(h)
        x_emb = self.embed(x)
        
        term1 = 0.5 * torch.sum(h ** 2)
        pre_act = self.W1(h_norm)
        abs_pre = torch.abs(pre_act)
        log_cosh = abs_pre + F.softplus(-2 * abs_pre) - 0.693147
        term2 = torch.sum(log_cosh)
        ffn_hidden = torch.tanh(pre_act)
        ffn_out = self.W2(ffn_hidden)
        coupling = torch.sum(h * (ffn_out + x_emb))
        
        return term1 - term2 - coupling

    def forward(self, x, steps=30, add_noise=None):
        """Forward pass to equilibrium with optional diffusion."""
        if add_noise is None:
            add_noise = self.training  # Only add noise during training
        
        h = self.embed(x)
        for step in range(steps):
            h, _ = self.forward_step(h, x, step=step, max_steps=steps, add_noise=add_noise)
        return self.Head(h)

    def energy(self, h, x, buffer=None):
        """Proper LogCosh energy function matching Tanh dynamics."""
        x_emb = self.embed(x)
        h_norm = self.norm(h)
        
        # Self-interaction term
        term1 = 0.5 * torch.sum(h ** 2)
        
        # LogCosh potential for first FFN layer (integral of tanh)
        pre_act = self.W1(h_norm)
        abs_pre = torch.abs(pre_act)
        log_cosh = abs_pre + F.softplus(-2 * abs_pre) - 0.693147  # log(2)
        term2 = torch.sum(log_cosh)
        
        # Coupling term
        ffn_hidden = torch.tanh(pre_act)
        ffn_out = self.W2(ffn_hidden)
        coupling = torch.sum(h * (ffn_out + x_emb))
        
        return term1 - term2 - coupling
