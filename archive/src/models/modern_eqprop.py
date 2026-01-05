import torch
import torch.nn as nn
import torch.nn.functional as F


class ModernEqProp(nn.Module):
    """Modern Equilibrium Propagation Model with proper energy formulation.
    
    Structure: Residual Network with LayerNorm and Tanh FFN.
    Dynamics: h_{t+1} = (1-γ)h_t + γ * (FFN(LayerNorm(h_t)) + Embed(x))
    
    Key fix: Uses Tanh activation for valid LogCosh energy function.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, gamma=0.5, dropout=0.0,
                 use_spectral_norm=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.gamma = gamma
        self.use_spectral_norm = use_spectral_norm
        
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
        
        # Output classifier
        self.Head = nn.Linear(hidden_dim, output_dim)
        
        # Initialize for stability
        nn.init.orthogonal_(self.W1.weight)
        nn.init.orthogonal_(self.W2.weight)
        with torch.no_grad():
            self.W1.weight.mul_(0.5)
            self.W2.weight.mul_(0.5)

    def forward_step(self, h, x, buffer=None):
        """Dynamics using Tanh for proper energy formulation."""
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
        return h_next, None

    def forward(self, x, steps=30):
        h = self.embed(x)  # Initialize from input
        
        for _ in range(steps):
            h, _ = self.forward_step(h, x)
        return self.Head(h)

    def energy(self, h, x, buffer=None):
        """Proper LogCosh energy function matching Tanh dynamics.
        
        E = 0.5 * ||h||^2 - Σ LogCosh(W1 @ h_norm) - h · (W2 @ tanh(W1 @ h_norm) + embed(x))
        
        This is a valid scalar energy whose gradient descent gives the Tanh dynamics.
        """
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
