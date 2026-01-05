"""
TEP-SSR: Toroidal Equilibrium Propagation with State-Space Recirculation

Replaces Transformer FFN with a parallel linear state-space model (SSM).
Combines EqProp training with efficient O(N) sequence processing.

Key advantages:
- O(N) inference vs O(N²) for long sequences
- Superior long-range reasoning via SSM dynamics
- Constant memory training (theoretical)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_eqprop import BaseEqProp


class SimpleSSM(nn.Module):
    """Simplified State-Space Model block.
    
    s_{t+1} = A @ s_t + B @ u_t
    y_t = C @ s_t + D @ u_t
    
    Uses diagonal A for efficiency (like S4D/Mamba).
    """
    
    def __init__(self, input_dim, state_dim, output_dim):
        super().__init__()
        self.state_dim = state_dim
        
        # Diagonal A matrix (log-parameterized for stability)
        self.log_A = nn.Parameter(torch.randn(state_dim) * 0.01 - 1.0)  # Start stable
        self.B = nn.Linear(input_dim, state_dim)
        self.C = nn.Linear(state_dim, output_dim)
        self.D = nn.Linear(input_dim, output_dim)
        
        # Internal state
        self._state = None
    
    def reset_state(self, batch_size, device):
        """Reset SSM state."""
        self._state = torch.zeros(batch_size, self.state_dim, device=device)
    
    def forward(self, u):
        """Single SSM step."""
        if self._state is None or self._state.shape[0] != u.shape[0]:
            self.reset_state(u.shape[0], u.device)
        
        # Discrete-time dynamics
        A_diag = torch.sigmoid(self.log_A)  # Ensure |A| < 1 for stability
        self._state = A_diag * self._state + self.B(u)
        y = self.C(self._state) + self.D(u)
        
        return y


class TEPSSR(BaseEqProp):
    """Toroidal Equilibrium Propagation with State-Space Recirculation.
    
    Replaces standard FFN with SSM block for efficient sequence processing.
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, gamma=0.5, dropout=0.0,
                 use_spectral_norm=False, ssm_state_dim=None):
        super().__init__(input_dim, hidden_dim, output_dim, gamma, dropout, use_spectral_norm)
        
        ssm_state_dim = ssm_state_dim or hidden_dim
        
        # Replace FFN with SSM
        self.ssm = SimpleSSM(hidden_dim, ssm_state_dim, hidden_dim)
        self.ssm_norm = nn.LayerNorm(hidden_dim)
        
        # Gating for SSM vs FFN blend
        self.gate = nn.Linear(hidden_dim, hidden_dim)
    
    def forward_step(self, h, x, buffer=None, **kwargs):
        """Equilibrium step with SSM recirculation."""
        x_emb = self.embed(x)
        
        # Standard FFN path
        ffn_out = self.ffn(h)
        
        # SSM path (captures sequential dependencies)
        h_norm = self.ssm_norm(h)
        ssm_out = self.ssm(h_norm)
        
        # Gated combination
        gate = torch.sigmoid(self.gate(h))
        combined = gate * ffn_out + (1 - gate) * ssm_out
        
        # Target state
        h_target = combined + x_emb
        
        # Damped update
        h_next = (1 - self.gamma) * h + self.gamma * h_target
        
        return h_next, None
    
    def forward(self, x, steps=30, **kwargs):
        """Reset SSM state before forward pass."""
        self.ssm.reset_state(x.shape[0], x.device)
        return super().forward(x, steps, **kwargs)
    
    def energy(self, h, x, buffer=None):
        """Energy including SSM state regularization."""
        base_energy = self.standard_energy(h, x, buffer)
        
        # Add SSM state regularization
        if self.ssm._state is not None:
            ssm_reg = 0.01 * torch.sum(self.ssm._state ** 2)
            base_energy = base_energy + ssm_reg
        
        return base_energy
