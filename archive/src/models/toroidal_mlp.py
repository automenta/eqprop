import torch
import torch.nn as nn
import torch.nn.functional as F


class ToroidalMLP(nn.Module):
    """Toroidal Equilibrium Propagation (TEP) MLP with ring buffer.
    
    Uses a stacked state tensor [batch, K+1, hidden_dim] where:
    - Index 0: Current state s(t)
    - Index 1..K: History h(t-1)..h(t-K)
    
    Dynamics:
    s(t+1) = (1-γ)s(t) + γ * tanh(Wx·x + Wh·s(t) + Σ α_k·h(t-k))
    
    The recirculation buffer stabilizes dynamics using exponentially
    weighted historical states.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, alpha=0.5, 
                 buffer_size=3, decay=0.9, use_spectral_norm=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.alpha = alpha  # Damping factor (gamma in dynamics)
        self.buffer_size = buffer_size
        self.use_spectral_norm = use_spectral_norm
        
        # Pre-compute recirculation weights α_k = decay^k
        alphas = [decay ** (k + 1) for k in range(buffer_size)]
        self.register_buffer('alphas', torch.tensor(alphas))
        
        self.Wx = nn.Linear(input_dim, hidden_dim, bias=True)
        self.Wh = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.Head = nn.Linear(hidden_dim, output_dim, bias=True)
        
        # Apply spectral normalization for convergence guarantee
        if use_spectral_norm:
            from torch.nn.utils.parametrizations import spectral_norm
            self.Wh = spectral_norm(self.Wh)
        
        # Learnable buffer mixing strength
        self.buffer_gamma = nn.Parameter(torch.tensor(0.1))

        # Initialize for stability
        nn.init.orthogonal_(self.Wh.weight)
        with torch.no_grad():
            self.Wh.weight.mul_(0.9)

    def forward_step(self, h_stack, x, buffer_state=None):
        """Update step with ring buffer management.
        
        Args:
            h_stack: Stacked history [batch, K+1, hidden_dim]
                     or [batch, hidden_dim] for legacy compatibility
            x: Input [batch, input_dim]
            buffer_state: Ignored (state is in h_stack)
            
        Returns:
            Updated stack [batch, K+1, hidden_dim], None
        """
        # Handle legacy single-tensor input
        if h_stack.dim() == 2:
            return self._legacy_forward_step(h_stack, x)
        
        s_t = h_stack[:, 0]        # Current state [batch, hidden_dim]
        history = h_stack[:, 1:]   # [batch, K, hidden_dim]
        
        # Compute recirculation term: Σ α_k · h(t-k)
        # alphas: [K], history: [batch, K, hidden_dim]
        alphas = self.alphas.view(1, -1, 1)  # [1, K, 1]
        recirculation = (history * alphas).sum(dim=1)  # [batch, hidden_dim]
        
        # Dynamics: tanh(Wx·x + Wh·s + buffer_gamma·recirculation)
        pre_act = self.Wx(x) + self.Wh(s_t) + self.buffer_gamma * recirculation
        h_new = torch.tanh(pre_act)
        
        # Damped update
        s_next = (1 - self.alpha) * s_t + self.alpha * h_new
        
        # Shift buffer: s_next becomes current, old current becomes h(t-1), etc.
        # new_stack = [s_next, s_t, h(t-1), ..., h(t-K+1)]
        new_stack = torch.cat([
            s_next.unsqueeze(1),
            s_t.unsqueeze(1),
            history[:, :-1]  # Drop oldest
        ], dim=1)
        
        return new_stack, None
    
    def _legacy_forward_step(self, h, x):
        """Legacy single-tensor mode for backward compatibility."""
        pre_act = self.Wx(x) + self.Wh(h)
        h_new = torch.tanh(pre_act)
        return (1 - self.alpha) * h + self.alpha * h_new, None

    def forward(self, x, steps=30):
        batch_size = x.size(0)
        
        # Initialize stacked state: [s_0, zeros...]
        s_0 = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        zeros = torch.zeros(batch_size, self.buffer_size, self.hidden_dim, 
                           device=x.device)
        h_stack = torch.cat([s_0.unsqueeze(1), zeros], dim=1)
        
        for _ in range(steps):
            h_stack, _ = self.forward_step(h_stack, x)
        
        # Classify using current state (index 0)
        return self.Head(h_stack[:, 0])

    def energy(self, h, x, buffer_state=None):
        """Energy function for TEP.
        
        Handles both stacked [batch, K+1, hidden_dim] and legacy [batch, hidden_dim].
        """
        # Extract current state
        if h.dim() == 3:
            s_t = h[:, 0]
            history = h[:, 1:]
            alphas = self.alphas.view(1, -1, 1)
            recirculation = (history * alphas).sum(dim=1)
        else:
            s_t = h
            recirculation = torch.zeros_like(h)
        
        # E = 0.5||s||^2 - LogCosh(pre_activation)
        term1 = 0.5 * torch.sum(s_t ** 2)
        
        pre_act = self.Wx(x) + self.Wh(s_t) + self.buffer_gamma * recirculation
        abs_pre = torch.abs(pre_act)
        log_cosh = abs_pre + F.softplus(-2 * abs_pre) - 0.693147
        term2 = torch.sum(log_cosh)
        
        return term1 - term2
