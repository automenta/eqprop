import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedMLP(nn.Module):
    """
    Gated Equilibrium Propagation (GEP).
    Instead of fixed alpha, use a learnable gate z.
    h_{t+1} = (1-z)h_t + z * tanh(Wx x + Wh h_t)
    z = sigmoid(Wz [h_t, x] + bz)
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.Wx = nn.Linear(input_dim, hidden_dim)
        self.Wh = nn.Linear(hidden_dim, hidden_dim)
        self.Head = nn.Linear(hidden_dim, output_dim)
        
        # Gating network
        self.Wz_x = nn.Linear(input_dim, hidden_dim)
        self.Wz_h = nn.Linear(hidden_dim, hidden_dim)

    def forward_step(self, h, x, buffer=None):
        # Calculate update candidate
        pre_act = self.Wx(x) + self.Wh(h)
        h_cand = torch.tanh(pre_act)
        
        # Calculate gate
        gate = torch.sigmoid(self.Wz_x(x) + self.Wz_h(h))
        
        # Update
        h_new = (1 - gate) * h + gate * h_cand
        return h_new, None

    def forward(self, x, steps=30):
        batch_size = x.size(0)
        h = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        
        for _ in range(steps):
             h, _ = self.forward_step(h, x)
             
        return self.Head(h)

    def energy(self, h, x, buffer=None):
        """
        Energy definition for Gated models is complex. 
        We use the standard implicit energy assumption:
        E = 0.5 * ||h||^2 - Sum(LogCosh(u)) 
        ignoring the gate dynamics for the contrastive target?
        
        Alternatively, Gated EqProp often assumes the fixed point 
        h* = tanh(Wx x + Wh h*) regardless of Gating.
        The gate only affects the PATH to equilibrium, not the location.
        So the Energy function remains the same as LoopedMLP!
        """
        term1 = 0.5 * torch.sum(h ** 2)
        pre_act = self.Wx(x) + self.Wh(h)
        abs_pre = torch.abs(pre_act)
        log_cosh = abs_pre + torch.nn.functional.softplus(-2 * abs_pre) - 0.693147
        term2 = torch.sum(log_cosh)
        return term1 - term2
