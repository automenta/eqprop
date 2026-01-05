import torch
import torch.nn as nn
import torch.nn.functional as F

class LoopedMLP(nn.Module):
    """
    A simple weight-tied Looped MLP.
    Dynamics: h_{t+1} = (1-alpha)h_t + alpha * tanh(W h_t + W_x x + b)
    """
    def __init__(self, input_dim, hidden_dim, output_dim, alpha=0.5, symmetric=False,
                 use_spectral_norm=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.alpha = alpha
        self.symmetric = symmetric
        self.use_spectral_norm = use_spectral_norm

        # Parameters
        self.Wx = nn.Linear(input_dim, hidden_dim, bias=True)
        self._Wh = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.Head = nn.Linear(hidden_dim, output_dim, bias=True)
        
        # Apply spectral normalization for convergence guarantee
        if use_spectral_norm:
            from torch.nn.utils.parametrizations import spectral_norm
            self._Wh = spectral_norm(self._Wh)
        
        nn.init.orthogonal_(self._Wh.weight)
        with torch.no_grad():
            self._Wh.weight.mul_(0.9)

    def get_wh_weight(self):
        if self.symmetric:
            w = self._Wh.weight
            return 0.5 * (w + w.t())
        return self._Wh.weight

    def forward_step(self, h, x, buffer_state=None):
        wh_w = self.get_wh_weight()
        wh_b = self._Wh.bias
        
        pre_act = self.Wx(x) + F.linear(h, wh_w, wh_b)
        h_new = torch.tanh(pre_act)
        return (1 - self.alpha) * h + self.alpha * h_new, None

    def forward(self, x, steps=30):
        batch_size = x.size(0)
        h = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        
        for _ in range(steps):
            h, _ = self.forward_step(h, x)
            
        return self.Head(h)
    
    def energy(self, h, x, buffer_list=None): # buffer_list ignored
        """
        Scalar Energy function. 
        E = 0.5 * ||h||^2 - Sum(LogCosh(Wx x + Wh h))
        """
        # Note: We rely on the autograd to compute dE/dTheta.
        # This energy function implies the fixed point h = tanh(Wx x + Wh h).
        
        # Self-interaction term: 0.5 * ||h||^2
        term1 = 0.5 * torch.sum(h ** 2)
        
        # Interaction potential: Integral of tanh is LogCosh
        wh_w = self.get_wh_weight()
        wh_b = self._Wh.bias
        
        # Note: energy calculation needs to match forward dynamics
        pre_act = self.Wx(x) + F.linear(h, wh_w, wh_b)
        # Stable LogCosh implementation
        # log cosh(x) = log( (e^x + e^-x)/2 )
        #             = x + softplus(-2x) - log2
        abs_pre = torch.abs(pre_act)
        log_cosh = abs_pre + torch.nn.functional.softplus(-2 * abs_pre) - 0.693147
        term2 = torch.sum(log_cosh)
        
        return term1 - term2
