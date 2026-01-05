import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvEqProp(nn.Module):
    """Convolutional Equilibrium Propagation Model.
    
    Structure: Single-Block ResNet-like Loop.
    Dynamics: h_{t+1} = (1-γ)h_t + γ * (Conv2(Tanh(Conv1(h_t))) + Embed(x))
    
    Energy-Based Design:
    - Symmetric or near-symmetric weights required for strict energy descent.
    - Here we use independent weights W1/W2 but rely on spectral norm for contraction.
    """
    def __init__(self, input_channels, hidden_channels, output_dim, gamma=0.5, 
                 use_spectral_norm=True):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.gamma = gamma
        
        # Input embedding (Image -> Hidden State)
        # 1x1 Conv to project input to hidden dim (or 3x3 with padding)
        self.embed = nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1)
        
        # Recurrent Weights (The "Loop")
        # W1: Expansion
        self.W1 = nn.Conv2d(hidden_channels, hidden_channels*2, kernel_size=3, padding=1)
        # W2: Contraction
        self.W2 = nn.Conv2d(hidden_channels*2, hidden_channels, kernel_size=3, padding=1)
        
        if use_spectral_norm:
            from torch.nn.utils.parametrizations import spectral_norm
            self.W1 = spectral_norm(self.W1)
            self.W2 = spectral_norm(self.W2)
            self.embed = spectral_norm(self.embed)
            
        self.norm = nn.GroupNorm(8, hidden_channels) # GroupNorm is better for small batch sizes
        
        # Classifier Head (Global Average Pooling -> Linear)
        self.Head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(hidden_channels, output_dim)
        )
        
        # Init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
        
        # Scale recurrent weights for stability
        with torch.no_grad():
            self.W1.weight.mul_(0.5)
            self.W2.weight.mul_(0.5)

    def forward_step(self, h, x, buffer=None):
        # h: [B, C, H, W]
        h_norm = self.norm(h)
        x_emb = self.embed(x)
        
        # FFN equivalent: Conv -> Tanh -> Conv
        # Tanh is crucial for energy bound (LogCosh)
        pre_act = self.W1(h_norm)
        hidden = torch.tanh(pre_act)
        ffn_out = self.W2(hidden)
        
        h_target = ffn_out + x_emb
        
        h_next = (1 - self.gamma) * h + self.gamma * h_target
        return h_next, None

    def forward(self, x, steps=25):
        # Initialize h to 0 or via input
        x_emb = self.embed(x)
        h = torch.zeros_like(x_emb) # Start from 0 state
        
        for _ in range(steps):
             h, _ = self.forward_step(h, x)
             
        # Readout
        return self.Head(h)

    def init_state(self, x):
        """Initialize state h matching embedding shape."""
        # We need the shape after embedding
        # x is [B, 3, 32, 32] -> embed -> [B, 64, 32, 32]
        # We can implement this efficiently without full forward
        B, _, H, W = x.shape
        # Assuming padding maintains H, W and stride=1
        return torch.zeros(B, self.hidden_channels, H, W, device=x.device)

    def energy(self, h, x, buffer=None):
        """Standard energy function for monitoring."""
        h_norm = self.norm(h)
        x_emb = self.embed(x)
        
        # 1. Spring term (Elasticity)
        E_spring = 0.5 * torch.sum(h**2)
        
        # 2. Interaction term (Connection to next state)
        # Note: For strict gradients, we'd need LogCosh, but this is sufficient for monitoring
        pre_act = self.W1(h_norm)
        hidden = torch.tanh(pre_act)
        ffn_out = self.W2(hidden)
        
        E_int = -torch.sum(h * (ffn_out + x_emb))
        
        return E_spring + E_int
