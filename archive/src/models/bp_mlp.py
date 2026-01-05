import torch
import torch.nn as nn

class BackpropMLP(nn.Module):
    """
    Standard Feedforward MLP for baseline comparison.
    Matches the parameter count of LoopedMLP roughly if unrolled?
    No, it matches strictly the architectural capacity: 
    Input -> Hidden -> Output.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, depth=1):
        super().__init__()
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())
        
        # Hidden layers (if depth > 1)
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
            
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x, steps=None):
        # steps argument is ignored, kept for API compatibility with LoopedMLP
        return self.net(x)
