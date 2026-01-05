"""Feed-forward network modules for equilibrium transformers.

This module provides FFN implementations:
- StandardFFN: Regular feed-forward network with GELU activation
- SymmetricFFN: Symmetric FFN with W2 = W1^T constraint for energy-based dynamics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from abc import ABC, abstractmethod


class FeedForward(ABC, nn.Module):
    """Base class for feed-forward networks."""
    
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
    
    @abstractmethod
    def forward(self, h: Tensor) -> Tensor:
        """Apply feed-forward transformation.
        
        Args:
            h: Input tensor [seq, batch, d_model]
            
        Returns:
            Output tensor [seq, batch, d_model]
        """
        pass


class StandardFFN(FeedForward):
    """Standard feed-forward network with GELU activation.
    
    Architecture: Linear -> GELU -> Dropout -> Linear
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__(d_model, d_ff)
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
    
    def forward(self, h: Tensor) -> Tensor:
        return self.net(h)


class SymmetricFFN(FeedForward):
    """Symmetric FFN with W2 = W1^T constraint for energy-based dynamics.
    
    This ensures the Jacobian is symmetric, required for EqProp gradient equivalence.
    Uses tanh activation for bounded energy.
    
    Reference: Scellier & Bengio 2017
    """
    
    def __init__(self, d_model: int, d_ff: int):
        super().__init__(d_model, d_ff)
        # Only store W1; W2 = W1^T is applied dynamically
        self.w1 = nn.Linear(d_model, d_ff)
    
    def forward(self, h: Tensor) -> Tensor:
        """Apply symmetric FFN: W1^T @ tanh(W1 @ h)."""
        h_ff = self.w1(h)  # [seq, batch, d_ff]
        h_ff = torch.tanh(h_ff)  # Bounded activation for energy-based dynamics
        # Apply W2 = W1^T constraint
        h_ff = F.linear(h_ff, self.w1.weight.t())  # [seq, batch, d_model]
        return h_ff
