"""Transformer models for equilibrium propagation."""

import torch
import torch.nn as nn
from torch import Tensor
from .attention import SoftmaxAttention, LinearAttention, SymmetricLinearAttention
from .ffn import StandardFFN, SymmetricFFN


class LoopedTransformerBlock(nn.Module):
    """Single weight-tied transformer block for equilibrium iteration.
    
    Supports both standard and symmetric modes:
    - Standard mode: Regular attention and FFN with LayerNorm
    - Symmetric mode: Symmetric weight tying (W_out=W_q^T, W_k=W_v, W2=W1^T)
      for energy-based dynamics required by EqProp theoretical guarantees.
      
    Reference: Scellier & Bengio (2017), "Equilibrium Propagation: Bridging the
    Gap between Energy-Based Models and Backpropagation"
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0, 
                 attention_type: str = 'softmax', symmetric: bool = False):
        """Initialize transformer block.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout rate (only used for softmax attention)
            attention_type: Type of attention ('softmax' or 'linear')
            symmetric: Whether to use symmetric weight tying for EqProp
            
        Raises:
            ValueError: If symmetric=True with attention_type='softmax'
        """
        super().__init__()
        self.attention_type = attention_type
        self.symmetric = symmetric
        self.d_model = d_model
        self.n_heads = n_heads
        
        # Validate configuration
        if symmetric and attention_type == 'softmax':
            raise ValueError("Symmetric mode requires attention_type='linear'")
        
        # Initialize attention module
        self.attention = self._create_attention(d_model, n_heads, dropout, 
                                                 attention_type, symmetric)
        
        # Initialize FFN module
        self.ffn = self._create_ffn(d_model, d_ff, dropout, symmetric)
        
        # Initialize normalization layers (only for non-symmetric mode)
        if not symmetric:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.norm_final = nn.LayerNorm(d_model)  # Universal Transformer style stability
    
    @staticmethod
    def _create_attention(d_model: int, n_heads: int, dropout: float,
                          attention_type: str, symmetric: bool):
        """Factory method for creating attention module."""
        if attention_type == 'softmax':
            return SoftmaxAttention(d_model, n_heads, dropout)
        else:  # linear
            if symmetric:
                return SymmetricLinearAttention(d_model, n_heads)
            else:
                return LinearAttention(d_model, n_heads)
    
    @staticmethod
    def _create_ffn(d_model: int, d_ff: int, dropout: float, symmetric: bool):
        """Factory method for creating FFN module."""
        if symmetric:
            return SymmetricFFN(d_model, d_ff)
        else:
            return StandardFFN(d_model, d_ff, dropout)
    
    def forward(self, h: Tensor, x: Tensor) -> Tensor:
        """Forward pass with equilibrium dynamics.
        
        Args:
            h: Hidden state [seq, batch, d_model]
            x: Input [seq, batch, d_model]
            
        Returns:
            Updated hidden state [seq, batch, d_model]
        """
        if not self.symmetric:
            return self._forward_standard(h, x)
        else:
            return self._forward_symmetric(h, x)
    
    def _forward_standard(self, h: Tensor, x: Tensor) -> Tensor:
        """Standard (non-symmetric) forward pass with LayerNorm."""
        # Attention block
        h_norm = self.norm1(h)
        attn_out = self.attention(h_norm, x)
        h = h + attn_out
        
        # FFN block
        h_norm = self.norm2(h)
        ffn_out = self.ffn(h_norm)
        h = h + ffn_out
        
        # Final normalization for stability
        h = self.norm_final(h)
        return h
    
    def _forward_symmetric(self, h: Tensor, x: Tensor) -> Tensor:
        """Symmetric forward pass with tanh activation for bounded energy."""
        attn_out = self.attention(h, x)
        ffn_out = self.ffn(h)
        
        # Combine with tanh for overall bounded energy
        return torch.tanh(h + attn_out + ffn_out)
