"""Attention mechanisms for equilibrium transformers.

This module provides various attention implementations:
- SoftmaxAttention: Standard transformer attention
- LinearAttention: Efficient linear attention with ELU+1 feature map
- SymmetricLinearAttention: Linear attention with symmetric weight constraints
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from abc import ABC, abstractmethod
from .utils import linear_attention


class Attention(ABC, nn.Module):
    """Base class for attention mechanisms."""
    
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
    
    @abstractmethod
    def forward(self, h: Tensor, x: Tensor) -> Tensor:
        """Apply attention mechanism.
        
        Args:
            h: Query tensor [seq, batch, d_model]
            x: Key/Value tensor [seq, batch, d_model]
            
        Returns:
            Attention output [seq, batch, d_model]
        """
        pass


class SoftmaxAttention(Attention):
    """Standard transformer softmax attention using PyTorch's MultiheadAttention."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__(d_model, n_heads)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=False)
    
    def forward(self, h: Tensor, x: Tensor) -> Tensor:
        """Cross-attention: Query=h, Key=x, Value=x."""
        attn_out, _ = self.attn(h, x, x, need_weights=False)
        return attn_out


class LinearAttention(Attention):
    """Efficient linear attention with ELU+1 feature map.
    
    Uses the Performer-style kernel trick: φ(Q) @ (φ(K)^T @ V)
    This reduces complexity from O(S^2) to O(S) in sequence length.
    """
    
    def __init__(self, d_model: int, n_heads: int):
        super().__init__(d_model, n_heads)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
    
    def forward(self, h: Tensor, x: Tensor) -> Tensor:
        """Apply linear attention with separate Q, K, V projections."""
        seq_len, batch_size, _ = h.shape
        
        # Project and reshape for multi-head
        Q = self._project_and_reshape(self.q_proj(h), seq_len, batch_size)
        K = self._project_and_reshape(self.k_proj(x), seq_len, batch_size)
        V = self._project_and_reshape(self.v_proj(x), seq_len, batch_size)
        
        # Apply feature map φ(x) = elu(x) + 1
        Q_prime = F.elu(Q) + 1.0
        K_prime = F.elu(K) + 1.0
        
        # Efficient linear attention with optimized contraction order
        # Compute KV = K^T @ V first (smaller intermediate tensor)
        KV = torch.einsum('bhsd,bhsv->bhdv', K_prime, V)
        # Sum K for normalization
        K_sum = K_prime.sum(dim=2)  # [B, H, D]
        Z = torch.einsum('bhsd,bhd->bhs', Q_prime, K_sum) + 1e-6
        out = torch.einsum('bhsd,bhdv->bhsv', Q_prime, KV) / Z.unsqueeze(-1)
        
        # Reshape back and apply output projection
        out = self._reshape_back(out, seq_len, batch_size)
        return self.out_proj(out)
    
    def _project_and_reshape(self, tensor: Tensor, seq_len: int, batch_size: int) -> Tensor:
        """Reshape [seq, batch, d_model] to [batch, heads, seq, head_dim]."""
        tensor = tensor.view(seq_len, batch_size, self.n_heads, self.head_dim)
        return tensor.permute(1, 2, 0, 3).contiguous()  # [B, H, S, D] - contiguous for better memory access
    
    def _reshape_back(self, tensor: Tensor, seq_len: int, batch_size: int) -> Tensor:
        """Reshape [batch, heads, seq, head_dim] to [seq, batch, d_model]."""
        tensor = tensor.permute(2, 0, 1, 3).contiguous()
        return tensor.view(seq_len, batch_size, self.d_model)


class SymmetricLinearAttention(Attention):
    """Linear attention with symmetric weight constraints for energy-based dynamics.
    
    Implements the following constraints required for EqProp:
    - W_out = W_q^T (output projection is query weight transposed)
    - W_k = W_v (key and value projections share weights)
    
    These constraints ensure the Jacobian is symmetric, which is required
    for Scellier & Bengio 2017's gradient equivalence theorem.
    """
    
    def __init__(self, d_model: int, n_heads: int):
        super().__init__(d_model, n_heads)
        # Only need Q and K projections; V shares weights with K
        self.w_q = nn.Linear(d_model, d_model, bias=True)
        self.w_k = nn.Linear(d_model, d_model, bias=True)
        # W_out = W_q^T is applied dynamically in forward pass
    
    def forward(self, h: Tensor, x: Tensor) -> Tensor:
        """Apply symmetric linear attention.
        
        Note: V = K (weight sharing) and output uses W_q^T.
        """
        seq_len, batch_size, _ = h.shape
        
        # Project Q and K (V = K by weight sharing)
        Q = self.w_q(h)  # [seq, batch, d_model]
        K = self.w_k(x)  # [seq, batch, d_model]
        V = K  # Symmetric constraint: W_k = W_v
        
        # Reshape for linear_attention function: [B*H, S, head_dim]
        Q = self._reshape_for_attention(Q, seq_len, batch_size)
        K = self._reshape_for_attention(K, seq_len, batch_size)
        V = self._reshape_for_attention(V, seq_len, batch_size)
        
        # Apply linear attention (handles feature map internally)
        attn_out = linear_attention(Q, K, V)  # [B*H, S, head_dim]
        
        # Reshape back to [seq, batch, d_model]
        B = batch_size
        S = seq_len
        attn_out = attn_out.view(B, self.n_heads, S, self.head_dim)
        attn_out = attn_out.permute(2, 0, 1, 3).reshape(S, B, self.d_model)
        
        # Apply W_out = W_q^T constraint (symmetric weight tying)
        attn_out = F.linear(attn_out, self.w_q.weight.t())
        
        return attn_out
    
    def _reshape_for_attention(self, tensor: Tensor, seq_len: int, batch_size: int) -> Tensor:
        """Reshape [seq, batch, d_model] to [batch*heads, seq, head_dim]."""
        # [S, B, D] -> [B, S, D]
        tensor = tensor.permute(1, 0, 2)
        # [B, S, D] -> [B, S, H, d]
        tensor = tensor.view(batch_size, seq_len, self.n_heads, self.head_dim)
        # [B, S, H, d] -> [B, H, S, d] -> [B*H, S, d]
        tensor = tensor.permute(0, 2, 1, 3).reshape(
            batch_size * self.n_heads, seq_len, self.head_dim
        )
        return tensor
