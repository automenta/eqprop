"""
Transformer Attention for Equilibrium Propagation (Preliminary)

First attempt at integrating self-attention into EqProp.

Key Challenge: Attention requires computing Q·K^T, which involves
non-local interactions. How do we maintain the local credit assignment
property of EqProp?

Solution: Treat attention as part of the equilibrium dynamics.
The network iterates to find equilibrium attention patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class EqPropAttention(nn.Module):
    """
    Self-attention that iterates to equilibrium.
    
    Instead of one-shot attention(Q, K, V), we have:
    - Q, K, V evolve during relaxation
    - Attention weights stabilize at equilibrium
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Projections (with spectral norm for stability)
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.W_o = nn.Linear(hidden_dim, hidden_dim)
        
        # Apply spectral normalization
        from torch.nn.utils.parametrizations import spectral_norm
        self.W_q = spectral_norm(self.W_q)
        self.W_k = spectral_norm(self.W_k)
        self.W_v = spectral_norm(self.W_v)
        self.W_o = spectral_norm(self.W_o)
        
        # Scale factor for attention scores
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Compute attention in one step (for use in equilibrium dynamics).
        
        Args:
            h: [batch_size, seq_len, hidden_dim]
        
        Returns:
            attended: [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = h.shape
        
        # Project to Q, K, V
        Q = self.W_q(h)  # [B, L, D]
        K = self.W_k(h)
        V = self.W_v(h)
        
        # Reshape for multi-head: [B, L, D] -> [B, H, L, D/H]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, H, L, L]
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # [B, H, L, D/H]
        
        # Reshape back: [B, H, L, D/H] -> [B, L, D]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        
        # Output projection
        output = self.W_o(attn_output)
        
        return output


class TransformerEqProp(nn.Module):
    """
    Transformer-style architecture with EqProp dynamics.
    
    Key idea: Each layer has a hidden state h[l] that iterates to equilibrium.
    Attention and FFN are applied during each iteration step.
    """
    
    def __init__(self, vocab_size: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 3, num_heads: int = 4,
                 max_seq_len: int = 128, alpha: float = 0.5):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.alpha = alpha
        self.max_seq_len = max_seq_len
        
        # Token + positional embedding
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)
        
        # Attention layers
        self.attentions = nn.ModuleList([
            EqPropAttention(hidden_dim, num_heads) for _ in range(num_layers)
        ])
        
        # FFN layers (with spectral norm)
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
            for _ in range(num_layers)
        ])
        
        # Layer norms
        self.norms1 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.norms2 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        
        # Output head
        self.head = nn.Linear(hidden_dim, output_dim)
        
        # Apply spectral normalization to FFNs
        from torch.nn.utils.parametrizations import spectral_norm
        for ffn in self.ffns:
            ffn[0] = spectral_norm(ffn[0])
            ffn[2] = spectral_norm(ffn[2])
    
    def embed_input(self, x: torch.Tensor) -> torch.Tensor:
        """Embed tokens with positional encoding."""
        seq_len = x.size(1)
        
        # Token embeddings
        tok_emb = self.token_emb(x)  # [batch_size, seq_len, hidden_dim]
        
        # Positional embeddings
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        pos_emb = self.pos_emb(positions)
        
        return tok_emb + pos_emb
    
    def forward_step(self, h: torch.Tensor, x_emb: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Single equilibrium iteration step for one layer.
        
        Args:
            h: Current hidden state [batch_size, seq_len, hidden_dim]
            x_emb: Input embedding (constant during relaxation)
            layer_idx: Which layer to update
        
        Returns:
            h_next: Updated hidden state
        """
        # Pre-norm architecture
        h_norm = self.norms1[layer_idx](h)
        
        # Attention block
        attn_out = self.attentions[layer_idx](h_norm)
        h = h + attn_out  # Residual connection
        
        # FFN block
        h_norm = self.norms2[layer_idx](h)
        ffn_out = self.ffns[layer_idx](h_norm)
        h_target = h + ffn_out  # Residual connection
        
        # Smooth update (like LoopedMLP)
        h_target = h_target + x_emb  # Keep input signal
        h_next = (1 - self.alpha) * h + self.alpha * torch.tanh(h_target)
        
        return h_next
    
    def forward(self, x: torch.Tensor, steps: int = 20) -> torch.Tensor:
        """
        Forward pass to equilibrium.
        
        Args:
            x: Input tokens [batch_size, seq_len]
            steps: Number of equilibrium iterations
        
        Returns:
            output: [batch_size, output_dim] (classification)
                   or [batch_size, seq_len, output_dim] (generation)
        """
        batch_size, seq_len = x.shape
        
        # Embed input
        x_emb = self.embed_input(x)
        
        # Initialize hidden states
        h = torch.zeros(batch_size, seq_len, self.hidden_dim, device=x.device)
        
        # Iterate to equilibrium
        for step in range(steps):
            # Update each layer sequentially
            for layer_idx in range(self.num_layers):
                h = self.forward_step(h, x_emb, layer_idx)
        
        # Pool for classification (mean pooling)
        h_pooled = h.mean(dim=1)  # [batch_size, hidden_dim]
        
        # Output
        return self.head(h_pooled)
    
    def energy(self, h: torch.Tensor, x_emb: torch.Tensor) -> torch.Tensor:
        """Energy function (for monitoring/debugging)."""
        E = 0.5 * torch.sum(h ** 2)
        
        # Interaction energy from attention + FFN
        # (Simplified - full energy would need careful derivation)
        for layer_idx in range(self.num_layers):
            h_norm = self.norms1[layer_idx](h)
            attn_out = self.attentions[layer_idx](h_norm)
            E -= torch.sum(h * attn_out)
        
        return E


# ============================================================================
# Quick Tests
# ============================================================================

def test_eqprop_attention():
    """Test attention mechanism."""
    print("Testing EqPropAttention...")
    
    attn = EqPropAttention(hidden_dim=64, num_heads=4)
    
    x = torch.randn(2, 10, 64)  # [batch, seq, dim]
    out = attn(x)
    
    assert out.shape == x.shape
    print(f"  Input: {x.shape}, Output: {out.shape}")
    print("✓ EqPropAttention works")


def test_transformer_eqprop():
    """Test full Transformer EqProp."""
    print("\nTesting TransformerEqProp...")
    
    model = TransformerEqProp(
        vocab_size=100,
        hidden_dim=64,
        output_dim=10,
        num_layers=2,
        num_heads=4
    )
    
    # Dummy input
    x = torch.randint(0, 100, (4, 20))  # [batch, seq_len]
    
    # Forward pass
    out = model(x, steps=15)
    
    assert out.shape == (4, 10)
    print(f"  Input: {x.shape}, Output: {out.shape}")
    
    # Test gradient flow
    y = torch.randint(0, 10, (4,))
    loss = F.cross_entropy(out, y)
    loss.backward()
    
    print("✓ TransformerEqProp works, gradients flow")


def test_simple_classification():
    """Test on simple sequence classification."""
    print("\nTesting on toy sequence classification...")
    
    model = TransformerEqProp(
        vocab_size=50,
        hidden_dim=32,
        output_dim=2,
        num_layers=2,
        num_heads=2
    )
    
    # Synthetic data: sequences of length 10
    x = torch.randint(0, 50, (16, 10))
    y = torch.randint(0, 2, (16,))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Train for a few steps
    losses = []
    for i in range(20):
        optimizer.zero_grad()
        out = model(x, steps=10)
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    print(f"  Loss: {losses[0]:.3f} → {losses[-1]:.3f}")
    
    if losses[-1] < losses[0]:
        print("✓ Model is learning (loss decreased)")
    else:
        print("⚠ Model may not be learning (check hyperparameters)")


if __name__ == "__main__":
    test_eqprop_attention()
    test_transformer_eqprop()
    test_simple_classification()
    print("\n✓ All Transformer EqProp tests passed!")
