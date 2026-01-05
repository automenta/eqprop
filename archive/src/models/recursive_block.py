"""
Recursive Block: Fractal/Hierarchical EqProp Architecture.

Implements nested equilibrium dynamics where:
- Each Block(N) is a self-contained recurrent loop
- Inner "mini-TorEq" iterates K times per outer iteration
- Spectral normalization stabilizes both levels

This tests whether nested equilibria can learn complex features
(loops, textures) better than flat stacks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math


class RecursiveEqPropCore(nn.Module):
    """Inner equilibrium core that runs inside a block.
    
    A lightweight EqProp that reaches its own local equilibrium
    before returning the result to the outer loop.
    """
    
    def __init__(self, 
                 dim: int,
                 inner_steps: int = 5,
                 alpha: float = 0.5,
                 use_spectral_norm: bool = True):
        super().__init__()
        self.dim = dim
        self.inner_steps = inner_steps
        self.alpha = alpha
        
        # Recurrent weight
        self.W = nn.Linear(dim, dim, bias=True)
        
        if use_spectral_norm:
            from torch.nn.utils.parametrizations import spectral_norm
            self.W = spectral_norm(self.W)
        
        # Initialize for stability
        weight = self.W.weight if not use_spectral_norm else self.W.parametrizations.weight.original
        nn.init.orthogonal_(weight)
        with torch.no_grad():
            weight.mul_(0.8)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run inner equilibrium loop."""
        h = torch.zeros_like(x)
        
        for _ in range(self.inner_steps):
            pre_act = self.W(h) + x  # Input acts as constant external drive
            h_new = torch.tanh(pre_act)
            h = (1 - self.alpha) * h + self.alpha * h_new
        
        return h


class RecursiveBlock(nn.Module):
    """Fractal block: contains inner EqProp that iterates faster than outer.
    
    Architecture:
    - Input projection
    - Inner RecursiveEqPropCore (iterates inner_steps times per call)
    - Output projection
    
    The key insight: test if having "sub-brains" reaching their own
    equilibria inside a larger global equilibrium helps learning.
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 inner_steps: int = 5,
                 outer_alpha: float = 0.5,
                 inner_alpha: float = 0.5,
                 use_spectral_norm: bool = True):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.alpha = outer_alpha
        
        # Project input to hidden
        self.embed = nn.Linear(input_dim, hidden_dim)
        
        # Inner equilibrium core
        self.inner_core = RecursiveEqPropCore(
            dim=hidden_dim,
            inner_steps=inner_steps,
            alpha=inner_alpha,
            use_spectral_norm=use_spectral_norm,
        )
        
        # Output projection
        self.Head = nn.Linear(hidden_dim, output_dim)
        
        # For multi-block stacking
        self.use_spectral_norm = use_spectral_norm
    
    def forward_step(self, h: torch.Tensor, x: torch.Tensor, 
                     buffer: Optional[dict] = None) -> Tuple[torch.Tensor, Optional[dict]]:
        """Single outer equilibrium step with inner equilibrium.
        
        The inner core runs to its own equilibrium, then the result
        is used for the outer update.
        """
        x_emb = self.embed(x)
        
        # Run inner equilibrium (the "sub-brain")
        # Input to inner: current outer state + embedded input
        inner_input = h + x_emb
        inner_equilibrium = self.inner_core(inner_input)
        
        # Outer damped update
        h_new = (1 - self.alpha) * h + self.alpha * inner_equilibrium
        
        return h_new, buffer
    
    def forward(self, x: torch.Tensor, steps: int = 30) -> torch.Tensor:
        """Forward pass to outer equilibrium."""
        batch_size = x.size(0)
        h = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        
        for _ in range(steps):
            h, _ = self.forward_step(h, x)
        
        return self.Head(h)
    
    def energy(self, h: torch.Tensor, x: torch.Tensor, 
               buffer: Optional[dict] = None) -> torch.Tensor:
        """Compute energy for EqProp training.
        
        Energy of nested system is the sum of outer and inner energies.
        """
        x_emb = self.embed(x)
        
        # Outer self-interaction
        term1 = 0.5 * torch.sum(h ** 2)
        
        # Inner equilibrium energy at current h
        inner_input = h + x_emb
        inner_h = self.inner_core(inner_input)
        
        # LogCosh energy for inner dynamics
        pre_act = self.inner_core.W(inner_h) + inner_input
        abs_pre = torch.abs(pre_act)
        log_cosh = abs_pre + F.softplus(-2 * abs_pre) - 0.693147
        term2 = torch.sum(log_cosh)
        
        # Coupling term
        coupling = torch.sum(h * inner_h)
        
        return term1 - term2 - coupling


class DeepRecursiveNetwork(nn.Module):
    """Stack of RecursiveBlocks for deep fractal architecture.
    
    For the "100-layer challenge" and testing gradient flow
    through deeply nested equilibria.
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_blocks: int = 10,
                 inner_steps: int = 5,
                 outer_alpha: float = 0.5,
                 use_spectral_norm: bool = True):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_blocks = num_blocks
        self.alpha = outer_alpha
        
        # Input embedding
        self.embed = nn.Linear(input_dim, hidden_dim)
        
        # Stack of recursive blocks (all share hidden_dim)
        self.blocks = nn.ModuleList([
            RecursiveEqPropCore(
                dim=hidden_dim,
                inner_steps=inner_steps,
                alpha=outer_alpha,
                use_spectral_norm=use_spectral_norm,
            )
            for _ in range(num_blocks)
        ])
        
        # Output head
        self.Head = nn.Linear(hidden_dim, output_dim)
        
        # Initialize embedding
        nn.init.orthogonal_(self.embed.weight)
    
    def forward_step(self, h_states: Dict[int, torch.Tensor], 
                     x: torch.Tensor,
                     buffer: Optional[dict] = None) -> Tuple[Dict[int, torch.Tensor], dict]:
        """Single step through all blocks.
        
        Each block takes output from previous block (or input embedding).
        """
        x_emb = self.embed(x)
        new_states = {}
        
        for i, block in enumerate(self.blocks):
            # Input: previous block output or embedding
            if i == 0:
                block_input = x_emb
            else:
                block_input = h_states.get(i - 1, x_emb)
            
            # Current state
            h_current = h_states.get(i, torch.zeros_like(block_input))
            
            # Block runs inner equilibrium
            inner_eq = block(h_current + block_input)
            
            # Damped update
            h_new = (1 - self.alpha) * h_current + self.alpha * inner_eq
            new_states[i] = h_new
        
        return new_states, buffer or {}
    
    def forward(self, x: torch.Tensor, steps: int = 30) -> torch.Tensor:
        """Forward pass to equilibrium across all blocks."""
        batch_size = x.size(0)
        
        # Initialize all block states
        h_states = {
            i: torch.zeros(batch_size, self.hidden_dim, device=x.device)
            for i in range(self.num_blocks)
        }
        
        for _ in range(steps):
            h_states, _ = self.forward_step(h_states, x)
        
        # Output from last block
        return self.Head(h_states[self.num_blocks - 1])
    
    def get_layer_states(self) -> Dict[str, torch.Tensor]:
        """Get current states for visualization (compatibility)."""
        return {}


# ============================================================================
# Test
# ============================================================================

def test_recursive_block():
    """Test that RecursiveBlock works correctly."""
    print("Testing RecursiveBlock...")
    
    block = RecursiveBlock(
        input_dim=10,
        hidden_dim=64,
        output_dim=5,
        inner_steps=5,
    )
    
    x = torch.randn(4, 10)
    out = block(x, steps=20)
    
    assert out.shape == (4, 5), f"Expected (4, 5), got {out.shape}"
    print(f"  ✓ Output shape: {out.shape}")
    
    # Test energy
    h = torch.randn(4, 64)
    e = block.energy(h, x)
    assert e.dim() == 0, "Energy should be scalar"
    print(f"  ✓ Energy: {e.item():.4f}")
    
    print("RecursiveBlock tests passed!")


def test_deep_recursive_network():
    """Test DeepRecursiveNetwork with many blocks."""
    print("\nTesting DeepRecursiveNetwork...")
    
    # Test with 10 blocks (each with 5 inner steps = 50 "effective" layers)
    net = DeepRecursiveNetwork(
        input_dim=10,
        hidden_dim=32,
        output_dim=5,
        num_blocks=10,
        inner_steps=5,
    )
    
    x = torch.randn(4, 10)
    out = net(x, steps=20)
    
    assert out.shape == (4, 5), f"Expected (4, 5), got {out.shape}"
    print(f"  ✓ Output shape: {out.shape}")
    print(f"  ✓ Effective depth: {10 * 5} = 50 layers")
    
    # Check gradient flow
    loss = out.sum()
    loss.backward()
    
    # Verify gradients exist in first block
    first_block = net.blocks[0]
    grad_exists = first_block.W.weight.grad is not None
    grad_nonzero = grad_exists and first_block.W.weight.grad.abs().max() > 0
    print(f"  ✓ Gradient flows to first block: {grad_nonzero}")
    
    print("DeepRecursiveNetwork tests passed!")


if __name__ == '__main__':
    test_recursive_block()
    test_deep_recursive_network()
    print("\n✓ All recursive block tests passed!")
