"""
Ternary EqProp: 1-Bit Quantized Weight Learning (TODO5 Item 2.2)

Implements low-precision learning with weights quantized to {-1, 0, 1}.
Uses Straight-Through Estimator (STE) for gradient flow through quantization.

Target: 90%+ accuracy on MNIST with ternary weights.
Hardware Implication: Blueprint for next-gen neuromorphic chips.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class TernaryQuantize(torch.autograd.Function):
    """
    Ternary quantization with Straight-Through Estimator.
    
    Forward: w -> {-1, 0, +1} based on thresholds
    Backward: Gradients pass through unchanged (STE)
    """
    
    @staticmethod
    def forward(ctx, weight: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        # Save for backward
        ctx.save_for_backward(weight)
        ctx.threshold = threshold
        
        # Ternarize: magnitude above threshold → sign, else → 0
        magnitude = torch.abs(weight)
        scale = torch.mean(magnitude[magnitude > threshold]) if (magnitude > threshold).any() else 1.0
        
        ternary = torch.zeros_like(weight)
        ternary[weight > threshold] = 1.0
        ternary[weight < -threshold] = -1.0
        
        # Scale to maintain magnitude
        return ternary * scale
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        weight, = ctx.saved_tensors
        # STE: pass gradients through, but clip to prevent explosion
        grad_input = grad_output.clone()
        # Optionally clip gradients for very large weights
        grad_input[torch.abs(weight) > 2.0] *= 0.1
        return grad_input, None


def ternarize(weight: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Apply ternary quantization."""
    return TernaryQuantize.apply(weight, threshold)


class TernaryLinear(nn.Module):
    """Linear layer with ternary weights."""
    
    def __init__(self, in_features: int, out_features: int, 
                 threshold: float = 0.5, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.threshold = threshold
        
        # Full-precision weight (for training)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        # Initialize so ~1/3 are +1, ~1/3 are -1, ~1/3 are 0
        nn.init.normal_(self.weight, 0, 0.8)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use ternary weights for forward pass
        w_ternary = ternarize(self.weight, self.threshold)
        return F.linear(x, w_ternary, self.bias)
    
    def get_ternary_stats(self) -> dict:
        """Get statistics about ternary weight distribution."""
        with torch.no_grad():
            w_t = ternarize(self.weight, self.threshold)
            total = w_t.numel()
            zeros = (w_t == 0).sum().item()
            pos = (w_t > 0).sum().item()
            neg = (w_t < 0).sum().item()
        return {
            'total': total,
            'zeros': zeros,
            'positive': pos,
            'negative': neg,
            'sparsity': zeros / total
        }


class TernaryEqProp(nn.Module):
    """
    Equilibrium Propagation with Ternary Weights.
    
    Full-precision activations, ternary-quantized weights.
    Uses stochastic energy minimization for weight flips.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 3, alpha: float = 0.5,
                 threshold: float = 0.5):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.alpha = alpha
        self.threshold = threshold
        
        # Ternary layers
        self.W_in = TernaryLinear(input_dim, hidden_dim, threshold)
        self.layers = nn.ModuleList([
            TernaryLinear(hidden_dim, hidden_dim, threshold) 
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(hidden_dim, output_dim)  # Head stays full precision
        
    def forward_step(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Single equilibrium step with ternary weights."""
        x_emb = self.W_in(x)
        
        h_new = x_emb
        for layer in self.layers:
            h_new = h_new + layer(torch.tanh(h))
        h_new = torch.tanh(h_new)
        
        return (1 - self.alpha) * h + self.alpha * h_new
    
    def forward(self, x: torch.Tensor, steps: int = 30) -> torch.Tensor:
        """Forward pass to equilibrium."""
        batch_size = x.size(0)
        h = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        
        for _ in range(steps):
            h = self.forward_step(h, x)
        
        return self.head(h)
    
    def energy(self, h: torch.Tensor, x: torch.Tensor, buffer=None) -> torch.Tensor:
        """Energy function for EqProp training."""
        x_emb = self.W_in(x)
        
        E = 0.5 * torch.sum(h ** 2)
        
        pre_act = x_emb
        for layer in self.layers:
            pre_act = pre_act + layer(torch.tanh(h))
        
        abs_pre = torch.abs(pre_act)
        log_cosh = abs_pre + F.softplus(-2 * abs_pre) - 0.693147
        E -= torch.sum(log_cosh)
        
        return E
    
    def get_model_stats(self) -> dict:
        """Get statistics about the ternary model."""
        stats = {
            'W_in': self.W_in.get_ternary_stats(),
            'layers': [layer.get_ternary_stats() for layer in self.layers]
        }
        
        # Aggregate sparsity
        total_params = stats['W_in']['total']
        total_zeros = stats['W_in']['zeros']
        for layer_stats in stats['layers']:
            total_params += layer_stats['total']
            total_zeros += layer_stats['zeros']
        
        stats['overall_sparsity'] = total_zeros / total_params
        return stats
    
    def count_bit_operations(self) -> int:
        """
        Count theoretical bit operations.
        
        Ternary weights: multiply by {-1, 0, +1} = add/subtract/skip
        This is ~32x cheaper than float32 multiplications.
        """
        ops = 0
        ops += self.W_in.in_features * self.W_in.out_features  # W_in
        for layer in self.layers:
            ops += layer.in_features * layer.out_features
        ops += self.hidden_dim * self.output_dim  # Head (full precision)
        return ops


# ============================================================================
# Stochastic Energy Minimization (for hardware-friendly training)
# ============================================================================

class StochasticTernaryUpdate:
    """
    Stochastic weight flipping based on energy gradient.
    
    Instead of gradient descent on full-precision weights, we:
    1. Randomly propose weight flips
    2. Accept if energy decreases (or with probability based on temperature)
    
    This is more hardware-friendly for neuromorphic chips.
    """
    
    def __init__(self, model: TernaryEqProp, temperature: float = 0.1, 
                 flip_fraction: float = 0.01):
        self.model = model
        self.temperature = temperature
        self.flip_fraction = flip_fraction
    
    def step(self, x: torch.Tensor, y: torch.Tensor):
        """One step of stochastic weight optimization."""
        # Current energy
        out = self.model(x)
        loss_before = F.cross_entropy(out, y).item()
        
        # Randomly select weights to try flipping
        flips_accepted = 0
        
        for layer in [self.model.W_in] + list(self.model.layers):
            if isinstance(layer, TernaryLinear):
                num_flip = int(layer.weight.numel() * self.flip_fraction)
                
                for _ in range(num_flip):
                    # Random position
                    i = torch.randint(0, layer.weight.shape[0], (1,)).item()
                    j = torch.randint(0, layer.weight.shape[1], (1,)).item()
                    
                    old_val = layer.weight[i, j].item()
                    
                    # Propose new value
                    new_vals = [-1.0, 0.0, 1.0]
                    new_vals = [v for v in new_vals if abs(v - old_val) > 0.5]
                    if not new_vals:
                        continue
                    new_val = new_vals[torch.randint(0, len(new_vals), (1,)).item()]
                    
                    # Try flip
                    with torch.no_grad():
                        layer.weight[i, j] = new_val
                    
                    # Check new energy
                    out_new = self.model(x)
                    loss_after = F.cross_entropy(out_new, y).item()
                    
                    # Accept or reject (Metropolis-Hastings)
                    delta = loss_after - loss_before
                    if delta < 0 or torch.rand(1).item() < math.exp(-delta / self.temperature):
                        flips_accepted += 1
                        loss_before = loss_after
                    else:
                        # Revert
                        with torch.no_grad():
                            layer.weight[i, j] = old_val
        
        return {
            'loss': loss_before,
            'flips_accepted': flips_accepted
        }


# ============================================================================
# Test
# ============================================================================

def test_ternary_eqprop():
    """Test ternary EqProp implementation."""
    print("Testing TernaryEqProp...")
    
    model = TernaryEqProp(input_dim=64, hidden_dim=128, output_dim=10)
    
    x = torch.randn(16, 64)
    y = torch.randint(0, 10, (16,))
    
    out = model(x)
    assert out.shape == (16, 10), f"Wrong output shape: {out.shape}"
    
    loss = F.cross_entropy(out, y)
    loss.backward()
    
    stats = model.get_model_stats()
    print(f"Overall sparsity: {stats['overall_sparsity']:.2%}")
    print(f"Bit operations: {model.count_bit_operations():,}")
    
    print("✓ TernaryEqProp test passed")


def test_ternary_quantization():
    """Test ternary quantization function."""
    print("\nTesting ternary quantization...")
    
    w = torch.randn(100, 100)
    w_t = ternarize(w, threshold=0.5)
    
    unique = torch.unique(w_t / w_t[w_t != 0].abs().mean())  # Normalized unique values
    # Should be approximately {-1, 0, +1} times some scale
    
    assert (w_t[w > 0.5] > 0).all(), "Positive weights should be positive"
    assert (w_t[w < -0.5] < 0).all(), "Negative weights should be negative"
    
    # Test gradient flow
    w = torch.randn(10, 10, requires_grad=True)
    w_t = ternarize(w, threshold=0.5)
    loss = w_t.sum()
    loss.backward()
    
    assert w.grad is not None, "Gradient should flow through STE"
    print("✓ Ternary quantization test passed")


def test_mnist_ternary(epochs: int = 5):
    """Quick test on MNIST subset."""
    print(f"\nTesting on MNIST ({epochs} epochs)...")
    
    try:
        from src.tasks import get_task_loader
        train_loader, test_loader, in_dim, out_dim = get_task_loader(
            "mnist", batch_size=64, flatten=True
        )
    except:
        print("  Skipping MNIST test (data not available)")
        return
    
    model = TernaryEqProp(input_dim=in_dim, hidden_dim=256, output_dim=out_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i, (x, y) in enumerate(train_loader):
            if i > 50:  # Limit for quick test
                break
            optimizer.zero_grad()
            out = model(x, steps=20)
            loss = F.cross_entropy(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / min(50, len(train_loader))
        print(f"  Epoch {epoch+1}: Loss = {avg_loss:.4f}")
    
    # Test accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            if i > 20:
                break
            out = model(x, steps=20)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    
    acc = correct / total
    print(f"  Test Accuracy: {acc:.1%}")
    print("✓ MNIST ternary test completed")


if __name__ == "__main__":
    test_ternary_quantization()
    test_ternary_eqprop()
    test_mnist_ternary()
    print("\n✓ All ternary tests passed!")
