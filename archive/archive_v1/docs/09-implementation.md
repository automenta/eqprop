# Implementation Guide

## Quick Start Checklist

- [ ] Clone repo, install dependencies
- [ ] Run `python test_gradient_equiv.py` — verify gradient matching
- [ ] Run `python train_mnist.py` — baseline training
- [ ] Check wandb dashboard for convergence curves
- [ ] Compare to `python train_mnist_bp.py` — BP baseline

---

## Core Classes

```python
class LoopedTransformerBlock(nn.Module):
    """Single weight-tied transformer block for equilibrium iteration."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        # Standard transformer components
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, h: Tensor, x: Tensor) -> Tensor:
        # h: [seq, batch, d_model], x: [seq, batch, d_model]
        h_norm = self.norm1(h)
        attn_out, _ = self.attn(h_norm, x, x)  # Cross-attend to input
        h = h + attn_out
        h_norm = self.norm2(h)
        h = h + self.ffn(h_norm)
        return h


class EquilibriumSolver:
    """Fixed-point solver with convergence monitoring."""
    
    def __init__(self, max_iters: int = 50, tol: float = 1e-5, damping: float = 0.9):
        self.max_iters = max_iters
        self.tol = tol
        self.damping = damping
        
    def solve(self, f: Callable, h0: Tensor, x: Tensor) -> Tuple[Tensor, int]:
        h = h0
        for t in range(self.max_iters):
            h_new = (1 - self.damping) * h + self.damping * f(h, x)
            residual = (h_new - h).norm()
            if residual < self.tol:
                return h_new, t + 1
            h = h_new
        return h, self.max_iters  # Did not converge


class EqPropTrainer:
    """Equilibrium Propagation training loop."""
    
    def __init__(self, model, solver, output_head, beta: float = 0.1, lr: float = 1e-3):
        self.model = model
        self.solver = solver
        self.output_head = output_head
        self.beta = beta
        self.optimizer = torch.optim.Adam(
            list(model.parameters()) + list(output_head.parameters()), 
            lr=lr
        )
        
    def train_step(self, x: Tensor, y: Tensor) -> Dict[str, float]:
        # Free phase
        h0 = torch.zeros_like(x)
        h_free, iters_free = self.solver.solve(self.model, h0, x)
        
        # Nudged phase
        def nudged_dynamics(h, x):
            h_new = self.model(h, x)
            y_pred = self.output_head(h_new.mean(dim=0))  # Pool over sequence
            nudge = self.beta * torch.autograd.grad(
                -F.cross_entropy(y_pred, y), h_new, retain_graph=True
            )[0]
            return h_new + nudge
        
        h_nudged, iters_nudged = self.solver.solve(nudged_dynamics, h_free.detach(), x)
        
        # Contrastive Hebbian update (simplified: use autodiff on difference)
        loss_proxy = ((h_nudged - h_free.detach()) ** 2).mean()
        self.optimizer.zero_grad()
        loss_proxy.backward()
        self.optimizer.step()
        
        # Metrics
        with torch.no_grad():
            y_pred = self.output_head(h_free.mean(dim=0))
            acc = (y_pred.argmax(-1) == y).float().mean()
            
        return {
            "loss": loss_proxy.item(),
            "accuracy": acc.item(),
            "iters_free": iters_free,
            "iters_nudged": iters_nudged
        }
```

---

## Hyperparameter Defaults

| Parameter | Default | Search Range | Notes |
|-----------|---------|--------------|-------|
| d_model | 128 | [64, 256, 512] | Start small |
| n_heads | 4 | [2, 4, 8] | d_model must be divisible |
| d_ff | 512 | 4 × d_model | Standard ratio |
| β (nudge) | 0.25 | [0.23, 0.5] | **Critical**: β≥0.23 for stability |
| α (damping) | 0.8 | [0.5, 1.0] | 0.8 optimal, 1.0 = no damping |
| ε (tolerance) | 1e-5 | [1e-6, 1e-3] | Trade-off: precision vs. speed |
| max_iters | 50 | [20, 100] | Set high initially |
| lr | 2e-3 | [1e-4, 1e-2] | EqProp stable with aggressive LR |
| dropout | 0.1 | [0, 0.2] | Regularization |

---

## Logging & Monitoring

Track per training step:
- `loss`, `accuracy`
- `iters_free`, `iters_nudged` (convergence speed)
- `grad_cosine_sim` (vs. BP baseline, sample periodically)
- `spectral_norm_jacobian` (stability diagnostic)

**Wandb/TensorBoard integration recommended.**

---

## Quick Commands

```bash
# Best config with larger model (CORRECTED β)
python train.py --d-model 256 --n-heads 8 --d-ff 1024 \
    --beta 0.25 --damping 0.8 --lr 0.002 --epochs 12 \
    --dropout 0.1 --compile

# Multi-seed validation
for s in 1 2 3 4 5; do
    python train.py --d-model 256 --beta 0.25 --damping 0.8 --lr 0.002 \
        --epochs 10 --seed $s --compile 2>&1 | tee seed_${s}.log
done

# Memory profiling
python profile_memory.py

# Gradient equivalence
python test_gradient_equiv.py

# CIFAR-10 (once implemented)
python train.py --dataset cifar10 --d-model 256 --epochs 50
```

---

## Implementation Learnings

Key insights from developing this codebase:

### Symmetric Mode Requirements

EqProp's gradient equivalence theorem requires symmetric Jacobians. The implementation enforces:

| Constraint | Implementation | Rationale |
|------------|----------------|------------|
| **W_out = W_q^T** | `SymmetricLinearAttention` uses `F.linear(out, w_q.weight.t())` | Attention output projection must be query weight transposed |
| **W_k = W_v** | Value computed as `V = K` (weight sharing) | Key/value share projections |
| **W2 = W1^T** | `SymmetricFFN` uses `F.linear(h, w1.weight.t())` | FFN output layer tied to input |

> [!IMPORTANT]
> Symmetric mode **requires** linear attention. Softmax attention breaks the energy formulation—the implementation raises `ValueError` if attempting `symmetric=True` with `attention_type='softmax'`.

### Update Strategy Patterns

Two update mechanisms are implemented in `src/updates.py`:

| Strategy | Theory | Use Case |
|----------|--------|----------|
| **MSEProxyUpdate** | Loss = (1/β) ‖model(h_free) − h_nudged‖² | Default; simple gradient descent compatible |
| **VectorFieldUpdate** | Backprop gradient vector v = (h_nudged − h_free)/β | Theoretically cleaner; accumulates gradients directly |
| **LocalHebbianUpdate** | Direct weight updates | O(1) memory; biologically plausible |

### Attention Hierarchy

Modular attention design (`src/attention.py`):

```
Attention (ABC) ← Base interface
├── SoftmaxAttention      # Standard nn.MultiheadAttention wrapper
├── LinearAttention       # Performer-style φ(Q)φ(K)ᵀV with ELU+1
└── SymmetricLinearAttention  # + weight constraints for EqProp
```

### Key Numerical Considerations

1. **LayerNorm placement**: Only in non-symmetric mode; symmetric mode uses `tanh` for bounded energy
2. **Feature map**: `φ(x) = elu(x) + 1` ensures positive values for linear attention
3. **Numerical stability**: `eps=1e-6` in attention denominators

### Critical Discovery: Tanh Saturation in Symmetric Mode

> [!WARNING]
> Symmetric mode causes **96.7% activation saturation** due to `tanh` bounds, killing gradient flow.

| Mode | Saturation (|h|>0.9) | Training Accuracy | Root Cause |
|------|------------------------|-------------------|------------|
| Symmetric | 96.7% | ~10% (failure) | tanh bounds → vanishing gradients |
| Non-symmetric | 0% | 92.7% | LayerNorm keeps activations healthy |

This explains why non-symmetric linear attention trains successfully while symmetric mode fails despite verified gradient equivalence.

---

## Project Structure

```
toreq/
├── docs/                    # Documentation (this directory)
├── src/
│   ├── attention.py         # Attention mechanisms
│   ├── ffn.py               # Feed-forward networks
│   ├── models.py            # LoopedTransformerBlock
│   ├── trainer.py           # EqPropTrainer
│   └── updates.py           # Update strategies
├── train.py                 # Main training script
├── train_mnist.py           # MNIST training
├── train_mnist_bp.py        # BP baseline
├── test_gradient_equiv.py   # Gradient verification
├── profile_memory.py        # Memory profiling
├── analyze_adaptive_compute.py  # Adaptive compute analysis
└── configs/                 # Configuration files
```
