# Architecture

## Looped Transformer Block

```
┌─────────────────────────────────────────────┐
│                                             │
│  x (input) ──┐                              │
│              ▼                              │
│  ┌─────────────────────┐                    │
│  │   h_t (hidden)      │◄────────────┐      │
│  └──────────┬──────────┘             │      │
│             ▼                        │      │
│  ┌─────────────────────┐             │      │
│  │  LayerNorm          │             │      │
│  └──────────┬──────────┘             │      │
│             ▼                        │      │
│  ┌─────────────────────┐             │      │
│  │  MultiHeadAttn(h,x) │             │      │
│  └──────────┬──────────┘             │      │
│             ▼                        │      │
│  ┌─────────────────────┐             │      │
│  │  + Residual         │             │      │
│  └──────────┬──────────┘             │      │
│             ▼                        │      │
│  ┌─────────────────────┐             │      │
│  │  LayerNorm          │             │      │
│  └──────────┬──────────┘             │      │
│             ▼                        │      │
│  ┌─────────────────────┐             │      │
│  │  FFN                │             │      │
│  └──────────┬──────────┘             │      │
│             ▼                        │      │
│  ┌─────────────────────┐             │      │
│  │  + Residual ────────┼─────────────┘      │
│  └──────────┬──────────┘                    │
│             ▼                               │
│         h_{t+1}                             │
│             │                               │
│             ▼ (iterate until ‖h-h'‖<ε)      │
│          h* ──► Output Head ──► ŷ           │
│                                             │
└─────────────────────────────────────────────┘
```

---

## Multi-Layer Toroid Configurations

While the single-block design is the simplest, TorEqProp naturally extends to **multi-layer toroids** where multiple distinct blocks iterate together:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MULTI-LAYER TOROID (L=3)                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   x ──► [Block 1] ──► [Block 2] ──► [Block 3] ──► h_t               │
│              ▲                                     │                │
│              └─────────────────────────────────────┘                │
│                         (iterate until convergence)                 │
│                                                                     │
│   Parameters: θ₁, θ₂, θ₃ (distinct, not weight-tied across blocks)  │
│   Each iteration: h_{t+1} = f₃(f₂(f₁(h_t, x)))                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Architecture Search Space

| Configuration | Blocks (L) | Weight Sharing | Parameters | Expressiveness |
|--------------|------------|----------------|------------|----------------|
| Single-block | 1 | — | 1× | Baseline |
| Multi-block independent | 2-4 | None | L× | Higher |
| Multi-block tied | 2-4 | Pairs share | ~L/2× | Regularized |
| Hierarchical | 2-4 | Local+Global | Variable | Task-dependent |

### Tradeoffs

- More blocks → more expressive, but harder to converge (larger Jacobian)
- Shared weights across blocks → regularization, easier convergence, fewer params
- The "sweet spot" L is an empirical question this research will answer

---

## Convergence Dynamics

$$h_{t+1} = (1-\alpha)h_t + \alpha \cdot f_\theta(h_t; x)$$

where $\alpha \in (0,1]$ is the damping factor. Convergence criterion:

$$\|h_{t+1} - h_t\|_2 < \epsilon \quad \text{or} \quad t > T_{\max}$$

**Required property**: Spectral radius $\rho(J_f) < 1$ where $J_f = \frac{\partial f}{\partial h}$

### Convergence Aids

To accelerate and stabilize equilibrium-finding, incorporate these techniques from the start:

| Technique | Description | Benefit | Reference |
|-----------|-------------|---------|------------|
| **Anderson Acceleration** | Extrapolate from last k iterates | 2-5× faster convergence | Bai et al. 2019 (DEQ) |
| **Learned Initialization** | Predict h₀ from x via small net | Skip early iterations | Universal Transformers |
| **Timestep Encoding** | Inject iteration count t into layers | Helps model "know" progress | Yang et al. 2024 |
| **Spectral Normalization** | Constrain weight norms | Guarantee contraction | Miyato et al. 2018 |

**Implementation Priority**:
1. Start with simple damped iteration (baseline)
2. Add Anderson acceleration if convergence is slow (>30 iters)
3. Add learned initialization if still slow
4. Use spectral norm if convergence is unstable

---

## Attention Variants for Guaranteed Convergence

Softmax attention may violate contraction. Include **linear attention** as a fallback with guaranteed convergence:

| Attention Type | Contraction Guarantee | Expressiveness | Use Case |
|----------------|----------------------|----------------|----------|
| **Softmax** | ❌ No (high Jacobian) | High | Primary (if stable) |
| **Linear (Performer)** | ✅ Yes (bounded) | Medium | Fallback baseline |
| **Cosine Similarity** | ✅ Yes (Lipschitz) | Medium | Alternative |
| **Gated Linear** | ✅ Yes | Medium-High | Best compromise |

```python
# Performer-style linear attention (guaranteed contraction)
def linear_attention(Q, K, V, eps=1e-6):
    """φ(x) = elu(x) + 1 feature map."""
    phi = lambda x: F.elu(x) + 1
    Q_prime, K_prime = phi(Q), phi(K)
    KV = torch.einsum('bnd,bnv->bdv', K_prime, V)
    Z = torch.einsum('bnd,nd->bn', Q_prime, K_prime.sum(dim=0)) + eps
    return torch.einsum('bnd,bdv->bnv', Q_prime, KV) / Z.unsqueeze(-1)
```

**Experimental Strategy**: Run all experiments with both softmax and linear attention. If softmax fails to converge, linear attention results are the MVP.

---

## Attention Hierarchy

Modular attention design (`src/attention.py`):

```
Attention (ABC) ← Base interface
├── SoftmaxAttention      # Standard nn.MultiheadAttention wrapper
├── LinearAttention       # Performer-style φ(Q)φ(K)ᵀV with ELU+1
└── SymmetricLinearAttention  # + weight constraints for EqProp
```

### Symmetric Mode Requirements

EqProp's gradient equivalence theorem requires symmetric Jacobians. The implementation enforces:

| Constraint | Implementation | Rationale |
|------------|----------------|------------|
| **W_out = W_q^T** | `SymmetricLinearAttention` uses `F.linear(out, w_q.weight.t())` | Attention output projection must be query weight transposed |
| **W_k = W_v** | Value computed as `V = K` (weight sharing) | Key/value share projections |
| **W2 = W1^T** | `SymmetricFFN` uses `F.linear(h, w1.weight.t())` | FFN output layer tied to input |

> [!IMPORTANT]
> Symmetric mode **requires** linear attention. Softmax attention breaks the energy formulation—the implementation raises `ValueError` if attempting `symmetric=True` with `attention_type='softmax'`.
