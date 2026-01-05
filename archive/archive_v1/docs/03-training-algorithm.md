# Training Algorithm

## Algorithm 1: TorEqProp Training Step

```
Input: x (input), y (target), β (nudge strength), ε (tolerance)
Output: Updated parameters θ

1. EQUILIBRIUM PHASE (Free)
   h ← 0  # or learned initialization
   repeat:
       h' ← (1-α)h + α·f_θ(h; x)
       if ‖h' - h‖ < ε: break
       h ← h'
   h* ← h
   A* ← {layer activations at h*}

2. EQUILIBRIUM PHASE (Nudged)  
   h ← h*
   repeat:
       h' ← (1-α)h + α·f_θ(h; x)
       ŷ ← OutputHead(h')
       h' ← h' + β · ∇_h L(ŷ, y)  # Nudge toward target
       if ‖h' - h‖ < ε: break
       h ← h'
   h^β ← h
   A^β ← {layer activations at h^β}

3. WEIGHT UPDATE (Contrastive Hebbian)
   for each layer l:
       ΔW_l ← (1/β) · (A^β_l ⊗ A^β_l - A*_l ⊗ A*_l)
       θ_l ← θ_l - η · ΔW_l
```

---

## Algorithm 1b: Purely Local Hebbian Nudging (Hardware-Friendly)

> [!TIP]
> For maximum biological plausibility and neuromorphic hardware compatibility, use **direct output clamping** instead of gradient-based nudging.

```
2b. NUDGED PHASE (Purely Local — No Autodiff)
    h ← h*
    repeat:
        h' ← (1-α)h + α·f_θ(h; x)
        
        # Direct output perturbation (no gradient computation)
        ŷ ← OutputHead(h')
        output_error ← (y_onehot - softmax(ŷ))  # Simple error signal
        h' ← h' + β · OutputHead.weight.T @ output_error  # Backproject error
        
        if ‖h' - h‖ < ε: break
        h ← h'
```

---

## Algorithm Comparison

| Aspect | Algorithm 1 (Autodiff) | Algorithm 1b (Purely Local) |
|--------|------------------------|------------------------------|
| Gradient computation | `torch.autograd.grad` | None |
| Error signal | ∇_h L | (y - ŷ) |
| Hardware compatible | GPU only | Neuromorphic (Loihi, SpiNNaker) |
| Biological plausibility | Medium | High |
| Theoretical guarantee | Exact (β→0 limit) | Approximate |

**Recommendation**: Use Algorithm 1 for validation (proves gradient equivalence), then demonstrate Algorithm 1b works comparably for hardware appeal.

---

## Update Strategy Patterns

Two update mechanisms are implemented in `src/updates.py`:

| Strategy | Theory | Use Case |
|----------|--------|----------|
| **MSEProxyUpdate** | Loss = (1/β) ‖model(h_free) − h_nudged‖² | Default; simple gradient descent compatible |
| **VectorFieldUpdate** | Backprop gradient vector v = (h_nudged − h_free)/β | Theoretically cleaner; accumulates gradients directly |
| **LocalHebbianUpdate** | Direct weight updates without autodiff | O(1) memory; biologically plausible |

### LocalHebbianUpdate (O(1) Memory)

**Status**: ACTIVATED - Pure Hebbian updates without autodiff for model parameters

**Implementation**:
- Removed MSE proxy fallback
- Direct weight updates: `W += lr * ΔW_hebbian`
- Only output head uses backprop
- Activation hooks capture free/nudged phases

```python
# Simplified Hebbian update rule
ΔW = (1/β) * (h_nudged @ h_nudged.T - h_free @ h_free.T)
W = W - lr * ΔW
```

---

## Key Numerical Considerations

1. **LayerNorm placement**: Only in non-symmetric mode; symmetric mode uses `tanh` for bounded energy
2. **Feature map**: `φ(x) = elu(x) + 1` ensures positive values for linear attention
3. **Numerical stability**: `eps=1e-6` in attention denominators

---

## Critical Discovery: Tanh Saturation in Symmetric Mode

> [!WARNING]
> Symmetric mode causes **96.7% activation saturation** due to `tanh` bounds, killing gradient flow.

| Mode | Saturation (|h|>0.9) | Training Accuracy | Root Cause |
|------|------------------------|-------------------|------------|
| Symmetric | 96.7% | ~10% (failure) | tanh bounds → vanishing gradients |
| Non-symmetric | 0% | 92.7% | LayerNorm keeps activations healthy |

This explains why non-symmetric linear attention trains successfully while symmetric mode fails despite verified gradient equivalence.
