# Best Configuration from Sweep

## Hyperparameter Search Results

**Date**: 2025-12-28
**Model**: LoopedTransformerBlock (L=1, linear attention)
**Dataset**: MNIST

### Search Space
- β ∈ {0.05, 0.1, 0.2}
- damping ∈ {0.8, 0.9, 0.95}
- lr ∈ {5e-4, 1e-3, 2e-3}
- d_model = 128, n_heads = 4, d_ff = 512

### Top 5 Configurations

| Rank | β | Damping | LR | Test Acc | Notes |
|------|---|---------|----|-----------| -----|
| 1 | 0.20 | 0.80 | 0.002 | **94.04%** | Best overall |
| 2 | 0.10 | 0.90 | 0.002 | 92.59% | More stable convergence |
| 3 | 0.20 | 0.90 | 0.001 | 92.81% | Conservative |
| 4 | 0.05 | 0.95 | 0.001 | 92.11% | Lowest β |
| 5 | 0.05 | 0.80 | 0.002 | 92.06% | Fast learning |

### Best Configuration
```yaml
d_model: 128
n_heads: 4
d_ff: 512
beta: 0.2
damping: 0.8
lr: 0.002
batch_size: 128
max_iters: 50
```

**Result**: 94.04% test accuracy after 5 epochs

### Key Insights

1. **Higher β works better**: β=0.2 outperformed β=0.05 and β=0.1
   - Larger nudge strength provides stronger gradient signal
   - Still maintains gradient equivalence (theoretical guarantee holds for small β)

2. **Lower damping is optimal**: damping=0.8 beat 0.9 and 0.95
   - Allows faster convergence (less dampening of updates)
   - No instability issues observed

3. **Higher learning rate beneficial**: lr=0.002 > 0.001 > 0.0005
   - EqProp can handle aggressive learning rates
   - Likely due to implicit regularization from equilibrium dynamics

### Next Steps

- [x] Run with d_model=256 for +1-2% boost → target: 95%+
- [ ] Add 2nd transformer block (L=2)
- [ ] Validate across 5 seeds for confidence intervals
- [ ] Compare convergence speed vs baseline

### Memory Profiling Results

Current implementation (MSE proxy):
- d_model=128: 1.04× worse than BP
- d_model=256: 1.06× worse than BP  
- d_model=512: 1.12× worse than BP

**Issue**: Still using autodiff fallback, not true O(1) memory yet.
**Solution**: Need to fully implement direct Hebbian updates (bypassing .backward())
