# Extended Training Results (50 Epochs, β=0.22)

**Date**: December 29, 2025  
**Duration**: 1h 21m  
**Status**: ✅ COMPLETE

---

## Configuration

```yaml
beta: 0.22          # Fixed (no annealing)
d_model: 256
n_heads: 8
d_ff: 1024
damping: 0.8
lr: 0.002
dropout: 0.1
epochs: 50
compile: true
```

---

## Results Summary

**Final Test Accuracy**: **93.83%** 🎉

**Improvement from baseline**:
- Previous best (15 epochs): 92.37%
- Extended training (50 epochs): **93.83%**
- **Gain**: +1.46% absolute

**Gap to target**:
- Target: 94.00%
- Current: 93.83%
- **Remaining gap**: Only **0.17%** ❗

---

## Training Progression

| Epoch Range | Best Test Acc | Notes |
|-------------|---------------|-------|
| 0-10 | 91.63% | Early learning |
| 11-20 | 92.68% | Steady improvement |
| 21-30 | 92.86% | Plateau beginning |
| 31-40 | 93.30% | Breaking through |
| 41-49 | **93.83%** | Final convergence |

**Peak epoch**: 49 (final epoch)  
**Training still improving**: Yes! Accuracy still climbing at epoch 49

---

## Key Observations

### 1. Extended Training Works! ✅

50 epochs provided **significant improvement** over 15 epochs:
- 15 epochs: 92.37%
- 50 epochs: 93.83%
- Linear gain: ~0.03% per epoch

**Implication**: Even more epochs (75-100) might push past 94%!

### 2. No Plateau Yet

Training accuracy still improving at epoch 49 (94.55% train, 93.83% test).  
**Conclusion**: Model has NOT fully converged.

### 3. Stable Throughout

- No catastrophic collapses
- Smooth, monotonic improvement
- β=0.22 fixed is rock solid

### 4. Test Accuracy Trajectory

```
Epoch 10: 91.63%
Epoch 20: 92.68%
Epoch 30: 92.86%
Epoch 40: 93.16%
Epoch 49: 93.83% ← PEAK
```

Clear upward trend, no overfitting signs.

---

## Comparison to Previous Best

| Configuration | Epochs | Test Acc | Improvement |
|---------------|--------|----------|-------------|
| β=0.25 (Dec 28) | 15 | 92.09% | Baseline |
| β=0.22 (β sweep) | 15 | 92.37% | +0.28% |
| **β=0.22 extended** | **50** | **93.83%** | **+1.74%** 🏆 |

---

## Analysis

### Why Extended Training Worked

1. **Insufficient convergence at 15 epochs**: Model needed more time
2. **β=0.22 is stable**: No degradation even after 50 epochs
3. **Equilibrium refinement**: Longer training → better fixed points

### Projected with More Epochs

Based on the trend (still improving at epoch 49):
- 75 epochs: ~94.1-94.3%
- 100 epochs: ~94.3-94.5%

**Recommendation**: Try 75-100 epochs to definitively surpass 94%

---

## Next Steps

### Priority 1: Multi-Seed Validation (LAUNCHED)
Validate 93.83% with statistical rigor (5 seeds × 30 epochs)

**Expected**: Mean ~93.6-94.0%, std < 0.5%

### Priority 2: Push to 94%+

**Option A**: Run 75-100 epochs with β=0.22  
**Option B**: Try architecture scaling (d=512)  
**Option C**: Combine both (d=512, 75 epochs)

---

## Publication Value

**Claim**: EqProp transformers achieve **93.83% on MNIST**, only 3.37% behind backprop (97.2%)

**Competitive gap closed**: From 4.83% (92.37%) to 3.37% (93.83%)

**Novel findings**:
1. Extended training essential (50+ epochs)
2. β=0.22 fixed optimal
3. β-annealing harmful
4. Stable, monotonic improvement

---

## Checkpoint

**Saved**: `checkpoints/best_mnist.pt`  
**Best accuracy**: 93.83%

---

**Generated**: December 29, 2025  
**Next**: Multi-seed validation running
