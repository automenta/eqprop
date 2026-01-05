# TorEqProp Verification Results

**Generated**: 2026-01-05 11:19:34


## Executive Summary

**Verification completed in 1.7 seconds.**

### Overall Results

| Metric | Value |
|--------|-------|
| Tracks Verified | 1 |
| Passed | 1 ✅ |
| Partial | 0 ⚠️ |
| Failed | 0 ❌ |
| Stubs (TODO) | 0 🔧 |
| Average Score | 100.0/100 |

### Track Summary

| # | Track | Status | Score | Time |
|---|-------|--------|-------|------|
| 4 | Ternary Weights | ✅ | 100 | 1.7s |


**Seed**: 42 (deterministic)

**Reproducibility**: All experiments use fixed seeds for exact reproduction.

---


## Track 4: Ternary Weights


✅ **Status**: PASS | **Score**: 100.0/100 | **Time**: 1.7s


**Claim**: Ternary weights {-1, 0, +1} achieve high sparsity with full learning capacity.

**Method**: Ternary quantization with threshold=0.1 and L1 regularization (λ=0.0005).

| Metric | Value |
|--------|-------|
| Initial Loss | 15.629 |
| Final Loss | 0.014 |
| Loss Reduction | 99.9% |
| **Sparsity** | **70.4%** |
| Final Accuracy | 99.3% |

**Weight Distribution**:
| Layer | -1 | 0 | +1 |
|-------|----|----|----|
| W_in | 15% | 70% | 15% |
| W_rec | 11% | 79% | 11% |
| W_out | 19% | 62% | 19% |

**Hardware Impact**: 32× efficiency (no FPU needed), only ADD/SUBTRACT operations.


