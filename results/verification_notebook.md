# TorEqProp Verification Results

**Generated**: 2026-01-04 22:41:45


## Executive Summary

**Verification completed in 0.0 seconds.**

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
| 22 | Golden Reference Harness | ✅ | 100 | 0.0s |


**Seed**: 42 (deterministic)

**Reproducibility**: All experiments use fixed seeds for exact reproduction.

---


## Track 22: Golden Reference Harness


✅ **Status**: PASS | **Score**: 100.0/100 | **Time**: 0.0s


**Claim**: NumPy kernel matches PyTorch autograd to within numerical tolerance.

**Experiment**: Compare hidden states at each relaxation step.

| Metric | Value | Threshold |
|--------|-------|-----------|
| Max Hidden Diff | 1.79e-07 | < 1.00e-05 |
| Output Diff | 8.94e-08 | < 1.00e-05 |
| Steps Compared | 30 | - |

**Step-by-Step Comparison** (first/last steps):

| Step | Max Difference |
|------|----------------|
| 0 | 5.96e-08 |
| 1 | 1.19e-07 |
| 2 | 1.19e-07 |
| 3 | 1.19e-07 |
| 4 | 1.19e-07 |
| 28 | 1.19e-07 |
| 29 | 1.19e-07 |

**Purpose**: This harness enables safe optimization of the engine. Any new kernel
implementation must pass this test before deployment.

**Status**: ✅ VALIDATED - Safe to optimize


