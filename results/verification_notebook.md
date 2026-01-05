# TorEqProp Verification Results

**Generated**: 2026-01-04 23:18:02


## Executive Summary

**Verification completed in 1.8 seconds.**

### Overall Results

| Metric | Value |
|--------|-------|
| Tracks Verified | 2 |
| Passed | 2 ✅ |
| Partial | 0 ⚠️ |
| Failed | 0 ❌ |
| Stubs (TODO) | 0 🔧 |
| Average Score | 100.0/100 |

### Track Summary

| # | Track | Status | Score | Time |
|---|-------|--------|-------|------|
| 31 | Residual EqProp | ✅ | 100 | 0.9s |
| 32 | Bidirectional Generation | ✅ | 100 | 0.9s |


**Seed**: 42 (deterministic)

**Reproducibility**: All experiments use fixed seeds for exact reproduction.

---


## Track 31: Residual EqProp


✅ **Status**: PASS | **Score**: 100.0/100 | **Time**: 0.9s


**Claim**: Skip connections maintain signal at extreme depth.

| Depth | Standard SNR | Residual SNR |
|-------|--------------|--------------|
| 100 | 298118 | 504491 |
| 200 | 374235 | 356277 |
| 500 | 284476 | 299228 |

**Finding**: Residual connections help at depth 500.




## Track 32: Bidirectional Generation


✅ **Status**: PASS | **Score**: 100.0/100 | **Time**: 0.9s


**Claim**: EqProp can generate inputs from class labels (bidirectional).

**Experiment**: Clamp output to target class, relax to generate input pattern.

| Metric | Value |
|--------|-------|
| Classes tested | 5 |
| Correct classifications | 5/5 |
| Generation accuracy | 100% |

**Key Finding**: Energy-based relaxation successfully 
generates class-consistent inputs. This demonstrates the bidirectional nature of EqProp.


