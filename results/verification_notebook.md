# TorEqProp Verification Results

**Generated**: 2026-01-04 23:41:36


## Executive Summary

**Verification completed in 24.6 seconds.**

### Overall Results

| Metric | Value |
|--------|-------|
| Tracks Verified | 1 |
| Passed | 0 ✅ |
| Partial | 0 ⚠️ |
| Failed | 1 ❌ |
| Stubs (TODO) | 0 🔧 |
| Average Score | 40.0/100 |

### Track Summary

| # | Track | Status | Score | Time |
|---|-------|--------|-------|------|
| 33 | CIFAR-10 Benchmark | ❌ | 40 | 24.6s |


**Seed**: 42 (deterministic)

**Reproducibility**: All experiments use fixed seeds for exact reproduction.

---


## Track 33: CIFAR-10 Benchmark


❌ **Status**: FAIL | **Score**: 40.0/100 | **Time**: 24.6s


**Claim**: ConvEqProp achieves competitive accuracy on CIFAR-10.

**Experiment**: Train ConvEqProp and CNN baseline on CIFAR-10 subset with mini-batch training.

| Model | Train Acc | Test Acc | Gap to BP |
|-------|-----------|----------|-----------|
| ConvEqProp | 22.8% | 16.0% | +16.5% |
| CNN Baseline | 59.6% | 32.5% | — |

**Configuration**:
- Training samples: 500
- Test samples: 200
- Batch size: 32
- Epochs: 5
- Hidden channels: 16
- Equilibrium steps: 15

**Key Finding**: ConvEqProp trails CNN on CIFAR-10 
(needs more epochs/data).




### Areas for Improvement

- Increase epochs and data for full CIFAR-10 benchmark
