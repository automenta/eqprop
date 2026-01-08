# Novel Hybrid Algorithm Evaluation Results

**Evaluation Date**: 2026-01-08 10:43:11
**Duration**: 0.00 hours
**Dataset**: MNIST
**Training Samples**: 200

## Executive Summary

**Best Overall**: `backprop` (0.533 accuracy on mnist_200000)

## Detailed Results

### MNIST - 50000 parameters

| Rank | Algorithm | Test Acc | Train Acc | Epochs | Time (s) | Params |
|------|-----------|----------|-----------|--------|----------|--------|
| 1 | `pc_hybrid` | 0.482 | 0.635 | 10 | 1.4 | 302,258 |
| 2 | `backprop` | 0.461 | 0.635 | 4 | 1.5 | 150,742 |
| 3 | `sparse_eq` | 0.127 | 0.125 | 5 | 1.6 | 150,742 |
| 4 | `feedback_alignment` | 0.124 | 0.060 | 11 | 1.5 | 150,742 |
| 5 | `eqprop` | 0.116 | 0.120 | 6 | 1.5 | 150,742 |
| 6 | `leq_fa` | 0.105 | 0.100 | 11 | 1.4 | 150,742 |
| 7 | `eq_align` | 0.102 | 0.125 | 8 | 1.5 | 150,742 |
| 8 | `mom_eq` | 0.097 | 0.085 | 7 | 1.5 | 150,742 |
| 9 | `ada_fa` | 0.080 | 0.070 | 11 | 1.4 | 301,158 |
| 10 | `cf_align` | 0.066 | 0.065 | 6 | 1.4 | 150,742 |

### MNIST - 100000 parameters

| Rank | Algorithm | Test Acc | Train Acc | Epochs | Time (s) | Params |
|------|-----------|----------|-----------|--------|----------|--------|
| 1 | `backprop` | 0.503 | 0.630 | 11 | 1.4 | 227,247 |
| 2 | `pc_hybrid` | 0.485 | 0.630 | 11 | 1.4 | 455,268 |
| 3 | `feedback_alignment` | 0.198 | 0.185 | 11 | 1.4 | 227,247 |
| 4 | `mom_eq` | 0.131 | 0.120 | 6 | 1.5 | 227,247 |
| 5 | `eqprop` | 0.125 | 0.090 | 6 | 1.5 | 227,247 |
| 6 | `cf_align` | 0.103 | 0.115 | 6 | 1.4 | 227,247 |
| 7 | `sparse_eq` | 0.087 | 0.100 | 6 | 1.6 | 227,247 |
| 8 | `leq_fa` | 0.076 | 0.065 | 12 | 1.4 | 227,247 |
| 9 | `eq_align` | 0.075 | 0.070 | 8 | 1.4 | 227,247 |
| 10 | `ada_fa` | 0.065 | 0.090 | 12 | 1.4 | 454,038 |

### MNIST - 200000 parameters

| Rank | Algorithm | Test Acc | Train Acc | Epochs | Time (s) | Params |
|------|-----------|----------|-----------|--------|----------|--------|
| 1 | `backprop` | 0.533 | 0.680 | 11 | 1.4 | 351,402 |
| 2 | `pc_hybrid` | 0.526 | 0.660 | 11 | 1.4 | 703,578 |
| 3 | `eq_align` | 0.125 | 0.080 | 8 | 1.4 | 351,402 |
| 4 | `mom_eq` | 0.120 | 0.155 | 6 | 1.4 | 351,402 |
| 5 | `eqprop` | 0.101 | 0.115 | 6 | 1.5 | 351,402 |
| 6 | `feedback_alignment` | 0.100 | 0.085 | 12 | 1.4 | 351,402 |
| 7 | `cf_align` | 0.098 | 0.100 | 6 | 1.4 | 351,402 |
| 8 | `leq_fa` | 0.089 | 0.130 | 12 | 1.4 | 351,402 |
| 9 | `sparse_eq` | 0.067 | 0.065 | 5 | 1.6 | 351,402 |
| 10 | `ada_fa` | 0.040 | 0.055 | 12 | 1.4 | 702,162 |

## Algorithm Characteristics

| Algorithm | Type | Key Innovation |
|-----------|------|----------------|
| `backprop` | Baseline | Standard gradient descent |
| `eqprop` | Baseline | Contrastive Hebbian learning |
| `feedback_alignment` | Baseline | Random fixed feedback |
| `eq_align` | Hybrid | EqProp dynamics + FA training |
| `ada_fa` | Hybrid | Adaptive feedback evolution |
| `cf_align` | Hybrid | Contrastive via FA signals |
| `leq_fa` | Hybrid | Layer-wise local settling |
| `pc_hybrid` | Radical | Predictive coding + FA |
| `eg_fa` | Radical | Energy-guided updates |
| `sparse_eq` | Radical | Top-K sparse dynamics |
| `mom_eq` | Radical | Momentum-accelerated settling |
| `sto_fa` | Radical | Stochastic feedback dropout |
| `em_fa` | Radical | Energy minimization objective |
