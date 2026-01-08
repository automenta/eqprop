# Novel Hybrid Algorithm Evaluation Results

**Evaluation Date**: 2026-01-08 10:40:51
**Duration**: 0.00 hours
**Dataset**: MNIST
**Training Samples**: 100

## Executive Summary

**Best Overall**: `pc_hybrid` (0.406 accuracy on mnist_200000)

## Detailed Results

### MNIST - 50000 parameters

| Rank | Algorithm | Test Acc | Train Acc | Epochs | Time (s) | Params |
|------|-----------|----------|-----------|--------|----------|--------|
| 1 | `pc_hybrid` | 0.198 | 0.380 | 5 | 1.2 | 302,258 |
| 2 | `mom_eq` | 0.145 | 0.140 | 3 | 1.2 | 150,742 |
| 3 | `ada_fa` | 0.122 | 0.100 | 7 | 1.2 | 301,158 |
| 4 | `cf_align` | 0.118 | 0.140 | 3 | 1.2 | 150,742 |
| 5 | `sparse_eq` | 0.116 | 0.080 | 2 | 1.3 | 150,742 |
| 6 | `eq_align` | 0.108 | 0.150 | 4 | 1.3 | 150,742 |
| 7 | `backprop` | 0.098 | 0.160 | 1 | 1.4 | 150,742 |
| 8 | `feedback_alignment` | 0.095 | 0.100 | 7 | 1.2 | 150,742 |

### MNIST - 100000 parameters

| Rank | Algorithm | Test Acc | Train Acc | Epochs | Time (s) | Params |
|------|-----------|----------|-----------|--------|----------|--------|
| 1 | `pc_hybrid` | 0.270 | 0.500 | 6 | 1.2 | 455,268 |
| 2 | `backprop` | 0.267 | 0.440 | 7 | 1.2 | 227,247 |
| 3 | `feedback_alignment` | 0.102 | 0.130 | 7 | 1.2 | 227,247 |
| 4 | `mom_eq` | 0.092 | 0.100 | 3 | 1.2 | 227,247 |
| 5 | `cf_align` | 0.082 | 0.070 | 3 | 1.2 | 227,247 |
| 6 | `ada_fa` | 0.081 | 0.090 | 7 | 1.2 | 454,038 |
| 7 | `sparse_eq` | 0.075 | 0.050 | 3 | 1.4 | 227,247 |
| 8 | `eq_align` | 0.073 | 0.100 | 4 | 1.3 | 227,247 |

### MNIST - 200000 parameters

| Rank | Algorithm | Test Acc | Train Acc | Epochs | Time (s) | Params |
|------|-----------|----------|-----------|--------|----------|--------|
| 1 | `pc_hybrid` | 0.406 | 0.700 | 7 | 1.2 | 703,578 |
| 2 | `backprop` | 0.341 | 0.590 | 7 | 1.2 | 351,402 |
| 3 | `mom_eq` | 0.149 | 0.130 | 3 | 1.2 | 351,402 |
| 4 | `feedback_alignment` | 0.142 | 0.120 | 7 | 1.2 | 351,402 |
| 5 | `sparse_eq` | 0.133 | 0.110 | 3 | 1.4 | 351,402 |
| 6 | `eq_align` | 0.126 | 0.140 | 4 | 1.2 | 351,402 |
| 7 | `ada_fa` | 0.094 | 0.120 | 7 | 1.2 | 702,162 |
| 8 | `cf_align` | 0.078 | 0.110 | 3 | 1.2 | 351,402 |

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
