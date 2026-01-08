# Novel Hybrid Algorithm Evaluation Results

**Evaluation Date**: 2026-01-08 10:42:36
**Duration**: 0.00 hours
**Dataset**: MNIST
**Training Samples**: 100

## Executive Summary

**Best Overall**: `pc_hybrid` (0.682 accuracy on mnist_200000)

## Detailed Results

### MNIST - 50000 parameters

| Rank | Algorithm | Test Acc | Train Acc | Epochs | Time (s) | Params |
|------|-----------|----------|-----------|--------|----------|--------|
| 1 | `pc_hybrid` | 0.574 | 0.750 | 12 | 1.3 | 302,258 |
| 2 | `backprop` | 0.344 | 0.380 | 1 | 1.4 | 150,742 |
| 3 | `feedback_alignment` | 0.119 | 0.110 | 13 | 1.3 | 150,742 |
| 4 | `ada_fa` | 0.115 | 0.150 | 14 | 1.3 | 301,158 |
| 5 | `sparse_eq` | 0.109 | 0.100 | 4 | 1.5 | 150,742 |
| 6 | `cf_align` | 0.108 | 0.090 | 6 | 1.3 | 150,742 |
| 7 | `eq_align` | 0.105 | 0.090 | 8 | 1.4 | 150,742 |
| 8 | `mom_eq` | 0.101 | 0.140 | 6 | 1.4 | 150,742 |

### MNIST - 100000 parameters

| Rank | Algorithm | Test Acc | Train Acc | Epochs | Time (s) | Params |
|------|-----------|----------|-----------|--------|----------|--------|
| 1 | `pc_hybrid` | 0.613 | 0.780 | 12 | 1.3 | 455,268 |
| 2 | `backprop` | 0.562 | 0.740 | 13 | 1.3 | 227,247 |
| 3 | `mom_eq` | 0.129 | 0.140 | 4 | 2.1 | 227,247 |
| 4 | `feedback_alignment` | 0.113 | 0.120 | 13 | 1.3 | 227,247 |
| 5 | `sparse_eq` | 0.090 | 0.120 | 4 | 2.1 | 227,247 |
| 6 | `cf_align` | 0.090 | 0.140 | 5 | 1.4 | 227,247 |
| 7 | `ada_fa` | 0.074 | 0.090 | 14 | 1.3 | 454,038 |
| 8 | `eq_align` | 0.065 | 0.040 | 8 | 1.4 | 227,247 |

### MNIST - 200000 parameters

| Rank | Algorithm | Test Acc | Train Acc | Epochs | Time (s) | Params |
|------|-----------|----------|-----------|--------|----------|--------|
| 1 | `pc_hybrid` | 0.682 | 0.860 | 13 | 1.3 | 703,578 |
| 2 | `backprop` | 0.559 | 0.750 | 6 | 1.4 | 351,402 |
| 3 | `mom_eq` | 0.126 | 0.150 | 6 | 1.4 | 351,402 |
| 4 | `cf_align` | 0.121 | 0.100 | 6 | 1.3 | 351,402 |
| 5 | `ada_fa` | 0.115 | 0.140 | 14 | 1.3 | 702,162 |
| 6 | `eq_align` | 0.093 | 0.080 | 8 | 1.3 | 351,402 |
| 7 | `sparse_eq` | 0.085 | 0.070 | 5 | 1.5 | 351,402 |
| 8 | `feedback_alignment` | 0.054 | 0.020 | 13 | 1.3 | 351,402 |

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
