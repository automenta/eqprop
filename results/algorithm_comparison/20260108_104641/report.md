# Novel Hybrid Algorithm Evaluation Results

**Evaluation Date**: 2026-01-08 11:48:12
**Duration**: 1.00 hours
**Dataset**: MNIST
**Training Samples**: 10,000

## Executive Summary

**Best Overall**: `pc_hybrid` (0.866 accuracy on mnist_200000)

## Detailed Results

### MNIST - 50000 parameters

| Rank | Algorithm | Test Acc | Train Acc | Epochs | Time (s) | Params |
|------|-----------|----------|-----------|--------|----------|--------|
| 1 | `backprop` | 0.862 | 0.854 | 77 | 94.6 | 150,742 |
| 2 | `pc_hybrid` | 0.855 | 0.845 | 75 | 94.6 | 302,258 |
| 3 | `mom_eq` | 0.334 | 0.335 | 46 | 94.8 | 150,742 |
| 4 | `sparse_eq` | 0.212 | 0.203 | 39 | 95.3 | 150,742 |
| 5 | `em_fa` | 0.139 | 0.127 | 79 | 94.4 | 150,742 |
| 6 | `leq_fa` | 0.107 | 0.108 | 78 | 94.5 | 150,742 |
| 7 | `eq_align` | 0.102 | 0.091 | 56 | 94.6 | 150,742 |
| 8 | `ada_fa` | 0.093 | 0.093 | 80 | 94.5 | 301,158 |
| 9 | `eqprop` | 0.093 | 0.089 | 44 | 94.7 | 150,742 |
| 10 | `cf_align` | 0.092 | 0.088 | 45 | 94.8 | 150,742 |
| 11 | `eg_fa` | 0.088 | 0.091 | 77 | 94.5 | 150,742 |
| 12 | `feedback_alignment` | 0.066 | 0.070 | 78 | 94.6 | 150,742 |
| 13 | `sto_fa` | 0.057 | 0.055 | 79 | 94.5 | 150,742 |

### MNIST - 100000 parameters

| Rank | Algorithm | Test Acc | Train Acc | Epochs | Time (s) | Params |
|------|-----------|----------|-----------|--------|----------|--------|
| 1 | `pc_hybrid` | 0.864 | 0.854 | 75 | 94.5 | 455,268 |
| 2 | `backprop` | 0.862 | 0.857 | 77 | 94.5 | 227,247 |
| 3 | `mom_eq` | 0.476 | 0.467 | 47 | 94.9 | 227,247 |
| 4 | `sparse_eq` | 0.233 | 0.233 | 40 | 95.1 | 227,247 |
| 5 | `em_fa` | 0.174 | 0.167 | 78 | 94.5 | 227,247 |
| 6 | `cf_align` | 0.145 | 0.139 | 46 | 94.7 | 227,247 |
| 7 | `leq_fa` | 0.123 | 0.125 | 79 | 94.5 | 227,247 |
| 8 | `eqprop` | 0.103 | 0.105 | 43 | 94.7 | 227,247 |
| 9 | `eq_align` | 0.103 | 0.106 | 57 | 94.7 | 227,247 |
| 10 | `ada_fa` | 0.100 | 0.102 | 79 | 94.5 | 454,038 |
| 11 | `feedback_alignment` | 0.094 | 0.094 | 77 | 94.5 | 227,247 |
| 12 | `eg_fa` | 0.094 | 0.107 | 78 | 94.4 | 227,247 |
| 13 | `sto_fa` | 0.072 | 0.070 | 79 | 94.4 | 227,247 |

### MNIST - 200000 parameters

| Rank | Algorithm | Test Acc | Train Acc | Epochs | Time (s) | Params |
|------|-----------|----------|-----------|--------|----------|--------|
| 1 | `pc_hybrid` | 0.866 | 0.860 | 74 | 94.5 | 703,578 |
| 2 | `backprop` | 0.862 | 0.855 | 77 | 94.5 | 351,402 |
| 3 | `mom_eq` | 0.397 | 0.378 | 47 | 94.9 | 351,402 |
| 4 | `sparse_eq` | 0.307 | 0.303 | 40 | 95.1 | 351,402 |
| 5 | `ada_fa` | 0.143 | 0.137 | 80 | 94.6 | 702,162 |
| 6 | `leq_fa` | 0.129 | 0.126 | 80 | 94.5 | 351,402 |
| 7 | `em_fa` | 0.112 | 0.110 | 78 | 94.6 | 351,402 |
| 8 | `eg_fa` | 0.091 | 0.088 | 78 | 94.4 | 351,402 |
| 9 | `eq_align` | 0.088 | 0.082 | 57 | 94.7 | 351,402 |
| 10 | `cf_align` | 0.080 | 0.085 | 46 | 94.7 | 351,402 |
| 11 | `eqprop` | 0.068 | 0.072 | 44 | 94.6 | 351,402 |
| 12 | `sto_fa` | 0.062 | 0.063 | 78 | 94.5 | 351,402 |
| 13 | `feedback_alignment` | 0.055 | 0.060 | 80 | 94.4 | 351,402 |

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
