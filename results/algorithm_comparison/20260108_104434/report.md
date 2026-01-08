# Novel Hybrid Algorithm Evaluation Results

**Evaluation Date**: 2026-01-08 10:45:32
**Duration**: 0.00 hours
**Dataset**: MNIST
**Training Samples**: 200

## Executive Summary

**Best Overall**: `backprop` (0.644 accuracy on mnist_200000)

## Detailed Results

### MNIST - 50000 parameters

| Rank | Algorithm | Test Acc | Train Acc | Epochs | Time (s) | Params |
|------|-----------|----------|-----------|--------|----------|--------|
| 1 | `pc_hybrid` | 0.582 | 0.735 | 11 | 1.4 | 302,258 |
| 2 | `backprop` | 0.575 | 0.690 | 4 | 1.5 | 150,742 |
| 3 | `leq_fa` | 0.157 | 0.150 | 12 | 1.3 | 150,742 |
| 4 | `eq_align` | 0.143 | 0.100 | 8 | 1.5 | 150,742 |
| 5 | `cf_align` | 0.116 | 0.125 | 6 | 1.4 | 150,742 |
| 6 | `eg_fa` | 0.116 | 0.130 | 12 | 1.4 | 150,742 |
| 7 | `feedback_alignment` | 0.114 | 0.110 | 12 | 1.5 | 150,742 |
| 8 | `sparse_eq` | 0.107 | 0.065 | 5 | 1.6 | 150,742 |
| 9 | `eqprop` | 0.106 | 0.095 | 6 | 1.5 | 150,742 |
| 10 | `sto_fa` | 0.106 | 0.100 | 11 | 1.4 | 150,742 |
| 11 | `ada_fa` | 0.101 | 0.105 | 12 | 1.4 | 301,158 |
| 12 | `mom_eq` | 0.101 | 0.120 | 7 | 1.5 | 150,742 |
| 13 | `em_fa` | 0.101 | 0.110 | 10 | 1.4 | 150,742 |

### MNIST - 100000 parameters

| Rank | Algorithm | Test Acc | Train Acc | Epochs | Time (s) | Params |
|------|-----------|----------|-----------|--------|----------|--------|
| 1 | `backprop` | 0.609 | 0.735 | 11 | 1.5 | 227,247 |
| 2 | `pc_hybrid` | 0.597 | 0.750 | 11 | 1.4 | 455,268 |
| 3 | `sparse_eq` | 0.185 | 0.185 | 6 | 1.6 | 227,247 |
| 4 | `mom_eq` | 0.152 | 0.165 | 6 | 1.5 | 227,247 |
| 5 | `sto_fa` | 0.149 | 0.195 | 11 | 1.4 | 227,247 |
| 6 | `em_fa` | 0.116 | 0.145 | 11 | 1.4 | 227,247 |
| 7 | `eq_align` | 0.116 | 0.135 | 8 | 1.4 | 227,247 |
| 8 | `cf_align` | 0.114 | 0.100 | 6 | 1.4 | 227,247 |
| 9 | `eqprop` | 0.104 | 0.110 | 6 | 1.5 | 227,247 |
| 10 | `feedback_alignment` | 0.101 | 0.110 | 11 | 1.4 | 227,247 |
| 11 | `leq_fa` | 0.101 | 0.100 | 12 | 1.4 | 227,247 |
| 12 | `ada_fa` | 0.081 | 0.085 | 12 | 1.4 | 454,038 |
| 13 | `eg_fa` | 0.079 | 0.075 | 12 | 1.4 | 227,247 |

### MNIST - 200000 parameters

| Rank | Algorithm | Test Acc | Train Acc | Epochs | Time (s) | Params |
|------|-----------|----------|-----------|--------|----------|--------|
| 1 | `backprop` | 0.644 | 0.785 | 12 | 1.4 | 351,402 |
| 2 | `pc_hybrid` | 0.610 | 0.745 | 11 | 1.4 | 703,578 |
| 3 | `ada_fa` | 0.121 | 0.120 | 12 | 1.5 | 702,162 |
| 4 | `sto_fa` | 0.120 | 0.120 | 12 | 1.4 | 351,402 |
| 5 | `eqprop` | 0.117 | 0.135 | 6 | 1.5 | 351,402 |
| 6 | `feedback_alignment` | 0.109 | 0.105 | 12 | 1.4 | 351,402 |
| 7 | `eq_align` | 0.107 | 0.125 | 8 | 1.4 | 351,402 |
| 8 | `sparse_eq` | 0.100 | 0.085 | 6 | 1.6 | 351,402 |
| 9 | `em_fa` | 0.094 | 0.110 | 12 | 1.4 | 351,402 |
| 10 | `mom_eq` | 0.085 | 0.090 | 7 | 1.5 | 351,402 |
| 11 | `cf_align` | 0.084 | 0.080 | 6 | 1.4 | 351,402 |
| 12 | `leq_fa` | 0.073 | 0.105 | 12 | 1.4 | 351,402 |
| 13 | `eg_fa` | 0.056 | 0.040 | 11 | 1.4 | 351,402 |

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
