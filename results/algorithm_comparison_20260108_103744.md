# Comprehensive Algorithm Comparison

**Generated**: 2026-01-08 10:37:44

## Summary

### MNIST @ 50000 parameters
**Winner**: backprop (0.759 accuracy)

### MNIST @ 100000 parameters
**Winner**: pc_hybrid (0.703 accuracy)

### MNIST @ 200000 parameters
**Winner**: backprop (0.770 accuracy)

## Detailed Results

### MNIST - 50000 params

| Rank | Algorithm | Test Acc | Train Acc | Epochs | Time (s) |
|------|-----------|----------|-----------|--------|----------|
| 1 | backprop | 0.759 | 0.752 | 5 | 7.9 |
| 2 | pc_hybrid | 0.751 | 0.753 | 5 | 7.8 |
| 3 | sparse_eq | 0.147 | 0.143 | 3 | 8.6 |
| 4 | feedback_alignment | 0.133 | 0.133 | 5 | 7.8 |
| 5 | mom_eq | 0.129 | 0.133 | 3 | 8.1 |
| 6 | cf_align | 0.094 | 0.101 | 3 | 8.0 |
| 7 | ada_fa | 0.081 | 0.080 | 5 | 7.8 |
| 8 | eq_align | 0.072 | 0.073 | 4 | 8.1 |

### MNIST - 100000 params

| Rank | Algorithm | Test Acc | Train Acc | Epochs | Time (s) |
|------|-----------|----------|-----------|--------|----------|
| 1 | pc_hybrid | 0.703 | 0.699 | 5 | 7.8 |
| 2 | backprop | 0.643 | 0.655 | 5 | 7.8 |
| 3 | cf_align | 0.137 | 0.140 | 3 | 8.0 |
| 4 | mom_eq | 0.101 | 0.112 | 3 | 8.1 |
| 5 | sparse_eq | 0.095 | 0.096 | 3 | 8.5 |
| 6 | feedback_alignment | 0.086 | 0.092 | 5 | 7.8 |
| 7 | eq_align | 0.061 | 0.064 | 4 | 8.0 |
| 8 | ada_fa | 0.059 | 0.061 | 5 | 7.7 |

### MNIST - 200000 params

| Rank | Algorithm | Test Acc | Train Acc | Epochs | Time (s) |
|------|-----------|----------|-----------|--------|----------|
| 1 | backprop | 0.770 | 0.766 | 5 | 7.8 |
| 2 | pc_hybrid | 0.681 | 0.690 | 5 | 7.8 |
| 3 | sparse_eq | 0.126 | 0.124 | 3 | 8.6 |
| 4 | cf_align | 0.110 | 0.108 | 3 | 8.1 |
| 5 | feedback_alignment | 0.099 | 0.102 | 5 | 7.8 |
| 6 | ada_fa | 0.090 | 0.091 | 5 | 7.8 |
| 7 | mom_eq | 0.083 | 0.072 | 3 | 8.2 |
| 8 | eq_align | 0.071 | 0.062 | 4 | 8.0 |
