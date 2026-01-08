# Evolution Report

**Generated**: 2026-01-08T09:33:04.398477
**Total Evaluations**: 1890
**Successful**: 1713
**Runs**: 1

## Best Per Model Type

| Model Type | Best Accuracy | Mean ± Std | Count |
|------------|---------------|------------|-------|
| looped_mlp | 0.9132 | 0.7175 ± 0.2283 | 366 |
| transformer | 0.2007 | 0.1359 ± 0.0424 | 330 |
| hebbian | 0.8376 | 0.4353 ± 0.2492 | 407 |
| feedback_alignment | 0.9139 | 0.7097 ± 0.1762 | 360 |

## Best Per Task

| Task | Best Accuracy | Best Model | Mean ± Std | Count |
|------|---------------|------------|------------|-------|
| fashion | 0.8255 | feedback_alignment | 0.7128 ± 0.1869 | 315 |
| shakespeare | 0.2007 | transformer | 0.1359 ± 0.0424 | 315 |
| mnist | 0.9139 | feedback_alignment | 0.7737 ± 0.2130 | 315 |
| cifar10 | 0.4032 | looped_mlp | 0.3198 ± 0.0897 | 315 |

## Multi-Task Breakthroughs

No breakthroughs yet.

# Multi-Objective Analysis

## Pareto-Optimal Solutions by Task

### CIFAR10

**Pareto-optimal solutions**: 10 / 209 (4.8%)

| Solution | Accuracy | Params | Time (s) | Acc/Param | Acc/Sec | Model |
|----------|----------|--------|----------|-----------|---------|-------|
| looped_mlp | 0.4032 | 855K | 16.2 | 0.47 | 0.025 | looped_mlp |
| looped_mlp | 0.3919 | 2015K | 13.7 | 0.19 | 0.029 | looped_mlp |
| hebbian | 0.3897 | 225K | 14.2 | 1.73 | 0.028 | hebbian |
| hebbian | 0.3783 | 388K | 14.0 | 0.97 | 0.027 | hebbian |
| hebbian | 0.3754 | 533K | 14.0 | 0.70 | 0.027 | hebbian |
| hebbian | 0.3690 | 172K | 14.3 | 2.14 | 0.026 | hebbian |
| hebbian | 0.3634 | 276K | 14.0 | 1.31 | 0.026 | hebbian |
| hebbian | 0.3524 | 266K | 14.0 | 1.32 | 0.025 | hebbian |
| feedback_ali | 0.3265 | 119K | 98.8 | 2.73 | 0.003 | feedback_alignment |
| hebbian | 0.2467 | 127K | 16.3 | 1.93 | 0.015 | hebbian |

**Best by Objective:**
- **Accuracy leader**: 0.4032
- **Smallest model**: 119K params
- **Best efficiency**: 2.73 acc/M-param
- **Fastest training**: 13.7s

**Trade-off Correlations:**
- Accuracy vs Params: 0.052
- Accuracy vs Time: -0.002

### FASHION

**Pareto-optimal solutions**: 12 / 179 (6.7%)

| Solution | Accuracy | Params | Time (s) | Acc/Param | Acc/Sec | Model |
|----------|----------|--------|----------|-----------|---------|-------|
| feedback_ali | 0.8255 | 12265K | 105.0 | 0.07 | 0.008 | feedback_alignment |
| looped_mlp | 0.8249 | 669K | 14.2 | 1.23 | 0.058 | looped_mlp |
| looped_mlp | 0.8214 | 929K | 9.2 | 0.88 | 0.090 | looped_mlp |
| looped_mlp | 0.8205 | 269K | 35.2 | 3.05 | 0.023 | looped_mlp |
| looped_mlp | 0.8182 | 747K | 8.9 | 1.09 | 0.092 | looped_mlp |
| looped_mlp | 0.8118 | 312K | 11.2 | 2.60 | 0.072 | looped_mlp |
| looped_mlp | 0.8108 | 169K | 10.9 | 4.77 | 0.074 | looped_mlp |
| feedback_ali | 0.7963 | 117K | 94.7 | 6.76 | 0.008 | feedback_alignment |
| feedback_ali | 0.7681 | 117K | 91.3 | 6.53 | 0.008 | feedback_alignment |
| hebbian | 0.7396 | 330K | 9.5 | 2.23 | 0.078 | hebbian |

**Best by Objective:**
- **Accuracy leader**: 0.8255
- **Smallest model**: 27K params
- **Best efficiency**: 26.45 acc/M-param
- **Fastest training**: 8.9s

**Trade-off Correlations:**
- Accuracy vs Params: 0.198
- Accuracy vs Time: 0.262

### MNIST

**Pareto-optimal solutions**: 15 / 179 (8.4%)

| Solution | Accuracy | Params | Time (s) | Acc/Param | Acc/Sec | Model |
|----------|----------|--------|----------|-----------|---------|-------|
| feedback_ali | 0.9139 | 15634K | 110.3 | 0.06 | 0.008 | feedback_alignment |
| feedback_ali | 0.9137 | 9350K | 84.4 | 0.10 | 0.011 | feedback_alignment |
| looped_mlp | 0.9132 | 169K | 17.0 | 5.37 | 0.054 | looped_mlp |
| looped_mlp | 0.9130 | 312K | 11.4 | 2.92 | 0.080 | looped_mlp |
| looped_mlp | 0.9126 | 169K | 11.1 | 5.37 | 0.082 | looped_mlp |
| looped_mlp | 0.9101 | 126K | 37.1 | 7.18 | 0.025 | looped_mlp |
| looped_mlp | 0.9059 | 933K | 10.8 | 0.97 | 0.084 | looped_mlp |
| looped_mlp | 0.9047 | 1027K | 9.4 | 0.88 | 0.096 | looped_mlp |
| looped_mlp | 0.9015 | 747K | 8.9 | 1.21 | 0.102 | looped_mlp |
| looped_mlp | 0.9006 | 916K | 8.8 | 0.98 | 0.102 | looped_mlp |

**Best by Objective:**
- **Accuracy leader**: 0.9139
- **Smallest model**: 46K params
- **Best efficiency**: 15.15 acc/M-param
- **Fastest training**: 8.8s

**Trade-off Correlations:**
- Accuracy vs Params: 0.118
- Accuracy vs Time: 0.043

### SHAKESPEARE

**Pareto-optimal solutions**: 5 / 15 (33.3%)

| Solution | Accuracy | Params | Time (s) | Acc/Param | Acc/Sec | Model |
|----------|----------|--------|----------|-----------|---------|-------|
| transformer | 0.2007 | 704K | 300.9 | 0.28 | 0.001 | transformer |
| transformer | 0.1762 | 71K | 309.0 | 2.45 | 0.001 | transformer |
| transformer | 0.1598 | 561K | 302.2 | 0.28 | 0.001 | transformer |
| transformer | 0.1596 | 313K | 230.4 | 0.51 | 0.001 | transformer |
| transformer | 0.1557 | 46K | 304.7 | 3.37 | 0.001 | transformer |

**Best by Objective:**
- **Accuracy leader**: 0.2007
- **Smallest model**: 46K params
- **Best efficiency**: 3.37 acc/M-param
- **Fastest training**: 230.4s

**Trade-off Correlations:**
- Accuracy vs Params: 0.450
- Accuracy vs Time: -0.168


## Top Configurations

### #1: feedback_alignment - 0.9139
```json
{
  "model_type": "feedback_alignment",
  "depth": 25,
  "width": 864,
  "activation": "gelu",
  "normalization": "spectral",
  "eq_steps": 94,
  "beta": 0.4556530785769102,
  "alpha": 0.24778580476633527,
  "use_sn": true,
  "n_power_iterations": 6,
  "use_residual": false,
  "residual_scale": 0.49906743483800087,
  "lr": 0.00016530881887905473,
  "num_heads": 1,
  "generation": 80,
  "parent_ids": [
    140732793926640
  ],
  "mutations_applied": []
}
```

### #2: feedback_alignment - 0.9137
```json
{
  "model_type": "feedback_alignment",
  "depth": 15,
  "width": 763,
  "activation": "relu",
  "normalization": "spectral",
  "eq_steps": 45,
  "beta": 0.4556530785769102,
  "alpha": 0.2342244108667752,
  "use_sn": true,
  "n_power_iterations": 5,
  "use_residual": true,
  "residual_scale": 0.49906743483800087,
  "lr": 0.00016530881887905473,
  "num_heads": 1,
  "generation": 84,
  "parent_ids": [
    140732797016048
  ],
  "mutations_applied": [
    "alpha\u21920.23"
  ]
}
```

### #3: looped_mlp - 0.9132
```json
{
  "model_type": "looped_mlp",
  "depth": 57,
  "width": 175,
  "activation": "silu",
  "normalization": "layernorm",
  "eq_steps": 32,
  "beta": 0.34292523860815594,
  "alpha": 0.6723257036926413,
  "use_sn": true,
  "n_power_iterations": 7,
  "use_residual": true,
  "residual_scale": 0.002920292269270472,
  "lr": 0.0003762115816206861,
  "num_heads": 1,
  "generation": 12,
  "parent_ids": [
    140732834018512
  ],
  "mutations_applied": [
    "depth\u219257",
    "width\u2192175",
    "eq_steps\u219232",
    "res_scale\u21920.00"
  ]
}
```

### #4: looped_mlp - 0.9130
```json
{
  "model_type": "looped_mlp",
  "depth": 67,
  "width": 288,
  "activation": "silu",
  "normalization": "layernorm",
  "eq_steps": 14,
  "beta": 0.39624867298094557,
  "alpha": 0.6723257036926413,
  "use_sn": true,
  "n_power_iterations": 7,
  "use_residual": true,
  "residual_scale": 0.10319409348412917,
  "lr": 0.0003762115816206861,
  "num_heads": 1,
  "generation": 8,
  "parent_ids": [
    140732795936496
  ],
  "mutations_applied": [
    "eq_steps\u219214",
    "n_power_iter\u21927",
    "res_scale\u21920.10"
  ]
}
```

### #5: looped_mlp - 0.9126
```json
{
  "model_type": "looped_mlp",
  "depth": 49,
  "width": 175,
  "activation": "tanh",
  "normalization": "layernorm",
  "eq_steps": 14,
  "beta": 0.4118560367820182,
  "alpha": 0.5457697826460018,
  "use_sn": true,
  "n_power_iterations": 8,
  "use_residual": false,
  "residual_scale": 0.10753909591927494,
  "lr": 0.00035718952422621414,
  "num_heads": 1,
  "generation": 16,
  "parent_ids": [
    140732795462704
  ],
  "mutations_applied": [
    "depth\u219249",
    "alpha\u21920.55"
  ]
}
```

### #6: looped_mlp - 0.9101
```json
{
  "model_type": "looped_mlp",
  "depth": 79,
  "width": 136,
  "activation": "silu",
  "normalization": "spectral",
  "eq_steps": 97,
  "beta": 0.4479521379160973,
  "alpha": 0.7227067976590095,
  "use_sn": true,
  "n_power_iterations": 4,
  "use_residual": false,
  "residual_scale": 0.9275625510065076,
  "lr": 0.0003943010640432873,
  "num_heads": 1,
  "generation": 10,
  "parent_ids": [
    140732797412176
  ],
  "mutations_applied": [
    "width\u2192136",
    "n_power_iter\u21924",
    "residual\u2192False"
  ]
}
```

### #7: looped_mlp - 0.9096
```json
{
  "model_type": "looped_mlp",
  "depth": 79,
  "width": 256,
  "activation": "silu",
  "normalization": "spectral",
  "eq_steps": 97,
  "beta": 0.4479521379160973,
  "alpha": 0.7227067976590095,
  "use_sn": true,
  "n_power_iterations": 4,
  "use_residual": true,
  "residual_scale": 0.0656377726014224,
  "lr": 0.0003943010640432873,
  "num_heads": 8,
  "generation": 7,
  "parent_ids": [
    140732795934032
  ],
  "mutations_applied": [
    "beta\u21920.448"
  ]
}
```

### #8: looped_mlp - 0.9092
```json
{
  "model_type": "looped_mlp",
  "depth": 79,
  "width": 256,
  "activation": "silu",
  "normalization": "layernorm",
  "eq_steps": 97,
  "beta": 0.4888643200157217,
  "alpha": 0.6425883679657858,
  "use_sn": true,
  "n_power_iterations": 4,
  "use_residual": true,
  "residual_scale": 0.11415002848838049,
  "lr": 0.00015585996754699754,
  "num_heads": 8,
  "generation": 9,
  "parent_ids": [
    140732797412176
  ],
  "mutations_applied": [
    "beta\u21920.489",
    "n_power_iter\u21924",
    "res_scale\u21920.11"
  ]
}
```

### #9: looped_mlp - 0.9092
```json
{
  "model_type": "looped_mlp",
  "depth": 51,
  "width": 639,
  "activation": "tanh",
  "normalization": "layernorm",
  "eq_steps": 14,
  "beta": 0.4564528447557125,
  "alpha": 0.6714683532690773,
  "use_sn": true,
  "n_power_iterations": 8,
  "use_residual": true,
  "residual_scale": 0.16245814534257821,
  "lr": 0.0005522346372252953,
  "num_heads": 1,
  "generation": 25,
  "parent_ids": [
    140732795119984
  ],
  "mutations_applied": [
    "beta\u21920.456",
    "alpha\u21920.67"
  ]
}
```

### #10: feedback_alignment - 0.9081
```json
{
  "model_type": "feedback_alignment",
  "depth": 36,
  "width": 763,
  "activation": "gelu",
  "normalization": "spectral",
  "eq_steps": 94,
  "beta": 0.4556530785769102,
  "alpha": 0.24778580476633527,
  "use_sn": true,
  "n_power_iterations": 6,
  "use_residual": false,
  "residual_scale": 0.49906743483800087,
  "lr": 0.00020216957784523765,
  "num_heads": 1,
  "generation": 72,
  "parent_ids": [
    140732793121360
  ],
  "mutations_applied": [
    "depth\u219236",
    "alpha\u21920.25",
    "n_power_iter\u21926"
  ]
}
```
