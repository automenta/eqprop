# Evolution Run Report

**Task**: mnist
**Generations**: 1
**Population Size**: 2
**Total Time**: 4.9s

## Best Individual

- **Accuracy**: 0.6512
- **Lipschitz**: 1.161
- **Speed**: 3.9 iter/s
- **Memory**: 56 MB

### Configuration
```json
{
  "model_type": "looped_mlp",
  "depth": 77,
  "width": 256,
  "activation": "gelu",
  "normalization": "layernorm",
  "eq_steps": 86,
  "beta": 0.36381561307671373,
  "alpha": 0.17534187831011963,
  "use_sn": true,
  "n_power_iterations": 7,
  "use_residual": true,
  "residual_scale": 0.12811363267554587,
  "lr": 0.0002244697452509409,
  "num_heads": 8,
  "generation": 0,
  "parent_ids": [],
  "mutations_applied": []
}
```

## Evolution Trajectory

| Gen | Best Acc | Mean Acc | Best L |
|-----|----------|----------|--------|
| 1 | 0.6512 | 0.3256 | 1.161 |