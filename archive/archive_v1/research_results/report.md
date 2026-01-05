# TorEqProp Research Report

> Generated: 2025-12-30 16:52:44
> Runtime: 4.8 minutes | Experiments: 6

## Results Summary

| Task | EqProp Best | BP Best | Winner |
|------|-------------|---------|--------|
| mnist | 0.167 | 0.926 | BP |
| xor | 0.761 | 1.000 | BP |
| xor3 | 0.652 | 1.000 | BP |

## Best EqProp Configurations

### xor (0.761)
```yaml
algorithm: eqprop
attention_type: linear
beta: 0.25
d_model: 256
damping: 0.9
lr: 0.0005
max_iters: 50
symmetric: True
tol: 0.0001
update_mode: vector_field
```

### xor3 (0.652)
```yaml
algorithm: eqprop
attention_type: linear
beta: 0.2
d_model: 256
damping: 0.95
lr: 0.0005
max_iters: 100
symmetric: True
tol: 0.0001
update_mode: vector_field
```

### mnist (0.167)
```yaml
algorithm: eqprop
attention_type: linear
beta: 0.25
d_model: 64
damping: 0.7
lr: 0.002
max_iters: 50
symmetric: False
tol: 0.0001
update_mode: vector_field
```

## All Trials

| # | Algo | Task | Accuracy | Time | Key Config |
|---|------|------|----------|------|------------|
| 1 | eqprop | xor | 0.761 | 12.4s | beta=0.25, damping=0.9, d_model=256, attention_type=linear, lr=0.0005 |
| 2 | bp | xor | 1.000 | 22.8s | lr=0.0005, optimizer=adamw, d_model=64 |
| 3 | eqprop | xor3 | 0.652 | 12.4s | beta=0.2, damping=0.95, d_model=256, attention_type=linear, lr=0.0005 |
| 4 | bp | xor3 | 1.000 | 23.3s | lr=0.0001, optimizer=adam, d_model=256 |
| 5 | eqprop | mnist | 0.167 | 59.2s | beta=0.25, damping=0.7, d_model=64, attention_type=linear, lr=0.002 |
| 6 | bp | mnist | 0.926 | 154.9s | lr=0.0001, optimizer=adam, d_model=64 |
