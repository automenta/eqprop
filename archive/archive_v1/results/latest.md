# TorEqProp Research

Experiments: 4

## Results

| # | Algo | Task | Accuracy | Config |
|-|--|-|-|-|
| 1 | eqprop | xor | 0.507 | beta=0.22, damping=0.9, d_model=16 |
| 2 | bp | xor | 1.000 | lr=0.001, d_model=16, optimizer=adam |
| 3 | eqprop | mnist | 0.877 | beta=0.22, damping=0.9, d_model=64 |
| 4 | bp | mnist | 0.943 | lr=0.001, d_model=64, optimizer=adam |
