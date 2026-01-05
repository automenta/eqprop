# TEP Experiment Report: Phase 1 - digits_8x8

**Generated**: 2025-12-30 21:54:33

## Summary

| Metric | TEP | BP |
|--------|-----|-----|
| Pareto Points | 24 | 28 |
| Best Accuracy | 0.9667 | 0.9944 |
| Best Time (s) | 1.73 | 3.87 |

## Winner: **BP**

## Top 5 TEP Configurations

### 1. Accuracy: 0.9667
- Time: 24.00s
- Params: 155,110
- Config: `{
  "algorithm": "tep",
  "n_hidden_layers": 1,
  "hidden_units": 110,
  "activation": "tanh",
  "lr": 0.025502980701628937,
  "batch_size": 32,
  "beta": 0.010218376758008088,
  "gamma": 0.8233921825567477,
  "eq_iters": 37,
  "tolerance": 0.00012894414393074176,
  "attention_type": "linear",
  "symmetric": false,
  "_valid_n_heads": 2
}`

### 2. Accuracy: 0.9667
- Time: 16.16s
- Params: 3,208
- Config: `{
  "algorithm": "tep",
  "n_hidden_layers": 1,
  "hidden_units": 13,
  "activation": "relu",
  "lr": 0.025502980701628937,
  "batch_size": 64,
  "beta": 0.09553057578887411,
  "gamma": 0.7413336699712433,
  "eq_iters": 26,
  "tolerance": 0.0002870875348195468,
  "attention_type": "linear",
  "symmetric": false,
  "_valid_n_heads": 1
}`

### 3. Accuracy: 0.9500
- Time: 12.11s
- Params: 3,622
- Config: `{
  "algorithm": "tep",
  "n_hidden_layers": 1,
  "hidden_units": 14,
  "activation": "tanh",
  "lr": 0.025502980701628937,
  "batch_size": 64,
  "beta": 0.010218376758008088,
  "gamma": 0.8995760999428688,
  "eq_iters": 37,
  "tolerance": 0.0002870875348195468,
  "attention_type": "linear",
  "symmetric": false,
  "_valid_n_heads": 2
}`

### 4. Accuracy: 0.9444
- Time: 4.97s
- Params: 39,862
- Config: `{
  "algorithm": "tep",
  "n_hidden_layers": 1,
  "hidden_units": 54,
  "activation": "relu",
  "lr": 0.025502980701628937,
  "batch_size": 256,
  "beta": 0.09553057578887411,
  "gamma": 0.710207073413759,
  "eq_iters": 21,
  "tolerance": 0.0002870875348195468,
  "attention_type": "linear",
  "symmetric": false,
  "_valid_n_heads": 2
}`

### 5. Accuracy: 0.9194
- Time: 10.40s
- Params: 1,498
- Config: `{
  "algorithm": "tep",
  "n_hidden_layers": 1,
  "hidden_units": 8,
  "activation": "tanh",
  "lr": 0.01877320105765493,
  "batch_size": 128,
  "beta": 0.10502105436744279,
  "gamma": 0.8469555631200623,
  "eq_iters": 42,
  "tolerance": 0.00010677482709481354,
  "attention_type": "linear",
  "symmetric": false,
  "_valid_n_heads": 8
}`

## Top 5 BP Configurations

### 1. Accuracy: 0.9944
- Time: 21.09s
- Params: 7,192
- Config: `{
  "algorithm": "bp",
  "n_hidden_layers": 1,
  "hidden_units": 21,
  "activation": "tanh",
  "lr": 0.0074112997810832455,
  "batch_size": 128,
  "optimizer": "adam",
  "weight_decay": 0.0,
  "_valid_n_heads": 1
}`

### 2. Accuracy: 0.9917
- Time: 32.20s
- Params: 2,299,342
- Config: `{
  "algorithm": "bp",
  "n_hidden_layers": 1,
  "hidden_units": 434,
  "activation": "tanh",
  "lr": 0.0007644457600399744,
  "batch_size": 64,
  "optimizer": "adamw",
  "weight_decay": 0.0,
  "_valid_n_heads": 2
}`

### 3. Accuracy: 0.9917
- Time: 41.66s
- Params: 2,299,342
- Config: `{
  "algorithm": "bp",
  "n_hidden_layers": 1,
  "hidden_units": 434,
  "activation": "tanh",
  "lr": 0.00016736010167825804,
  "batch_size": 64,
  "optimizer": "adamw",
  "weight_decay": 0.001,
  "_valid_n_heads": 2
}`

### 4. Accuracy: 0.9917
- Time: 11.31s
- Params: 2,299,342
- Config: `{
  "algorithm": "bp",
  "n_hidden_layers": 1,
  "hidden_units": 434,
  "activation": "tanh",
  "lr": 0.0007644457600399744,
  "batch_size": 256,
  "optimizer": "adam",
  "weight_decay": 0.0001,
  "_valid_n_heads": 2
}`

### 5. Accuracy: 0.9917
- Time: 11.44s
- Params: 2,299,342
- Config: `{
  "algorithm": "bp",
  "n_hidden_layers": 1,
  "hidden_units": 434,
  "activation": "tanh",
  "lr": 0.0003310418956324517,
  "batch_size": 256,
  "optimizer": "adamw",
  "weight_decay": 0.001,
  "_valid_n_heads": 2
}`

