# Language Modeling Scale Study Results
**Total experiments:** 32
**Generated:** 2026-01-06 21:19:24

## Dataset Size Scaling

| Size | Model | Perplexity | Accuracy | Params | Time |
|------|-------|------------|----------|--------|------|
| 1,000 | backprop | 49.45  | 0.179 | 475,054 | 5s |
| 1,000 | eqprop_looped_mlp | 15.63 (0.32×) | 0.206 | 77,614 | 9s |
| 1,000 | eqprop_recurrent_core | 19.61 (0.40×) | 0.194 | 177,070 | 54s |
| 1,000 | eqprop_full | 17.84 (0.36×) | 0.175 | 442,030 | 123s |
| 2,000 | backprop | 16.61  | 0.261 | 475,825 | 5s |
| 2,000 | eqprop_looped_mlp | 13.24 (0.80×) | 0.282 | 78,385 | 9s |
| 2,000 | eqprop_recurrent_core | 13.07 (0.79×) | 0.284 | 177,841 | 54s |
| 2,000 | eqprop_full | 13.12 (0.79×) | 0.291 | 442,801 | 121s |
| 5,000 | backprop | 11.19  | 0.325 | 476,853 | 5s |
| 5,000 | eqprop_looped_mlp | 12.81 (1.14×) | 0.272 | 79,413 | 9s |
| 5,000 | eqprop_recurrent_core | 14.64 (1.31×) | 0.254 | 178,869 | 53s |
| 5,000 | eqprop_full | 14.85 (1.33×) | 0.254 | 443,829 | 117s |
| 10,000 | backprop | 11.30  | 0.319 | 477,881 | 5s |
| 10,000 | eqprop_looped_mlp | 13.05 (1.16×) | 0.265 | 80,441 | 9s |
| 10,000 | eqprop_recurrent_core | 15.09 (1.34×) | 0.265 | 179,897 | 54s |
| 10,000 | eqprop_full | 14.76 (1.31×) | 0.265 | 444,857 | 117s |
| 10,000 | backprop | 10.36  | 0.349 | 345,401 | 4s |
| 10,000 | eqprop_recurrent_core | 14.23 (1.26×) | 0.284 | 179,897 | 53s |
| 10,000 | backprop | 11.69  | 0.313 | 345,401 | 3s |
| 10,000 | eqprop_recurrent_core | 17.58 (1.56×) | 0.238 | 179,897 | 36s |
| 10,000 | backprop | 12.16  | 0.293 | 345,401 | 3s |
| 10,000 | eqprop_recurrent_core | 19.41 (1.72×) | 0.225 | 179,897 | 37s |
| 10,000 | backprop | 11.86  | 0.293 | 345,401 | 3s |
| 10,000 | eqprop_recurrent_core | 16.15 (1.43×) | 0.239 | 179,897 | 36s |
| 20,000 | backprop | 10.51  | 0.319 | 478,138 | 5s |
| 20,000 | eqprop_looped_mlp | 13.51 (1.29×) | 0.267 | 80,698 | 9s |
| 20,000 | eqprop_recurrent_core | 15.33 (1.46×) | 0.254 | 180,154 | 53s |
| 20,000 | eqprop_full | 15.57 (1.48×) | 0.251 | 445,114 | 123s |
| 50,000 | backprop | 8.16  | 0.388 | 478,395 | 9s |
| 50,000 | eqprop_looped_mlp | 11.45 (1.40×) | 0.284 | 80,955 | 15s |
| 50,000 | eqprop_recurrent_core | 11.66 (1.43×) | 0.298 | 180,411 | 95s |
| 50,000 | eqprop_full | 11.16 (1.37×) | 0.300 | 445,371 | 213s |

## Key Findings

**Crossover point:** Around 1,000 characters
- EqProp (eqprop_looped_mlp): 15.63 PPL
- Backprop: 49.45 PPL
- Advantage: 3.16× better

