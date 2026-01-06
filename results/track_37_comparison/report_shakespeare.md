# Language Modeling Comparison: EqProp vs Backprop

**Dataset**: shakespeare

## Results Summary

| Model | Variant | Scale | Params | Perplexity | Accuracy | BPC | Time |
|-------|---------|-------|--------|------------|----------|-----|------|
| backprop | standard | 90% | 1,916,225 | 4.34 | 56.0% | 2.13 | 105.0s |
| backprop | standard | 75% | 1,615,745 | 4.35 | 56.2% | 2.13 | 98.1s |
| backprop | standard | 100% | 2,175,041 | 4.37 | 56.4% | 2.13 | 108.5s |
| backprop | standard | 50% | 1,091,585 | 4.50 | 54.9% | 2.18 | 86.8s |
| eqprop | attention_only | 100% | 2,174,529 | 4.69 | 54.6% | 2.25 | 971.2s |
| eqprop | hybrid | 100% | 2,174,529 | 4.73 | 53.7% | 2.24 | 410.3s |
| eqprop | attention_only | 90% | 1,915,745 | 4.74 | 54.3% | 2.25 | 963.8s |
| eqprop | attention_only | 75% | 1,615,305 | 4.75 | 54.0% | 2.25 | 936.8s |
| eqprop | hybrid | 90% | 1,915,745 | 4.80 | 52.8% | 2.29 | 408.4s |
| eqprop | hybrid | 75% | 1,615,305 | 4.87 | 52.7% | 2.29 | 400.1s |
| eqprop | attention_only | 50% | 1,091,225 | 4.90 | 52.5% | 2.32 | 911.9s |
| eqprop | hybrid | 50% | 1,091,225 | 5.08 | 51.4% | 2.34 | 376.0s |
| eqprop | full | 100% | 2,174,529 | 5.80 | 47.4% | 2.54 | 2069.0s |
| eqprop | full | 90% | 1,915,745 | 5.91 | 47.2% | 2.56 | 2046.2s |
| eqprop | full | 75% | 1,615,305 | 6.19 | 45.5% | 2.64 | 2010.2s |
| eqprop | recurrent_core | 100% | 593,217 | 6.42 | 44.8% | 2.68 | 688.3s |
| eqprop | recurrent_core | 90% | 525,425 | 6.58 | 44.0% | 2.72 | 692.8s |
| eqprop | full | 50% | 1,091,225 | 6.70 | 43.4% | 2.74 | 1902.9s |
| eqprop | recurrent_core | 75% | 446,445 | 6.93 | 42.7% | 2.79 | 678.4s |
| eqprop | recurrent_core | 50% | 307,685 | 7.37 | 40.6% | 2.88 | 638.2s |
| eqprop | looped_mlp | 90% | 177,665 | 11.89 | 27.0% | 3.58 | 108.4s |
| eqprop | looped_mlp | 100% | 197,697 | 11.89 | 27.0% | 3.58 | 107.9s |
| eqprop | looped_mlp | 75% | 154,065 | 11.91 | 27.1% | 3.58 | 108.9s |
| eqprop | looped_mlp | 50% | 111,665 | 11.92 | 26.9% | 3.58 | 107.3s |

## Parameter Efficiency Analysis

**Backprop baseline (100%)**: 4.37 perplexity
