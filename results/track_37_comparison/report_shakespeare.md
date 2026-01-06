# Language Modeling Comparison: EqProp vs Backprop

**Dataset**: shakespeare

## Results Summary

| Model | Variant | Scale | Params | Perplexity | Accuracy | BPC | Time |
|-------|---------|-------|--------|------------|----------|-----|------|
| backprop | standard | 100% | 296,249 | 10.63 | 34.2% | 3.98 | 10.9s |
| eqprop | full | 100% | 295,993 | 10.74 | 33.0% | 3.43 | 192.4s |

## Parameter Efficiency Analysis

**Backprop baseline (100%)**: 10.63 perplexity
