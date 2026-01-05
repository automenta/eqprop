# Phase 1 Configuration - Single-Layer Focused Optimization

## Overview

Phase 1 has been **optimized for single-layer experiments** to maximize scientific value:

## Configuration

```python
Phase: "Rapid Signal Detection - Single Layer"
Task: digits_8x8 only (XOR removed as too trivial)
Architecture: n_hidden_layers = 1 (fixed)
Trials: 300 per algorithm
Budget: 8 hours total
```

## Rationale

### Why Single Layer?

**Creates optimization pressure:**
- Forces finding optimal `hidden_units` (4-512 range)
- Makes TEP parameters (β, γ, eq_iters) more critical
- Reveals parameter efficiency differences
- Prevents architecture from masking algorithmic differences

### Why Only digits_8x8?

**Better signal than XOR:**
- 64-dimensional input (vs 2 for XOR)
- 10 classes (vs 2)
- ~1,400 training samples
- **Produces accuracy variance** (not just 100%)
- Fast enough for 300 trials in 8 hours

**XOR removed because:**
- Too trivial (both algorithms hit 100%)
- Doesn't reveal optimization efficiency
- No pressure on parameter count/convergence speed

## Search Space Focus

With single layer, optimization focuses on:

| Parameter | Range | Impact |
|-----------|-------|--------|
| `hidden_units` | 4-512 (log) | **Capacity vs efficiency trade-off** |
| `lr` | 1e-4 to 1e-1 | Convergence speed |
| `batch_size` | 32-256 | Memory & stability |
| `activation` | tanh/relu | Expressiveness |

### TEP-Specific (Critical in Single Layer)

| Parameter | Range | Impact |
|-----------|-------|--------|
| `beta` | 0.01-0.5 | **Nudging strength** |
| `gamma` | 0.5-0.99 | **Stability control** |
| `eq_iters` | 5-50 | **Computation budget** |
| `loop_radius` | 1-8 | State reuse |

## Expected Outcomes

### Pareto Front Objectives

1. **Accuracy**: Expect 70-95% range on digits_8x8
2. **Wall Time**: Varied by eq_iters and hidden_units
3. **Param Count**: Wide range (100 to 50k+)
4. **Convergence**: Steps to 90% of best accuracy

### Hypothesis Testing

**TEP advantages may show in:**
- Fewer parameters for same accuracy
- Faster convergence with optimal β
- Better accuracy/time trade-offs

**BP advantages may show in:**
- Higher peak accuracy
- More stable across hyperparameters

## Phase 2 Expansion

After Phase 1 success, Phase 2 will:
- **Unlock multi-layer**: 1-4 layers
- Add harder tasks: MNIST 28×28, CartPole
- Use best configs from Phase 1 as seeds

## Running Phase 1

```bash
# Full Phase 1 (8 hours, 300 trials each)
python -m tep_experiment --phase 1

# Reduced budget for testing
python -m tep_experiment --phase 1 --n-trials 50 --budget-hours 2

# Monitor progress
python -m tep_experiment --dashboard
```

## Implementation Details

Phase-aware constraint in `sampler.py`:
```python
if phase == 1:
    config["n_hidden_layers"] = 1  # Force single layer
# Phases 2-3 use full 1-4 range
```

This preserves the search space parameter for later phases while focusing Phase 1.
