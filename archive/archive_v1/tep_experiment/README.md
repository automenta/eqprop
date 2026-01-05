# TEP Experiment: Rigorous TEP vs BP Comparison

This package implements the **Toroidal Equilibrium Propagation (TEP)** experiment specification for rigorous, fair comparison against Backpropagation (BP).

## Quick Start

```bash
# Quick smoke test (~2 minutes)
python -m tep_experiment --smoke-test

# Phase 1: Rapid Signal Detection (~6-10 hours)
python -m tep_experiment --phase 1

# Full pipeline (all phases with gating)
python -m tep_experiment --full
```

## Core Principles

1. **Identical Treatment**: TEP and BP receive the same hyperparameter optimization budget and search space complexity
2. **Multi-Objective**: 4 objectives - accuracy (max), time (min), params (min), convergence speed (min)
3. **Phased Gating**: Automatic go/no-go decisions based on success criteria
4. **Statistical Rigor**: Multi-seed evaluation with Pareto front analysis

## Phases

### Phase 1: Rapid Signal Detection (6-10 hours)
- **Tasks**: XOR, 8×8 MNIST digits
- **Goal**: Detect any promising signal with minimal investment
- **Success**: TEP Pareto front shows advantage over BP

### Phase 2: Validation (12-24 hours)
- **Tasks**: Full MNIST 28×28, CartPole-v1
- **Triggered by**: Phase 1 success

### Phase 3: Comprehensive Benchmarking
- **Tasks**: CIFAR-10, sequence tasks, additional RL
- **Triggered by**: Phase 2 success

## Search Space

### Shared (TEP & BP)
| Parameter | Range |
|-----------|-------|
| Hidden Layers | 1-4 |
| Hidden Units | 4-512 (log) |
| Activation | tanh, relu |
| Learning Rate | 1e-4 to 1e-1 (log) |
| Batch Size | 32, 64, 128, 256 |

### TEP-Specific
| Parameter | Range |
|-----------|-------|
| Loop Radius | 1-8 |
| Nudging β | 0.01-0.5 (log) |
| Dampening γ | 0.5-0.99 |
| Equilibrium Iters | 5-50 |

## Monitoring

```bash
# Launch Optuna dashboard
python -m tep_experiment --dashboard
```

## Output

Results are saved to `tep_results/`:
- `phase1_xor_report.md` - Per-task reports
- `final_report.md` - Comprehensive comparison
- Pareto front visualizations (if matplotlib available)

## Architecture

```
tep_experiment/
├── __init__.py       # Package exports
├── config.py         # Search spaces, phase configs
├── tasks.py          # Task registry (XOR, 8x8 digits, MNIST, RL)
├── sampler.py        # Optuna samplers with identical treatment
├── objectives.py     # 4-objective evaluation
├── runner.py         # Unified trial execution with timeout
├── engine.py         # Main orchestration with NSGA-II
├── analysis.py       # Pareto front comparison
└── cli.py            # Command-line interface
```

## Dependencies

- PyTorch
- Optuna (`pip install optuna`)
- scikit-learn (for 8x8 digits)
- gymnasium (for RL tasks)
- optuna-dashboard (optional, for monitoring)
