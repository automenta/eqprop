# TorEqProp Research System

## Quick Start

```bash
# Quick 3-minute validation (smoke test)
python toreq.py

# 10-minute exploration
python toreq.py 10

# Deep exploration with more parameter variation
python toreq.py 20 --deep
```

## What It Does

Compares **TorEqProp** vs **Backprop** through rigorous hyperparameter optimization:

- ✅ Task-appropriate model sizes (d_model=16 for XOR, 64 for MNIST)
- ✅ Progress bars during each experiment
- ✅ Live parameter importance tracking
- ✅ Immediate actionable insights
- ✅ Full config transparency

## Output

- **Console**: Live results with insights
- **results/latest.md**: Full report with all trials

## Core Files

- `toreq.py` - THE research system (single entry point)
- `hyperopt_engine.py` - Hyperparameter optimization core
- `statistics.py` - Statistical analysis

## Task Tiers

| Task | d_model | Epochs | Time |
|------|---------|--------|------|
| xor, xor3 | 16-32 | 10 | ~10s |
| mnist, fashion | 64-128 | 3 | ~60s |
| cartpole | 32-64 | 30 | ~90s |
