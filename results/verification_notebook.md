# TorEqProp Verification Results

**Generated**: 2026-01-05 22:11:56


## Executive Summary

**Verification completed in 10.1 seconds.**

### Overall Results

| Metric | Value |
|--------|-------|
| Tracks Verified | 2 |
| Passed | 1 ✅ |
| Partial | 1 ⚠️ |
| Failed | 0 ❌ |
| Stubs (TODO) | 0 🔧 |
| Average Score | 87.5/100 |

### Track Summary

| # | Track | Status | Score | Time |
|---|-------|--------|-------|------|
| 0 | Framework Validation | ✅ | 100 | 0.5s |
| 37 | Language Modeling | ⚠️ | 75 | 9.7s |


**Seed**: 42 (deterministic)

**Reproducibility**: All experiments use fixed seeds for exact reproduction.

---


## Track 0: Framework Validation


✅ **Status**: PASS | **Score**: 100.0/100 | **Time**: 0.5s

🧪 **Evidence Level**: Smoke Test


**Framework Self-Test Results**

| Test | Status |
|------|--------|
| Cohen's d calculation | ✅ |
| Statistical significance (t-tests) | ✅ |
| Evidence classification | ✅ |
| Human-readable interpretations | ✅ |
| Statistical comparison formatting | ✅ |
| Reproducibility hashing | ✅ |

**Tests Passed**: 6/6

**Purpose**: This track validates the validation framework itself, ensuring all statistical
functions work correctly before running model validation tracks.


**Limitations**:
- Framework-level test only, does not validate EqProp models



## Track 37: Language Modeling


⚠️ **Status**: PARTIAL | **Score**: 75.0/100 | **Time**: 9.7s

🧪 **Evidence Level**: Smoke Test


**Claim**: EqProp matches or exceeds Backprop in language modeling while potentially using fewer parameters.

**Dataset**: Shakespeare
**Config**: hidden=128, layers=3, epochs=15

## Results

| Model | Params | Perplexity | Accuracy |
|-------|--------|------------|----------|
| backprop_100 | 419,509 | 18.36 | 25.4% |
| eqprop_full_100 | 419,253 | 26.00 | 14.1% |
| eqprop_recurrent_core_100 | 154,293 | 25.12 | 16.1% |
| eqprop_full_90 | 370,013 | 25.99 | 13.8% |
| eqprop_recurrent_core_90 | 136,973 | 26.98 | 14.2% |


**Analysis**:
- Backprop baseline: 18.36 perplexity
- Best EqProp: 25.12 perplexity (eqprop_recurrent_core_100)
- EqProp matches Backprop: ❌ No
- EqProp more efficient: ❌ Not demonstrated

**Note**: Run full experiment with `python experiments/language_modeling_comparison.py --epochs 50` for complete analysis.




### Areas for Improvement

- Tune EqProp hyperparameters (eq_steps, alpha, lr)
- Test smaller EqProp models (75% params)
