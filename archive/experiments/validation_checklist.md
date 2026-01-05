# Pre-Publication Validation Checklist

> **Purpose**: Ensure all claims are validated before paper submission  
> **Last Updated**: 2025-12-31

---

## Quick Status

| Claim | Validated | Seeds | Confidence | Status |
|-------|-----------|-------|------------|--------|
| Spectral Norm L < 1 | ✅ | 3 | High | Ready |
| Competitive Accuracy | ✅ | 1 | Medium | Need more seeds |
| β-Annealing Instability | ✅ | 1 | Medium | Need more seeds |
| Optimal β = 0.22 | ✅ | 1 | Medium | Need more seeds |
| O(1) Memory | ⚠️ | - | Low | Not yet working |

---

## Claim 1: Spectral Normalization Maintains L < 1

### Required Evidence

- [x] Lipschitz constant measured for untrained models
- [x] Lipschitz constant measured after training WITHOUT spectral norm
- [x] Lipschitz constant measured after training WITH spectral norm
- [ ] Results from 3+ random seeds
- [ ] Statistical significance test

### Validation Commands

```bash
# Run Lipschitz analysis
python scripts/test_spectral_norm_all.py

# Multi-seed validation
for seed in 42 123 456; do
    python scripts/test_spectral_norm_all.py --seed $seed
done
```

### Expected Results

| Model | L (Untrained) | L (Trained, no SN) | L (Trained, SN) |
|-------|---------------|-------------------|-----------------|
| LoopedMLP | ~0.69 | ~0.74 | **<0.60** |
| ToroidalMLP | ~0.70 | **>1.0** ❌ | **<0.60** |
| ModernEqProp | ~0.54 | **>5.0** ❌ | **<0.60** |

### Pass Criteria

- [ ] All models with SN have L < 1.0
- [ ] At least one model without SN has L > 1.0
- [ ] Results consistent across 3+ seeds

---

## Claim 2: Competitive Accuracy (97.50%)

### Required Evidence

- [x] ModernEqProp accuracy matches or exceeds documented results
- [x] Comparison against Backprop baseline
- [ ] 5-seed validation with confidence intervals
- [ ] Training curves plot

### Validation Commands

```bash
# Single run
python scripts/competitive_benchmark.py

# Multi-seed validation
python scripts/competitive_benchmark.py --seeds 5 --output results/accuracy_multiseed.json
```

### Expected Results

| Model | Mean Acc | Std | 95% CI |
|-------|----------|-----|--------|
| BackpropMLP | ~97.5% | ~0.3% | [97.0%, 98.0%] |
| ModernEqProp | ~97.0% | ~0.5% | [96.5%, 97.5%] |

### Pass Criteria

- [ ] ModernEqProp mean accuracy > 96.5%
- [ ] Gap to Backprop < 2.0%
- [ ] Standard deviation < 1.0%

---

## Claim 3: β-Annealing Causes Instability

### Required Evidence

- [x] β-annealing shows collapse (documented in archive)
- [x] Fixed β = 0.20 is stable (documented in archive)
- [ ] Reproducible with current codebase
- [ ] 3-seed validation

### Validation Commands

```bash
# Test β-annealing collapse
python scripts/beta_annealing_test.py --anneal-from 0.3 --anneal-to 0.20 --epochs 20

# Test fixed β stability
python scripts/fixed_beta_test.py --beta 0.20 --epochs 20
```

### Expected Results

| Configuration | Epochs to Collapse | Final Accuracy |
|--------------|-------------------|----------------|
| β-annealing 0.3→0.20 | 10-15 | <20% (collapsed) |
| β = 0.20 fixed | Never | >90% |

### Pass Criteria

- [ ] β-annealing shows clear collapse (accuracy drops >50%)
- [ ] Fixed β maintains accuracy throughout training
- [ ] Results reproducible across seeds

---

## Claim 4: Optimal β = 0.22

### Required Evidence

- [x] β sweep results show 0.22 as optimal (documented)
- [ ] All β values [0.20, 0.26] are stable
- [ ] 0.22 consistently beats other β values
- [ ] 3-seed validation

### Validation Commands

```bash
# Full β sweep
python scripts/beta_sweep.py --betas 0.18,0.20,0.21,0.22,0.23,0.24,0.25,0.28 --epochs 15 --seeds 3
```

### Expected Results

| β | Accuracy (mean ± std) | Rank |
|---|----------------------|------|
| 0.22 | 92.0% ± 0.5% | 1 |
| 0.25 | 91.5% ± 0.5% | 2-3 |
| 0.20 | 91.0% ± 0.5% | 4-5 |

### Pass Criteria

- [ ] β = 0.22 has highest mean accuracy
- [ ] All β values in [0.20, 0.26] complete training
- [ ] Standard deviation < 1.0% for all β values

---

## Claim 5: O(1) Memory Training (Optional for Paper A)

### Required Evidence

- [ ] LocalHebbianUpdate achieves >80% accuracy
- [ ] Memory scales as O(1) with depth
- [ ] Memory < Backprop at large scales

### Current Status

⚠️ **Not yet validated** - LocalHebbianUpdate not learning (9.72% accuracy)

### Blocking Issues

1. Update mismatch with equilibrium states
2. Missing Wh (recurrent weight) updates
3. Sign/scaling alignment needed

### Path Forward

See [docs/LOCAL_HEBBIAN.md](file:///home/me/toreq/docs/LOCAL_HEBBIAN.md) for detailed debugging plan.

---

## Pre-Submission Checklist

### Code Quality

- [ ] All scripts run without errors
- [ ] Results are reproducible
- [ ] Random seeds are fixed and documented
- [ ] Dependencies listed in requirements.txt

### Documentation

- [ ] README.md is up to date
- [ ] RESEARCH_STATUS.md reflects current state
- [ ] All claims have evidence files

### Data

- [ ] Results saved in JSON format
- [ ] Training curves saved
- [ ] Model checkpoints available

### Figures

- [ ] Main results table generated
- [ ] Training curves plotted
- [ ] Lipschitz analysis visualized
- [ ] β sweep results plotted

### Paper

- [ ] Abstract reviewed
- [ ] All tables filled with real data
- [ ] All figures embedded
- [ ] References complete
- [ ] Co-authors reviewed

---

## Validation Pipeline

Run the complete validation pipeline:

```bash
# Step 1: Validate all claims
python toreq.py --validate-claims

# Step 2: Check data availability
python scripts/generate_paper.py --validate-only

# Step 3: Generate paper
python scripts/generate_paper.py --paper spectral_normalization

# Step 4: Review output
cat papers/spectral_normalization_paper_generated.md
```

### Expected Output

```
============================================================
CLAIMS VALIDATION REPORT
============================================================

Claim: Spectral Normalization Maintains L < 1
  Status: ✅ VALIDATED
  Details: Documented in INSIGHTS.md

Claim: Competitive Accuracy (97.50%)
  Status: ✅ VALIDATED
  Details: Documented in RESULTS.md (97.50%)

Claim: β-Annealing Causes Instability
  Status: ✅ VALIDATED
  Details: Documented in archive_v1/docs/05-results.md

Claim: Optimal β = 0.22
  Status: ✅ VALIDATED
  Details: β=0.22 documented in INSIGHTS.md

============================================================
✅ All claims validated - ready for publication
============================================================
```

---

## Sign-Off

Before submission, confirm:

- [ ] **Author 1**: Reviewed claims and evidence
- [ ] **Author 2**: Verified code reproducibility
- [ ] **Advisor**: Approved for submission

Date: _______________

Signatures: _______________
