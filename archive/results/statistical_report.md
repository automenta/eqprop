# Statistical Validation Report

**Generated**: 2026-01-02T15:30:37.069201

---
## Digits (8x8) Results

| Model | Mean Acc | Std Dev | Seeds | Status |
|-------|----------|---------|-------|--------|
| BackpropMLP | 97.04% | ±0.35% | 3 | ✅ |
| LoopedMLP (SN) | 94.63% | ±0.73% | 3 | ✅ |
| ModernEqProp (SN) | 94.26% | ±0.35% | 3 | ✅ |
| MSTEP (SN) | 93.89% | ±0.68% | 3 | ✅ |
| EnhancedMSTEP (SN) | 90.28% | ±4.25% | 3 | ⚠️ High variance |

### Significance vs LoopedMLP (SN)

Baseline: **LoopedMLP (SN)** (94.63% ± 0.73%)

| Model | Gap | P-value | Cohen's d | Significance |
|-------|-----|---------|-----------|--------------|
| BackpropMLP | -2.41% | 0.0135 | -4.22 (large) | ✅ Yes |
| ModernEqProp (SN) | 0.37% | 0.5518 | 0.65 (medium) | ⚠️ No |
| MSTEP (SN) | 0.74% | 0.3528 | 1.05 (large) | ⚠️ No |
| EnhancedMSTEP (SN) | 4.35% | 0.2266 | 1.43 (large) | ⚠️ No |

---

## MNIST Results

| Model | Mean Acc | Std Dev | Seeds | Status |
|-------|----------|---------|-------|--------|
| BackpropMLP | 94.91% | ±0.05% | 3 | ✅ |
| LoopedMLP (SN) | 94.22% | ±0.08% | 3 | ✅ |
| ModernEqProp (SN) | 85.88% | ±2.20% | 3 | ⚠️ High variance |
| MSTEP (SN) | 76.98% | ±6.91% | 3 | ⚠️ High variance |
| EnhancedMSTEP (SN) | 83.21% | ±2.76% | 3 | ⚠️ High variance |

### Significance vs LoopedMLP (SN)

Baseline: **LoopedMLP (SN)** (94.22% ± 0.08%)

| Model | Gap | P-value | Cohen's d | Significance |
|-------|-----|---------|-----------|--------------|
| BackpropMLP | -0.69% | 0.0006 | -9.77 (large) | ✅ Yes |
| ModernEqProp (SN) | 8.34% | 0.0059 | 5.35 (large) | ✅ Yes |
| MSTEP (SN) | 17.23% | 0.0243 | 3.53 (large) | ✅ Yes |
| EnhancedMSTEP (SN) | 11.01% | 0.0049 | 5.63 (large) | ✅ Yes |

---

## Fashion-MNIST Results

| Model | Mean Acc | Std Dev | Seeds | Status |
|-------|----------|---------|-------|--------|
| BackpropMLP | 83.25% | ±0.33% | 3 | ✅ |
| LoopedMLP (SN) | 83.32% | ±0.21% | 3 | ✅ |
| ModernEqProp (SN) | 76.89% | ±2.51% | 3 | ⚠️ High variance |
| MSTEP (SN) | 74.82% | ±2.81% | 3 | ⚠️ High variance |
| EnhancedMSTEP (SN) | 60.25% | ±14.16% | 3 | ⚠️ High variance |

### Significance vs LoopedMLP (SN)

Baseline: **LoopedMLP (SN)** (83.32% ± 0.21%)

| Model | Gap | P-value | Cohen's d | Significance |
|-------|-----|---------|-----------|--------------|
| BackpropMLP | 0.07% | 0.8123 | 0.25 (small) | ⚠️ No |
| ModernEqProp (SN) | 6.43% | 0.0226 | 3.61 (large) | ✅ Yes |
| MSTEP (SN) | 8.50% | 0.0130 | 4.26 (large) | ✅ Yes |
| EnhancedMSTEP (SN) | 23.07% | 0.0826 | 2.30 (large) | ⚠️ No |

---

## CartPole-BC Results

| Model | Mean Acc | Std Dev | Seeds | Status |
|-------|----------|---------|-------|--------|
| BackpropMLP | 99.80% | ±0.08% | 3 | ✅ |
| LoopedMLP (SN) | 97.13% | ±1.64% | 3 | ✅ |
| ModernEqProp (SN) | 98.23% | ±1.45% | 3 | ✅ |
| MSTEP (SN) | 93.60% | ±7.61% | 3 | ⚠️ High variance |
| EnhancedMSTEP (SN) | 77.90% | ±21.14% | 3 | ⚠️ High variance |

### Significance vs LoopedMLP (SN)

Baseline: **LoopedMLP (SN)** (97.13% ± 1.64%)

| Model | Gap | P-value | Cohen's d | Significance |
|-------|-----|---------|-----------|--------------|
| BackpropMLP | -2.67% | 0.0837 | -2.29 (large) | ⚠️ No |
| ModernEqProp (SN) | -1.10% | 0.5173 | -0.71 (medium) | ⚠️ No |
| MSTEP (SN) | 3.53% | 0.5557 | 0.64 (medium) | ⚠️ No |
| EnhancedMSTEP (SN) | 19.23% | 0.2689 | 1.28 (large) | ⚠️ No |

---

## Acrobot-BC Results

| Model | Mean Acc | Std Dev | Seeds | Status |
|-------|----------|---------|-------|--------|
| BackpropMLP | 97.97% | ±0.47% | 3 | ✅ |
| LoopedMLP (SN) | 96.83% | ±1.17% | 3 | ✅ |
| ModernEqProp (SN) | 75.97% | ±0.85% | 3 | ✅ |
| MSTEP (SN) | 77.10% | ±1.14% | 3 | ✅ |
| EnhancedMSTEP (SN) | 77.07% | ±0.58% | 3 | ✅ |

### Significance vs LoopedMLP (SN)

Baseline: **LoopedMLP (SN)** (96.83% ± 1.17%)

| Model | Gap | P-value | Cohen's d | Significance |
|-------|-----|---------|-----------|--------------|
| BackpropMLP | -1.13% | 0.2719 | -1.27 (large) | ⚠️ No |
| ModernEqProp (SN) | 20.87% | 0.0000 | 20.44 (large) | ✅ Yes |
| MSTEP (SN) | 19.73% | 0.0001 | 17.08 (large) | ✅ Yes |
| EnhancedMSTEP (SN) | 19.77% | 0.0000 | 21.45 (large) | ✅ Yes |

---

---

## Issues Detected

### ⚠️ Warnings

- EnhancedMSTEP (SN): High variance (±4.25% > 2.0%) - Seeds: ['84.44', '94.44', '91.94']
- ModernEqProp (SN): High variance (±2.20% > 2.0%) - Seeds: ['88.95', '84.80', '83.89']
- MSTEP (SN): High variance (±6.91% > 2.0%) - Seeds: ['83.52', '67.43', '80.00']
- EnhancedMSTEP (SN): High variance (±2.76% > 2.0%) - Seeds: ['80.08', '86.80', '82.74']
- ModernEqProp (SN): High variance (±2.51% > 2.0%) - Seeds: ['74.11', '76.37', '80.20']
- MSTEP (SN): High variance (±2.81% > 2.0%) - Seeds: ['75.95', '77.56', '70.95']
- EnhancedMSTEP (SN): High variance (±14.16% > 2.0%) - Seeds: ['46.71', '79.80', '54.25']
- MSTEP (SN): High variance (±7.61% > 2.0%) - Seeds: ['99.90', '98.00', '82.90']
- EnhancedMSTEP (SN): High variance (±21.14% > 2.0%) - Seeds: ['48.60', '87.40', '97.70']

---

## Publishability Assessment

⚠️ **NEEDS ATTENTION** - Multiple warnings detected