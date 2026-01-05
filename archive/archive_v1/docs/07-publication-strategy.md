# Publication Strategy

## Target Venues

> **Primary Target**: ICML/NeurIPS 2025 submission

---

## Maximum Impact Paper Structures

### Option 1: Unified Story
**"TorEqProp: Training Transformers via Equilibrium Propagation"**
- Gradient equivalence + 95% accuracy + adaptive compute + biological plausibility

### Option 2: Systems Focus  
**"O(1) Memory Training for Transformers via Local Hebbian Learning"**
- Memory profiling + large model demo + neuromorphic potential

### Option 3: Theory Focus
**"On Gradient Equivalence in Equilibrium Attention Mechanisms"**
- β→0 limit + non-symmetric discovery + convergence analysis

### Option 4: Adaptive Compute
**"Implicit Depth: How Equilibrium Transformers Allocate Compute"**
- Iteration dynamics + difficulty correlation + early exit

---

## Success Recognition Matrix

> [!TIP]
> **Novel Publishable Outcomes** — Not all successes look like the original hypothesis.

| Outcome | Success Type | Publication Venue | Narrative |
|---------|--------------|-------------------|-----------|
| **Full hypothesis confirmed** | Primary | NeurIPS/ICML main | "EqProp trains transformers with O(1) memory and BP-equivalent gradients" |
| **Linear attention only** | Partial | NeurIPS/ICML main | "EqProp for efficient linear transformers" — still novel, still O(1) memory |
| **Softmax requires hybrid** | Partial | ICLR/TMLR | "Hybrid BP-EqProp: Local learning for attention, global for softmax" |
| **Convergence analysis only** | Theoretical | COLT/ALT | "On the convergence conditions for equilibrium in looped transformers" |
| **Adaptive compute validated** | Emergent | ICML workshop | "Implicit depth: Equilibrium iterations as learned computation budget" |
| **Negative result** | Scientific | NeurIPS track / TMLR | "On the limitations of contrastive Hebbian learning for attention mechanisms" |

**Key Insight**: Even a negative result is publishable if:
1. The hypothesis was reasonable and well-motivated
2. The experiments were rigorous
3. The failure mode is clearly characterized
4. Implications for future work are articulated

---

## Adaptive Pivot Strategies

### Pivot A: Softmax Attention Fails → Linear Attention Focus

**Trigger**: Gradient mismatch persists with softmax; works with linear attention.

**Action**:
1. Reframe contribution as "TorEqProp for Efficient Transformers"
2. Emphasize that linear attention is an active research area (Performer, Linear Transformers)
3. Drop CIFAR/SST-2, focus on tasks where linear attention is competitive
4. Position as: "Biologically plausible training for the class of efficient transformers"

**Modified Claims**:
- ~~"Train any transformer via EqProp"~~ → "Train linear-attention transformers via EqProp"
- O(1) memory claim remains valid
- Biological plausibility claim remains valid

---

### Pivot B: Training Too Slow → Focus on Memory Advantage

**Trigger**: Wall-clock is 50-100× slower than BP, but accuracy matches.

**Action**:
1. Reframe as "memory-efficient training for resource-constrained settings"
2. Target edge devices, neuromorphic hardware, federated learning
3. Emphasize that this enables training models that **cannot fit in memory with BP**
4. Add experiments showing TorEqProp trains larger d_model than BP on same GPU

**Modified Claims**:
- Add: "TorEqProp enables training 4× larger models on the same hardware"
- De-emphasize wall-clock; emphasize memory-accuracy tradeoff curve

---

### Pivot C: Equilibrium Unstable → Analyze Stability Conditions

**Trigger**: Convergence is fragile, requires very specific hyperparameters.

**Action**:
1. Pivot to theoretical contribution: characterize stability conditions
2. Derive precise conditions on attention mechanism for contraction
3. Propose modified attention that guarantees contraction (novel architecture)
4. Paper becomes: "Stable Equilibrium Transformers: Theory and Design"

**Modified Output**:
- New architecture proposal (e.g., "Contractive Attention")
- Theoretical analysis of Jacobian spectral properties
- Practical guidelines for equilibrium-compatible design

---

### Pivot D: Partial Success → Hybrid Approach

**Trigger**: EqProp works for FFN layers but not attention; or works for early layers but not later ones.

**Action**:
1. Propose "Layerwise Learning Rule Selection"
2. Use EqProp where it works, BP for the rest
3. Still reduces memory (EqProp layers need no activation storage)
4. Frame as: "Toward biologically plausible transformers via hybrid local-global learning"

**Novel Contribution**:
- First systematic study of which layers benefit from local vs. global learning
- Practical hybrid training algorithm
- Analysis of the locality-globality tradeoff in neural network training

---

## Decision Tree

```
                    ┌─────────────────────────────────────┐
                    │  Experiment 1: Gradient Verification │
                    └───────────────────┬─────────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
            Cosine > 0.99       Cosine 0.8-0.99      Cosine < 0.8
            (Full success)      (Partial)            (Failure)
                    │                   │                   │
                    ▼                   ▼                   ▼
            Continue to          Try linear           Debug deeply
            Experiment 2         attention            (2 weeks max)
                    │                   │                   │
                    │                   │           ┌───────┴───────┐
                    │                   │           ▼               ▼
                    │                   │       Fixed?          Not fixed
                    │                   │           │               │
                    │                   ▼           ▼               ▼
                    │           Linear works?   Continue        TERMINATE
                    │           ┌─────┴─────┐                   Negative
                    │           ▼           ▼                   result paper
                    │         Yes          No
                    │           │           │
                    │           ▼           ▼
                    │     Pivot A:      Pivot C:
                    │     Linear        Stability
                    │     focus         analysis
                    ▼
            ┌───────────────────────────────────────┐
            │  Experiment 2: MNIST Training          │
            └───────────────────┬───────────────────┘
                                │
            ┌───────────────────┼───────────────────┐
            ▼                   ▼                   ▼
        Acc > 95%          Acc 85-95%          Acc < 85%
        Speed < 50×        or Speed > 50×      after sweep
            │                   │                   │
            ▼                   ▼                   ▼
        Full success       Pivot B:             Pivot D:
        → Exp 3           Memory focus          Hybrid
            │               or Pivot D          approach
            ▼
    ┌───────────────────────────────────────────────┐
    │  Experiments 3-4: Scaling & Adaptive Compute   │
    └───────────────────────────────────────────────┘
```

---

## Wall-Clock Reality Check

> [!WARNING]
> **Addressing the Elephant in the Room**: Training speed comparison.

| Method | Forward Passes per Update | Estimated Slowdown |
|--------|---------------------------|-------------------|
| Backprop | 1 forward + 1 backward ≈ 2 | 1× (baseline) |
| TorEqProp | 50 free + 50 nudged = 100 | **50×** (pessimistic) |
| TorEqProp (optimized) | 20 free + 20 nudged = 40 | **20×** (optimistic) |
| TorEqProp + Anderson | 10 free + 10 nudged = 20 | **10×** (aggressive) |

**Honest Assessment**: TorEqProp will likely be 10-50× slower than BP per training step. This must be offset by:

1. **Memory advantage**: Train models that don't fit with BP
2. **Parallelization**: Each equilibrium step is embarrassingly parallel
3. **Hardware co-design**: Neuromorphic chips could run equilibrium natively
4. **Inference benefit**: Adaptive compute at test time

**Paper Strategy**: Acknowledge slowdown upfront; position memory as primary advantage.

---

## Publishable Contributions Summary

| Result | Status | Venue |
|--------|--------|-------|
| First transformer via EqProp | ✅ 92.11% | Workshop/TMLR |
| Gradient equivalence | ✅ 0.9972 | Theory track |
| O(1) memory activated | ✅ Ready | Systems track |
| β=0.25 finding | ✅ Discovered | Empirical methods |
| Non-symmetric mode | ✅ Validated | Theoretical |

**Status**: Multiple independent publishable contributions achieved.
