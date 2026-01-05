# Experimental Plan

## Experiment 1: Gradient Verification (Week 1-2)

**Objective**: Prove EqProp gradients match BP gradients.

| Component | Specification |
|-----------|---------------|
| Model | 1-block looped transformer, d=64, heads=4 |
| Data | MNIST (28×28 flattened to sequence) |
| Metric | Cosine sim(∇_EqProp, ∇_BP), L2 error |
| β values | [0.5, 0.1, 0.01, 0.001] |
| Success | Cosine sim > 0.99 at β=0.001 |

**Protocol**:
1. Forward pass to equilibrium (max 50 iters)
2. Compute EqProp gradient via contrastive activations
3. Compute BP gradient via torch.autograd on equilibrium
4. Compare across 100 random batches

---

## Experiment 2: Training Dynamics (Week 2-4)

**Objective**: Train to convergence, compare learning curves.

| Component | Specification |
|-----------|---------------|
| Model | 1-block looped transformer, d=128, heads=4 |
| Data | MNIST train/test split |
| Baseline | Same architecture trained with BP |
| Metrics | Train loss, test accuracy, iterations/sample |
| Success | ≥95% test accuracy, within 2% of BP baseline |

**Ablations**:
- β ∈ {0.01, 0.05, 0.1, 0.2}
- Damping α ∈ {0.5, 0.7, 0.9, 1.0}
- Solver: fixed-point vs. Anderson acceleration
- **Toroid depth L ∈ {1, 2, 3, 4}** — critical architecture search
- Block weight-sharing: independent vs. tied pairs

---

## Experiment 2.5: Architecture Search (Week 3-4)

**Objective**: Find optimal toroid depth and configuration.

| Configuration | Blocks | d_model | Expected Trade-off |
|--------------|--------|---------|-------------------|
| Shallow-wide | 1 | 256 | Fast convergence, limited depth |
| Medium | 2 | 128 | Balanced |
| Deep-narrow | 4 | 64 | High expressiveness, slow convergence |
| Tied-pairs | 4 (2 unique) | 128 | Regularized, efficient |

**Metrics**:
- Accuracy vs. toroid depth L
- Iterations to convergence vs. L
- Gradient quality (cosine sim) vs. L — does depth degrade EqProp?

**Key Question**: Does adding layers help more than adding iterations at fixed L=1?

---

## Experiment 3: Scaling (Week 4-6)

**Objective**: Validate on harder tasks with best architecture from Exp 2.5.

| Task | Model Size | Target |
|------|------------|--------|
| CIFAR-10 | Best L from Exp 2.5, d=256 | ≥70% accuracy |
| CIFAR-10 | L+1 blocks (test scaling) | ≥75% accuracy |
| Text classification (SST-2) | d=256, vocab=10k | ≥80% accuracy |

**Scaling metrics**:
- Iterations to convergence vs. model dimension
- Iterations to convergence vs. toroid depth L
- Wall-clock time vs. BP (same hardware)
- Peak memory vs. BP

---

## Experiment 3.5: Algorithmic Reasoning (Week 5-6)

**Objective**: Test equilibrium's benefit for iterative reasoning tasks.

> [!NOTE]
> **Hypothesis**: Equilibrium models should excel at tasks requiring variable computation depth — the model can "think longer" on hard instances.

| Task | Description | Why Equilibrium Helps | Target |
|------|-------------|----------------------|--------|
| **Parity** | XOR of N bits | Requires N sequential ops | 100% (N≤20) |
| **Addition** | Add two N-digit numbers | Carry propagation is iterative | 95% (N≤10) |
| **Copying** | Repeat input sequence | Tests memory capacity | 100% |
| **Sorting** | Sort N numbers | Comparison chains | 90% (N≤8) |

**Protocol**:
1. Train on fixed sequence length, test on variable lengths
2. Measure **iterations vs. problem difficulty** (e.g., # of 1s for parity)
3. Compare to fixed-depth transformer with matched parameters

**Key Metrics**:
- Does iteration count correlate with problem complexity?
- Does the equilibrium model generalize to longer sequences?
- Can we visualize "reasoning steps" via intermediate h_t states?

**Success Criterion**: On at least one task, TorEqProp shows adaptive compute that correlates with difficulty AND outperforms matched fixed-depth baseline.

---

## Experiment 4: Adaptive Compute (Week 6-8)

**Objective**: Demonstrate variable-depth computation.

**Protocol**:
1. Train with fixed max_iters=50
2. At test time, measure iterations to ε-convergence per sample
3. Correlate iteration count with sample difficulty (margin, uncertainty)
4. Compare to DEQ baseline (same equilibrium architecture, BP-trained)

**Hypothesis**: Hard samples require more iterations; this correlates with model uncertainty.

---

## Success Criteria

### Minimum Viable Publication (MVP)

| Criterion | Threshold | Priority |
|-----------|-----------|----------|
| Gradient equivalence demonstrated | Cosine sim > 0.99 | 🔴 Critical |
| MNIST convergence | ≥95% accuracy | 🔴 Critical |
| Training completes | <24h on single GPU | 🟡 High |
| Memory advantage shown | <50% of BP peak memory | 🟡 High |

### Stretch Goals

| Goal | Threshold | Priority |
|------|-----------|----------|
| CIFAR-10 competitive | Within 5% of BP baseline | 🟢 Medium |
| Text classification | ≥75% SST-2 accuracy | 🟢 Medium |
| Neuromorphic simulation | Run on Loihi/SpiNNaker | 🔵 Low |

---

## Experimental Priorities (Ranked)

### Tier 1: Publication-Ready
1. **95%+ accuracy** — validates approach works
2. **Gradient equivalence plot** — theoretical justification
3. **Adaptive compute analysis** — novel contribution
4. **5-seed validation** — statistical rigor

### Tier 2: Strengthens Paper
5. **O(1) memory demo** — hardware implications
6. **CIFAR-10 scaling** — generalization
7. **DEQ comparison** — positions in literature
8. **Convergence analysis** — iterations/sample

### Tier 3: Stretch Goals
9. **SST-2 text** — modality transfer
10. **Algorithmic reasoning** — structure discovery
11. **Neuromorphic simulation** — hardware direction
12. **Equilibrium VAE** — generative extension
