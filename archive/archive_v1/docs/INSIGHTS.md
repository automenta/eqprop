# Experimental Insights & Lessons Learned

**Last Updated**: December 29, 2025

---

## Critical Discoveries

### 1. β-Annealing Instability (Dec 2025)

**Discovery**: β-annealing causes catastrophic training collapse, NOT low β values themselves.

**Evidence**:
- β=0.20 with annealing → collapsed at epoch 14
- β=0.20 fixed (no annealing) → **stable, 91.52% accuracy**

**Root Cause**: Equilibrium dynamics are highly sensitive to parameter transitions
- During annealing, the equilibrium point shifts
- Model cannot adapt quickly enough → loss spikes
- Gradients become unreliable → training collapses

**Lesson**: **Never use β-annealing for EqProp transformers**

**Practical Impact**: 
- Saves hours of wasted experiments
- Prevents mysterious collapses
- Simplifies hyperparameter tuning (one less variable)

---

### 2. Optimal β = 0.22 (Dec 2025)

**Discovery**: β=0.22 achieves best accuracy (92.37%), contradicting both theory and initial intuition.

**Theory says**: β→0 for gradient equivalence
**Previous belief**: β=0.25 was optimal
**Reality**: β=0.22 is the sweet spot

**Why β=0.22?**
1. **Sufficient training signal**: β > 0.20 provides strong enough nudge
2. **Good gradient approximation**: β not too large to distort gradients
3. **Stable equilibrium**: Doesn't cause convergence issues

**Accuracy vs β Curve**:
```
0.20: 91.52% ────┐
0.21: 91.55% ────│ Ramping up
0.22: 92.37% ────┘ PEAK 🏆
0.23: 91.98% ────┐
0.24: 92.04% ────│ Plateau
0.25: 92.12% ────┘
0.26: 91.64% ──── Decline
```

**Lesson**: Optimal β is problem-dependent, test empirically

---

### 3. No Universal Stability Threshold (Dec 2025)

**Discovery**: ALL β ∈ [0.20, 0.26] train stably (contrary to hypothesis).

**Previous hypothesis**: β < 0.23 causes collapse
**Reality**: No collapse observed in any run

**Implications**:
- Wide stable range for β selection
- Safety margin for hyperparameter tuning
- Can experiment with lower β if needed

**Lesson**: Don't extrapolate from annealing experiments to fixed β

---

### 4. Log Parsing is Critical (Dec 2025)

**Discovery**: Incorrect log parsing led to false "all runs failed" conclusion.

**Issue**: Regex expected format didn't match actual log output
- Expected: `"Train Acc: 0.XX, Test Acc: 0.YY"` (single line)
- Actual: Separate lines

**Impact**: Lost 2 hours investigating "errors" that didn't exist

**Lesson**: 
- Verify parsing logic with sample logs BEFORE running experiments
- Add unit tests for log parsers
- Include sanity checks (e.g., "all zeros = probably parsing bug")

---

### 5. Fixed vs Annealed Parameters (Dec 2025)

**Discovery**: Parameter schedules can be more harmful than constant values.

**General principle**: Equilibrium models are sensitive to parameter changes
- Each β value induces a different equilibrium manifold
- Transitions between manifolds can be unstable
- Fixed parameters → stable training dynamics

**Lesson**: Default to fixed hyperparameters unless there's strong evidence for scheduling

**Exceptions where scheduling might work**:
- Very gradual changes (e.g., β: 0.25→0.24 over 50 epochs)
- After initial convergence (epochs 20+)
- With explicit equilibrium re-initialization

---

## Accuracy Optimization Strategies

### What Doesn't Work

1. **β-annealing** ❌ (causes collapse)
2. **Very low β (< 0.20)** ⚠️ (marginal gains, risk not worth it)
3. **Very high β (> 0.26)** ⚠️ (performance degrades)

### What Works

1. **β=0.22 fixed** ✅ (92.37% - current best)
2. **d_model=256** ✅ (better than 128)
3. **Dropout=0.1** ✅ (regularization helps)
4. **15 epochs** ✅ (good baseline)

### What to Try Next

**Priority 1: Extended Training**
- β=0.22, 30-50 epochs
- Hypothesis: May not have fully converged at 15 epochs
- Expected gain: +0.5-1.0%

**Priority 2: Architecture Scaling**
- d_model=512 (2× capacity)
- n_heads=16 (better attention)
- d_ff=2048 (larger FFN)
- Expected gain: +1.0-1.5%

**Priority 3: Learning Rate Schedule**
- Cosine annealing (LR, not β!)
- Warmup for first few epochs
- Expected gain: +0.3-0.5%

**Priority 4: Layer Normalization**
- Add LayerNorm to transformer block
- May improve gradient flow
- Expected gain: +0.2-0.5%

---

## Experimental Design Best Practices

### Before Running Experiments

1. **Test parsing logic** on sample logs
2. **Run small-scale pilot** (1-2 epochs) to verify setup
3. **Document hypothesis** clearly
4. **Estimate compute time** and plan accordingly

### During Experiments

1. **Monitor early epochs** for anomalies
2. **Save checkpoints** at regular intervals
3. **Log comprehensive metrics** (not just accuracy)
4. **Track resource usage** (GPU memory, time per epoch)

### After Experiments

1. **Verify results** before drawing conclusions
2. **Cross-check** with previous experiments
3. **Document surprises** (e.g., β=0.20 didn't collapse)
4. **Update beliefs** based on evidence

---

## Parameter Recommendations

### For Maximum Accuracy (MNIST)

```yaml
# Validated configuration (92.37%)
beta: 0.22           # FIXED (no annealing!)
d_model: 256
n_heads: 8
d_ff: 1024
damping: 0.8
lr: 0.002
dropout: 0.1
epochs: 15           # Minimum; try 30-50 for better results
attention: linear
symmetric: false
compile: true        # ~5% speedup
```

### For Stability

- β ∈ [0.22, 0.25]: Safe range
- damping=0.8: Proven stable
- Avoid parameter schedules initially

### For Speed

- `--compile`: Use torch.compile
- Larger batch size if GPU memory allows
- Reduce max_iters if equilibrium converges quickly

---

## Theory-Practice Gaps

### Gap 1: Optimal β

- **Theory**: β→0 maximizes gradient equivalence
- **Practice**: β=0.22 maximizes accuracy
- **Explanation**: Finite learning rate and discrete optimization matter

### Gap 2: Symmetric Mode

- **Theory**: Symmetric weights required for energy-based formulation
- **Practice**: Non-symmetric works fine (even better)
- **Explanation**: Energy formulation is sufficient but not necessary

### Gap 3: Memory Efficiency

- **Theory**: O(1) memory (constant regardless of iterations)
- **Practice**: 1.06× BP overhead (slightly MORE memory)
- **Explanation**: Implementation may not fully utilize local updates

---

## Open Questions for Future Research

1. **Why exactly β=0.22?** Can we derive this theoretically?
2. **How to achieve true O(1) memory?** Current implementation doesn't show advantage.
3. **Does adaptive compute emerge on harder tasks?** MNIST might be too simple.
4. **Can we combine EqProp with other techniques?** (e.g., knowledge distillation)
5. **How does this scale to larger models?** (d_model=1024+, multiple layers)
6. **Is there a better convergence criterion?** Current tolerance may be suboptimal.

---

## Future Experiments Timeline

### Week 1: Accuracy Push to 94%

- **Day 1**: Extended training (β=0.22, 50 epochs)
- **Day 2**: Architecture scaling (d_model=512)
- **Day 3**: Multi-seed validation (5 seeds)

### Week 2: Understanding & Optimization

- **Day 4**: Learning rate schedule experiments
- **Day 5**: Layer normalization ablation
- **Day 6**: β fine-tuning (β ∈ [0.21, 0.23])

### Week 3: Scaling & Generalization

- **Day 7**: CIFAR-10 implementation
- **Day 8**: Larger model experiments (d=1024)
- **Day 9**: Adaptive compute analysis on harder datasets

---

## Key Takeaways

1. 🎯 **Use β=0.22 fixed** for best results
2. ❌ **Never use β-annealing** (causes instability)
3. ✅ **All β ∈ [0.20, 0.26] are stable** (wide safety margin)
4. 🚀 **92.37% → 94% gap is achievable** with extended training + architecture improvements
5. 📊 **Always verify log parsing** before drawing conclusions
6. 🔬 **Test hypotheses empirically** - theory doesn't always match practice

---

**Document Purpose**: Guide future experimental decisions and avoid repeating mistakes. Update as new insights emerge.
