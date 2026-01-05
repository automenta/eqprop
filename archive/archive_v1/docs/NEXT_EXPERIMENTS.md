# Next Experiments Roadmap

**Goal**: Push accuracy from 92.37% → 94%+

---

## Priority 1: Extended Training with β=0.22 (HIGH)

**Hypothesis**: Current 15-epoch training may not have fully converged.

**Experiment**:
```bash
./run_extended_training.sh  # 50 epochs with β=0.22
```

**Configuration**:
- β=0.22 (fixed, no annealing)
- d_model=256, n_heads=8, d_ff=1024
- 50 epochs (vs 15 baseline)
- All other params same as β sweep

**Expected outcome**: 92.37% → 93.5-94%
**Estimated time**: ~3.5 hours
**Risk**: Low (proven stable)

---

## Priority 2: Architecture Scaling (HIGH)

**Hypothesis**: Larger model capacity will improve accuracy.

### Experiment 2a: d_model=512
```bash
python train.py --d-model 512 --n-heads 16 --d-ff 2048 \
    --beta 0.22 --damping 0.8 --lr 0.002 --epochs 30 \
    --dropout 0.1 --compile
```

**Expected**: +1.0-1.5% improvement
**Time**: ~2 hours (30 epochs)

### Experiment 2b: Deeper FFN
```bash
python train.py --d-model 256 --n-heads 8 --d-ff 2048 \
    --beta 0.22 --damping 0.8 --lr 0.002 --epochs 30 \
    --dropout 0.1 --compile
```

**Expected**: +0.5-1.0% improvement
**Time**: ~2 hours

---

## Priority 3: Multi-Seed Validation (MEDIUM)

**Purpose**: Establish statistical significance of β=0.22 optimality.

**Experiment**:
```bash
for seed in 1 2 3 4 5; do
    python train.py --d-model 256 --beta 0.22 --damping 0.8 \
        --lr 0.002 --epochs 30 --dropout 0.1 --seed $seed \
        --compile 2>&1 | tee logs/multiseed/beta022_seed${seed}.log
done
```

**Analysis**: Calculate mean ± std dev
**Expected**: Mean ~92.4%, std < 0.5%
**Time**: ~5 hours (5 runs × 30 epochs)

---

## Priority 4: Learning Rate Tuning (MEDIUM)

**Hypothesis**: Current lr=0.002 may not be optimal for β=0.22.

**Experiment**: Test lr ∈ {0.001, 0.0015, 0.002, 0.0025, 0.003}

```bash
for lr in 0.001 0.0015 0.002 0.0025 0.003; do
    python train.py --d-model 256 --beta 0.22 --lr $lr \
        --epochs 20 --compile 2>&1 | tee logs/lr_sweep/lr_${lr}.log
done
```

**Expected**: May find better lr than 0.002
**Time**: ~3 hours

---

## Priority 5: Fine-Grained β Refinement (LOW)

**Hypothesis**: β∈[0.21, 0.23] might contain even better value.

**Experiment**: Test β ∈ {0.21, 0.215, 0.22, 0.225, 0.23}

```bash
for beta in 0.21 0.215 0.22 0.225 0.23; do
    python train.py --d-model 256 --beta $beta --epochs 20 \
        --compile 2>&1 | tee logs/beta_refine/beta_${beta}.log
done
```

**Expected**: Marginal gains (< 0.2%)
**Time**: ~2.5 hours
**Priority**: LOW (diminishing returns)

---

## Priority 6: Layer Normalization Ablation (MEDIUM)

**Hypothesis**: LayerNorm may improve gradient flow and stability.

**Implementation needed**:
1. Add LayerNorm to transformer block
2. Test with/without LayerNorm

**Expected**: +0.2-0.5% improvement
**Effort**: 1 hour implementation + 2 hours experiments

---

## Priority 7: Dropout Rate Tuning (LOW)

**Current**: dropout=0.1
**Test**: {0.0, 0.05, 0.1, 0.15, 0.2}

**Expected**: Minor improvements
**Priority**: LOW (current value likely near-optimal)

---

## Recommended Execution Order

### Week 1: Accuracy Push
1. **Day 1**: Extended training (50 epochs, β=0.22) → Quick win
2. **Day 2**: Multi-seed validation → Statistical rigor
3. **Day 3**: Architecture scaling (d=512) → Capacity boost

### Week 2: Fine-Tuning
4. **Day 4**: Learning rate sweep → Optimization
5. **Day 5**: Layer normalization → Architecture improvement
6. **Day 6**: Analysis and documentation → Prepare for publication

---

## Success Criteria

**Minimum success**: Reach 93.5% (validated across seeds)
**Target success**: Reach 94.0%
**Stretch goal**: Reach 94.5%+

---

## Backup Plans

If 94% proves difficult:

1. **Lower target**: 93.5%+ is still publishable (92.37% → 93.5% = +1.13%)
2. **Emphasize novelty**: β-annealing discovery is independently valuable
3. **Scaling story**: Focus on CIFAR-10 or other datasets
4. **Theory contribution**: Characterization of optimal β for equilibrium models

---

## Resource Estimates

**GPU time needed**:
- Extended training: 3.5 hours
- Multi-seed: 5 hours
- Architecture experiments: 4 hours
- Learning rate sweep: 3 hours
- **Total**: ~15-20 hours GPU time

**Timeline**: 3-5 days with single GPU

---

## Key Insights to Apply

1. ✅ **Use β=0.22 fixed** (validated optimal)
2. ❌ **Never use β-annealing** (proven unstable)
3. ✅ **Extended training likely helps** (15 epochs may be too short)
4. ✅ **Larger models should work** (d=512 worth trying)
5. ✅ **Wide stable range** (safe to experiment with β∈[0.20-0.26])

---

**Last updated**: December 29, 2025
**Next action**: Run extended training with β=0.22
