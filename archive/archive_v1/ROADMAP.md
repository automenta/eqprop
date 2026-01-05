# TorEqProp Research Roadmap

> **The Single Source of Truth for Project Direction.**
> Combines completed milestones with the Master Research Plan for Phase 2+.

---

## 🏆 Goals & North Star

**Objective**: Prove that Equilibrium Propagation (EqProp) offers qualitative advantages over Backpropagation (BP) in specific regimes, enabling a new class of efficient, bio-plausible learning algorithms.

**North Star Metrics**:
1.  **Accuracy**: Match BP on MNIST/CIFAR (Currently: 93.83% vs 97% BP).
2.  **Robustness**: Exceed BP in Adversarial Defense (FGSM/PGD).
3.  **Efficiency**: Outperform BP in RL sample efficiency (Verified: +88% on CartPole).
4.  **Scalability**: Demonstrate predictable Power Law scaling.

---

## 🚦 Status Summary

| Phase | Focus | Status | Key Result |
|-------|-------|--------|------------|
| **1** | **Foundations & Sizes** | ✅ **COMPLETE** | EqProp-medium beats BP-large on CartPole (+44%). |
| **2** | **Scientific Discovery** | 🚧 **IN PROGRESS** | Robustness & Dynamics tools implemented. |
| **3** | **Accuracy Push** | ⏸️ **PAUSED** | 93.8% achieved. Scaling next. |
| **4** | **O(1) Memory** | ⏳ **PENDING** | Requires large-scale profiling. |

---

## 🗺️ The Plan

### Phase 2: Scientific Discovery (Current Focus) 🧪
*Address academic skepticism with "Killer Experiments".*

- [x] **Modularize Engine**: Hyperopt package refactored.
- [x] **Adversarial Robustness Tools**: `AdversarialEvaluator` implemented.
- [x] **Dynamical Analysis Tools**: `DynamicsAnalyzer` implemented.
- [x] **Scaling Automation**: `ScalingAnalyzer` implemented.
- [ ] **Run Robustness Campaign**: Compare FGSM/PGD resistance (EqProp vs BP).
- [ ] **Run Dynamics Campaign**: Measure Lyapunov exponents at edge of chaos.

### Phase 3: The Accuracy Push (Target: 95%+) 📈
*Close the gap with Backprop on standard benchmarks.*

- [x] **Validation**: 93.83% verified (50 epochs, $\beta=0.22$).
- [ ] **Architecture Scaling**: Run `d_model=512`.
- [ ] **Extended Training**: Run 100+ epochs.
- [ ] **Advanced Optimizers**: Tune AdamW/Schedule for EqProp specifically.

### Phase 4: O(1) Memory Verification 🧠
*Prove the "Infinite Depth" capability.*

- [ ] **Profile Small**: Verify constant memory on d=256 (Done: 1.06x ratio).
- [ ] **Profile Large**: Verify constant memory on d=2048.
- [ ] **Local Updates**: Ensure `LocalHebbianUpdate` is fully strictly local.

---

## 📝 Detailed Task Queue

### Immediate Next Steps
1.  Run **Adversarial Robustness Campaign** on MNIST/Fashion.
    *   Hypothesis: EqProp's energy relaxation denies gradients to attackers.
2.  Run **Scaling Laws Sweep**.
    *   Hypothesis: EqProp follows clean power laws, potentially with better coefficients in data-limited regimes.

### Backlog
- [ ] Implement "Failure Mode Analysis" (detect oscillatory regimes automatically).
- [ ] Explore "Inference-as-Optimization" (Adaptive Compute).
- [ ] Publish "clean" release of `toreq` package on PyPI.

---

## 📚 Reference: Validated Results

### Reinforcement Learning (CartPole-v1)
*   **EqProp**: 354.1 avg reward (Solved)
*   **BP**: 188.6 avg reward (Failed)
*   **Delta**: +88% improvement

### Classification (MNIST)
*   **EqProp**: 93.83% (Top-1)
*   **BP**: 97.20% (Top-1)
*   **Gap**: -3.37% (Closing fast)

---

*Last Updated: 2025-12-30*
