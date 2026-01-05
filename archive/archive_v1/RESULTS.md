# TorEqProp Research Results & Insights

> **A living document of scientific discoveries, empirical data, and validated claims.**

---

## 🏅 Headline Results

### 1. Reinforcement Learning Breakthrough
**Claim**: EqProp optimizes Policy Gradients better than Backprop in sparse/unstable regimes.
*   **Result**: +88% Average Reward on CartPole-v1.
*   **Data**: EqProp (354.1) vs BP (188.6).
*   **Status**: ✅ **VALIDATED** (p < 0.05).

### 2. Gradient Equivalence
**Claim**: EqProp gradients approach BP gradients as $\beta \to 0$.
*   **Result**: Cosine similarity > 0.99 at $\beta=0.001$.
*   **Status**: ✅ **VERIFIED**.

### 3. Stability Threshold
**Claim**: There exists a stable range of $\beta$ for Transformers.
*   **Result**: $\beta \in [0.20, 0.26]$ is stable. $\beta < 0.15$ yields marginal gains, $\beta > 0.30$ diverges.
*   **Discovery**: **Fixed $\beta=0.22$** outperforms annealing strategies.
*   **Status**: ✅ **VALIDATED** (5 seeds).

---

## 📊 Detailed Campaign Results

### Campaign A: Accuracy Push (MNIST)
*Goal: Match Backprop accuracy.*

| Configuration | Epochs | Result | Gap to BP |
|---------------|--------|--------|-----------|
| Baseline ($\beta=0.25$) | 15 | 92.09% | -5.1% |
| Fixed $\beta=0.22$ | 15 | 92.37% | -4.8% |
| **Extended** | **50** | **93.83%** | **-3.4%** |
| BP Baseline | 15 | 97.20% | 0.0% |

**Insight**: Training does not saturate at 15 epochs. EqProp learns slower per-epoch but continues to improve steadily.

### Campaign B: Size Comparison
*Goal: Detect "Punching Above Weight".*

*   **CartPole**: EqProp-Medium (d=128) beats BP-Large (d=256) by **44%**.
*   **MNIST**: BP maintains slight edge across all scales, but gap narrows at small sizes.

---

## 🧠 Experimental Insights

### 1. Beta Dynamics
Theory suggests $\beta \to 0$ recovers exact gradients. However, in practice, extremely small $\beta$ leads to vanishing signals in deep transformers. A "Goldilocks" zone around $\beta=0.22$ provides the best trade-off between bias (large $\beta$) and variance/noise (small $\beta$).

### 2. Symmetry is Overrated?
While theory requires symmetric weights ($W_{fb} = W_{ff}^T$), our experiments show that **asymmetric (separate)** weights often train more stably, provided they are initialized similarly. This challenges the strict requirements of original EqProp theory.

### 3. The "RL Advantage"
We hypothesize EqProp succeeds in RL because the "equilibrium" state naturally filters out high-frequency noise in policy evaluations, acting as a robust regularizer that BP lacks.

---

## 📉 Scaling Laws (Preliminary)

*   **Compute efficiency**: On small tasks (XOR, TinyLM), EqProp is **1.5x - 2x faster** than BP per wall-clock second due to fewer memory operations, despite iterative settling.
*   **Memory**: Verification of O(1) vs O(L) scaling is pending Phase 4.

---

*Last Updated: 2025-12-30*
