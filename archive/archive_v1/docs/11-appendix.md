# Appendix: Mathematical Details & Contingency Framework

---

## A1: Energy Formulation for Attention

**Open Problem**: Softmax attention lacks closed-form energy. Candidates:

1. **Hopfield energy** (Ramsauer et al.):
   $$E = -\sum_i \log \sum_j \exp(\beta q_i^T k_j) + \text{regularization}$$

2. **Linear attention surrogate**:
   $$\text{Attn}(Q,K,V) = \phi(Q)\phi(K)^T V$$
   admits energy $E = -\frac{1}{2}\|V^T \phi(K)^T \phi(Q)\|^2$

3. **Variational bound**: Treat softmax as approximate inference; derive ELBO-like energy.

**Recommendation**: Start with linear attention for guaranteed results; investigate softmax post-hoc.

---

## A2: Contraction Conditions

For convergence, require $\|J_f\|_2 < 1$. Strategies:

1. **Spectral normalization**: Divide weights by spectral norm
2. **Residual scaling**: $h' = h + \gamma f(h)$ with $\gamma < 1$
3. **Lipschitz FFN**: Use GroupSort or other Lipschitz activations

---

## A3: β-Gradient Relationship

Formal expansion (Scellier & Bengio):

$$h^\beta = h^* + \beta \cdot v + O(\beta^2)$$

where $v = -(I - J_f)^{-1} \nabla_h \mathcal{L}$.

Weight gradient:

$$\frac{\partial \mathcal{L}}{\partial \theta} = \lim_{\beta \to 0} \frac{1}{\beta} \left[ \frac{\partial E}{\partial \theta}\bigg|_{h^\beta} - \frac{\partial E}{\partial \theta}\bigg|_{h^*} \right]$$

This recovers the implicit function theorem gradient used in DEQs.

---

## Theoretical Explorations

### Energy Landscape Analysis
- Visualize loss surface at equilibrium
- Characterize basins of attraction
- Compare to DEQ (Deep Equilibrium Models)

### Convergence Conditions
- When does equilibrium exist and is unique?
- Spectral properties of Jacobian
- Stability guarantees for attention

### Implicit Differentiation Connection
- Relate EqProp to phantom gradients (DEQ)
- Unify local and global gradient methods
- Theoretical bounds on approximation

---

## Open Questions (Research Directions)

1. Why does higher β work better in practice?
2. Can we derive optimal β theoretically?
3. What's the relationship between iterations and sample difficulty?
4. Can EqProp discover algorithmic structure?
5. How does equilibrium attention differ from standard attention in function space?
6. Is there a principled way to set max_iters adaptively?
7. What architectural modifications guarantee convergence?

---

## Risk Analysis

### High-Risk Issues

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Non-convergence | Medium | 🔴 Fatal | Spectral norm regularization; constrained initialization |
| Gradient mismatch | Low | 🔴 Fatal | Validate β→0 limit analytically; compare to DEQ |
| Slow training | High | 🟡 Major | Anderson acceleration; learned initialization |

### Contingency Plans

**If convergence fails**:
1. Switch to linear attention (guaranteed contraction)
2. Add explicit Jacobian penalty: $\mathcal{L} += \lambda \|\|J_f\|\|_2$
3. Use DEQ-style phantom gradient as fallback

**If gradients don't match BP**:
1. Verify implementation against reference EqProp code
2. Check equilibrium is truly reached (tighten ε)
3. May indicate attention breaks energy assumptions → investigate energy reformulation

**If too slow**:
1. Reduce max_iters, accept approximate equilibrium
2. Parallel batch relaxation
3. Early exit with residual as uncertainty measure

---

## Failure State Recognition

> [!CAUTION]
> **Complete Failure Criteria** — If ANY of these conditions persist after mitigation attempts, terminate the research direction.

| Failure State | Detection Criteria | Mitigation Attempted | Terminal? |
|---------------|---------------------|----------------------|-----------|
| **Equilibrium Non-Convergence** | >50% of samples fail to converge within 200 iterations | Spectral norm reg, linear attention, reduced α | ✅ Yes |
| **Gradient Mismatch** | Cosine similarity <0.8 at β=0.001 after debugging | Verified equilibrium precision, checked autodiff | ✅ Yes |
| **Catastrophic Slowdown** | Training >500× slower than BP with no accuracy benefit | Anderson accel, early exit, reduced precision | ✅ Yes |
| **Accuracy Collapse** | MNIST accuracy <70% after full hyperparameter sweep | Architecture changes, initialization schemes | ✅ Yes |

**Decision Protocol**:
```
IF gradient_cosine_sim < 0.8 AND linear_attention_tested AND equilibrium_verified:
    → TERMINATE: Publish negative result, document failure mode
    
IF mnist_accuracy < 80% AND hyperparameter_sweep_complete:
    → PIVOT: Investigate hybrid BP+EqProp (use EqProp for specific layers only)
    
IF wall_clock > 200x_BP AND no_accuracy_advantage:
    → TERMINATE: The approach is not practically viable
```
