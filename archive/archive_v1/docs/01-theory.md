# Theoretical Foundation

> **TorEqProp** proposes training transformers via Equilibrium Propagation on weight-tied (toroidal) architectures, eliminating backpropagation's asymmetric backward pass.

---

## Core Hypothesis

> **H1**: A weight-tied transformer iterated to fixed-point equilibrium can be trained via contrastive Hebbian learning (EqProp) with gradients equivalent to implicit differentiation through the equilibrium.

**Testable Predictions**:

1. $\lim_{\beta \to 0} \frac{\Delta \theta_{\text{EqProp}}}{\beta} = \nabla_\theta \mathcal{L}$ (BP gradient)
2. Convergence to equilibrium occurs in $O(\log(1/\epsilon))$ iterations for well-conditioned systems
3. Training curves (loss, accuracy) match BP baselines within statistical noise

---

## Gradient Equivalence Theorem

**Theorem** (Scellier & Bengio, 2017; adapted): For energy-based dynamics at equilibrium $h^*$, as $\beta \to 0$:

$$\frac{1}{\beta}(h^\beta - h^*) \to -(I - J_f)^{-1} \nabla_h \mathcal{L}$$

and the contrastive update equals:

$$\lim_{\beta \to 0} \frac{\Delta \theta}{\beta} = \nabla_\theta \mathcal{L}\big|_{h=h^*}$$

**Empirical Validation**: Compute both gradients, report cosine similarity and L2 error.

---

## β-Gradient Relationship

Formal expansion (Scellier & Bengio):

$$h^\beta = h^* + \beta \cdot v + O(\beta^2)$$

where $v = -(I - J_f)^{-1} \nabla_h \mathcal{L}$.

Weight gradient:

$$\frac{\partial \mathcal{L}}{\partial \theta} = \lim_{\beta \to 0} \frac{1}{\beta} \left[ \frac{\partial E}{\partial \theta}\bigg|_{h^\beta} - \frac{\partial E}{\partial \theta}\bigg|_{h^*} \right]$$

This recovers the implicit function theorem gradient used in DEQs.

---

## Energy Formulation for Attention

**Open Problem**: Softmax attention lacks closed-form energy. Candidates:

1. **Hopfield energy** (Ramsauer et al.):
   $$E = -\sum_i \log \sum_j \exp(\beta q_i^T k_j) + \text{regularization}$$

2. **Linear attention surrogate**:
   $$\text{Attn}(Q,K,V) = \phi(Q)\phi(K)^T V$$
   admits energy $E = -\frac{1}{2}\|V^T \phi(K)^T \phi(Q)\|^2$

3. **Variational bound**: Treat softmax as approximate inference; derive ELBO-like energy.

**Recommendation**: Start with linear attention for guaranteed results; investigate softmax post-hoc.

---

## Contraction Conditions

For convergence, require $\|J_f\|_2 < 1$. Strategies:

1. **Spectral normalization**: Divide weights by spectral norm
2. **Residual scaling**: $h' = h + \gamma f(h)$ with $\gamma < 1$
3. **Lipschitz FFN**: Use GroupSort or other Lipschitz activations

---

## Key Theoretical Discoveries

### β > 0 Needed for Stability

| Finding | Theory | Practice |
|---------|--------|----------|
| Optimal β | β→0 for exact gradients | β≥0.23 for stability |
| Implication | Larger nudge = stronger learning signal | Theory-practice gap is publishable |

### Non-Symmetric Mode Works

- EqProp's gradient equivalence theorem requires symmetric Jacobians
- **However**: Non-symmetric linear attention trains successfully (92.7% MNIST)
- Symmetric constraints not required for practical training

See [Results](05-results.md) for empirical validation.
