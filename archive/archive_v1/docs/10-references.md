# References

## Core Theory

| Citation | Key Contribution |
|----------|------------------|
| **Scellier & Bengio (2016)** *"Equilibrium Propagation: Bridging the Gap Between Energy-Based Models and Backpropagation"* [arXiv:1602.05179](https://arxiv.org/abs/1602.05179) | Foundation of EqProp: two-phase contrastive Hebbian learning with β-nudging. Proves gradient equivalence in the limit β→0. Shows STDP-compatible updates. |
| **Farinha et al. (2020)** *"Equilibrium Propagation for Complete Directed Neural Networks"* [arXiv:2006.08798](https://arxiv.org/abs/2006.08798) | Extends EqProp to **arbitrary directed architectures** (not just symmetric/Hopfield). Introduces Lyapunov-based convergence analysis. Adds sparsity-inducing methods for pruning. |
| **Meulemans et al. (2022)** *"Minimizing Control for Credit Assignment with Strong Feedback"* [arXiv:2204.07249](https://arxiv.org/abs/2204.07249) | Deep Feedback Control (DFC): frames learning as **control minimization**. Uses strong feedback (not infinitesimal like EqProp). Learns forward and feedback connections simultaneously with fully local rules. Shows robustness to noise. |

---

## Architectural Foundations

| Citation | Key Contribution |
|----------|------------------|
| **Bai et al. (2019)** *"Deep Equilibrium Models"* NeurIPS | DEQ architecture: implicit differentiation through fixed-point. Demonstrated equilibrium transformers at scale. |
| **Dehghani et al. (2018)** *"Universal Transformers"* ICLR | Weight-tied (looped) transformers with adaptive computation time. |
| **Ramsauer et al. (2021)** *"Hopfield Networks is All You Need"* ICLR | Modern Hopfield networks with transformer-compatible energy. |

---

## Convergence & Stability

| Citation | Key Contribution |
|----------|------------------|
| **Yang et al. (2024)** *"Looped Transformers for In-Context Learning"* | Expressive power analysis of looped architectures; timestep encoding. |
| **Laborieux et al. (2021)** *"Scaling Equilibrium Propagation to Deep ConvNets"* | Practical EqProp at scale with convergence techniques. |
| **Hoover et al. (2023)** *"Energy Transformer"* | Energy-based attention mechanisms; theoretical grounding for equilibrium attention. |

---

## Biological Plausibility

| Citation | Key Contribution |
|----------|------------------|
| **Lillicrap et al. (2020)** *"Backpropagation and the Brain"* Nature Reviews Neuroscience | Survey of biologically plausible alternatives to backprop. |
| **Whittington & Bogacz (2019)** *"Theories of Error Back-Propagation in the Brain"* Trends in Cognitive Sciences | Predictive coding and energy-based learning in neural circuits. |

---

## Differentiating From TorEqProp

| Approach | Relationship to TorEqProp |
|----------|---------------------------|
| **DEQ** | Uses implicit differentiation with BP—not biologically plausible; TorEqProp uses contrastive Hebbian updates |
| **Hopfield Transformers** | Energy is descriptive; TorEqProp's energy is prescriptive (drives dynamics) |
| **Predictive Coding** | Different local update rule; not transformer-native |
| **DFC** | Complementary approach—uses strong feedback; potential future hybrid with TorEqProp |

---

## Additional Resources

### Efficient Attention
- **Performer** (Choromanski et al., 2020): Linear attention via random features
- **Linear Transformers** (Katharopoulos et al., 2020): Linearized attention mechanism

### Neuromorphic Computing
- **Loihi** (Intel): Neuromorphic chip compatible with local learning rules
- **SpiNNaker** (Manchester): Spiking neural network hardware

### Optimization
- **Anderson Acceleration**: Convergence acceleration for fixed-point iteration
- **Spectral Normalization** (Miyato et al., 2018): Weight constraint for stability
