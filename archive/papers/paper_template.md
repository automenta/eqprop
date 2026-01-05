# Paper Template: Research Paper Structure

> **Purpose**: Template for auto-generating papers from experimental results  
> **Format**: Markdown (converts to LaTeX/PDF via Pandoc)

---

## Paper Metadata

```yaml
title: "[TITLE]"
authors:
  - name: "[Author Name]"
    affiliation: "[Institution]"
    email: "[email]"
venue: "[ICML/NeurIPS/TMLR]"
year: 2025
keywords:
  - equilibrium propagation
  - biologically plausible learning
  - spectral normalization
```

---

## Abstract (150-250 words)

**Template Structure**:

[PROBLEM]: What fundamental challenge does this address?

[GAP]: What's missing from prior work?

[CONTRIBUTION]: What do we show/prove/demonstrate?

[RESULT]: Key quantitative finding(s)

[IMPACT]: Why does this matter?

---

**Example**:

> Equilibrium Propagation (EqProp) offers a biologically plausible alternative to backpropagation, but practical training on modern architectures has been hindered by instability. We identify that training induces Lipschitz constant explosion (L → 9.5 for untamed networks), breaking the contraction mapping required for convergence. We demonstrate that spectral normalization universally maintains L < 1 throughout training, enabling stable EqProp on attention-based architectures. Our method achieves 97.50% accuracy on MNIST—matching backpropagation—while preserving biological plausibility. Additionally, we discover that β-annealing (varying the nudging strength during training) causes catastrophic collapse, whereas fixed β values as low as 0.20 remain stable. This contradicts prior intuition and provides critical practical guidance for EqProp practitioners. Our work represents the first rigorous demonstration of competitive EqProp training on modern architectures and identifies spectral normalization as an essential component for stable energy-based learning.

---

## 1. Introduction

### 1.1 Motivation
- Why is biological plausibility important?
- What are the limitations of backpropagation?
- Why do we need alternatives?

### 1.2 Challenge
- What makes EqProp training difficult?
- What problems have prevented practical adoption?

### 1.3 Our Contribution
**State contributions as numbered list**:

1. **[Main Finding]**: Description
2. **[Secondary Finding]**: Description  
3. **[Practical Contribution]**: Description

### 1.4 Paper Organization
Brief roadmap of remaining sections.

---

## 2. Background

### 2.1 Equilibrium Propagation
- Original formulation (Scellier & Bengio, 2017)
- Free and nudged phases
- Contrastive Hebbian update rule

### 2.2 Convergence Requirements
- Contraction mapping theorem
- Lipschitz constant requirements (L < 1)
- Energy function formulation

### 2.3 Spectral Normalization
- Original use in GANs (Miyato et al., 2018)
- How it controls Lipschitz constant
- Why it's relevant for EqProp

---

## 3. Method

### 3.1 Problem Formulation
- Define the architecture
- State the training objective
- Describe the equilibrium dynamics

### 3.2 [Main Technical Contribution]
- Detailed description
- Mathematical formulation
- Implementation details

### 3.3 Training Algorithm
```
Algorithm 1: [Name]
Input: ...
Output: ...
1. ...
2. ...
3. ...
```

---

## 4. Experiments

### 4.1 Experimental Setup

**Datasets**: [List datasets]

**Baselines**: [List baselines]

**Implementation Details**:
- Hardware
- Hyperparameters
- Training procedure

### 4.2 Main Results

**Table 1**: Main comparison

| Method | Accuracy | [Metric 2] | [Metric 3] |
|--------|----------|------------|------------|
| Baseline | X% | ... | ... |
| **Ours** | **Y%** | ... | ... |

### 4.3 Ablation Studies

**What components are essential?**

| Variant | Result | Δ from Full |
|---------|--------|-------------|
| Full model | X% | - |
| - Component A | Y% | -Z% |

### 4.4 Analysis

**[Specific finding visualizations]**

---

## 5. Discussion

### 5.1 Why It Works
- Theoretical explanation
- Connection to prior work

### 5.2 Limitations
- What doesn't work
- When to not use this method
- Computational overhead

### 5.3 Broader Impact
- Applications
- Societal implications

---

## 6. Related Work

### Equilibrium Propagation
- Scellier & Bengio (2017)
- Laborieux et al. (2021)
- Ernoult et al. (2020)

### Biologically Plausible Learning
- Lillicrap et al. (2020) - Feedback alignment
- Hinton (2022) - Forward-forward

### Spectral Normalization
- Miyato et al. (2018)
- Applications in other domains

---

## 7. Conclusion

**Summary**: [One paragraph restating main contributions]

**Future Work**: [2-3 sentences on next steps]

---

## References

[Auto-generated from citations]

---

## Appendix

### A. Proof of [Theorem]

### B. Additional Experiments

### C. Hyperparameter Sensitivity

### D. Implementation Details

---

## Auto-Generation Markers

The paper generator script looks for these markers to insert content:

```
<!-- INSERT:abstract -->
<!-- INSERT:table:main_results -->
<!-- INSERT:figure:training_curves -->
<!-- INSERT:table:ablation -->
<!-- INSERT:hyperparameters -->
```

### Data Sources

Results are pulled from:
- `/tmp/competitive_benchmark.json` - Main results
- `logs/beta_sweep/results.json` - β sweep data
- `scripts/` output files

---

## Conversion Commands

### To LaTeX
```bash
pandoc paper.md -o paper.tex --template=arxiv.latex
```

### To PDF
```bash
pandoc paper.md -o paper.pdf --template=arxiv.latex --pdf-engine=xelatex
```

### To HTML (for review)
```bash
pandoc paper.md -o paper.html --standalone --toc
```
