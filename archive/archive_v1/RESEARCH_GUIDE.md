# TorEqProp Research Guide

> **Your comprehensive manual for participating in TorEqProp research.**
> Covers: Quick Start, Infrastructure, Adding Experiments, and Running Campaigns.

---

## 1. Quick Start

### Installation
```bash
git clone https://github.com/yourusername/toreq.git
cd toreq
pip install -r requirements.txt
```

### Smoke Test (Verify Installation)
Run this check to ensure everything works in under 4 minutes:
```bash
# Ultra-fast check (single seed, tiny model)
python -m hyperopt.cli --smoke-test --ultra-fast
```

---

## 2. Core Components

The codebase is modularized into three layers:

*   **Engine** (`toreq/engine/`): Core EqProp algorithms (`eqprop.py`), Update rules (`updates.py`), and Neural layers (`neurl.py`).
*   **Experiments** (`src/`, `train*.py`): Task-specific definitions (Classification, RL, Algorithmic).
*   **Scientific Analysis** (`toreq/hyperopt/`, `toreq/analysis/`): Tools for statistically rigorous comparison and discovery.

---

## 3. Running Scientific Campaigns

We focus on "Killer Experiments" that demonstrate unique properties of EqProp.

### A. Adversarial Robustness (The "Robustness" Claim)
Test if EqProp is naturally more robust to noise than Backprop (BP).

```python
from toreq.analysis.robustness import AdversarialEvaluator
# Use your model/dataloader
evaluator = AdversarialEvaluator(model)
results = evaluator.evaluate(test_loader, method="pgd", epsilon=0.1)
print(f"Attack Success Rate: {results.attack_success_rate}%")
```

### B. Fair Comparison Campaign (The "Efficiency" Claim)
Run a rigorous comparison against a tuned Baseline with **Time-Budget Matching**.

```bash
# Run full hyperopt campaign for MNIST
python -m hyperopt.cli --campaign --task mnist --n-trials 50 --strategy lhs
```
**Features:**
1.  **Latin Hypercube Sampling**: Efficient coverage of hyperparameter space.
2.  **Time-Budget Matching**: Fairness ensures both methods get same wall-clock time.
3.  **Statistical Rigor**: Auto-calculates p-values and Cohen's d effect sizes.

### C. Scaling Laws (The "Scalability" Claim)
Automate the collection of Loss vs Parameters trends.

```python
from toreq.analysis.scaling import ScalingAnalyzer
# ...
analyzer.plot_scaling_laws(results)
```

---

## 4. Experiment Infrastructure Checklist

### Reproducibility Standards
For any result to be trusted, you must ensure:
- [ ] **Seed control**: Log all seeds. Use `seeds=[0, 1, 2]` minimum.
- [ ] **Full Config**: Save the exact YAML/JSON config.
- [ ] **Baseline Tuning**: Ensure the BP baseline is well-tuned (AdamW, Schedule).
- [ ] **Sanity Checks**: Run `test_gradient_equiv.py` if you modify core math.

### Adding New Experiments
The framework uses a registry pattern in `src/experiment_framework.py`.

1.  **Create Class**: Inherit from `Experiment`.
2.  **Register**: `ExperimentRegistry.register_experiment("my_type", MyExperiment)`.
3.  **Configure**: Add to `configs/experiments.yaml`.

---

## 5. Hyperoptimization Engine (`toreq/hyperopt/`)

The engine is the heart of our discovery process.

### CLI Usage
```bash
# Run a specific task
python -m hyperopt.cli --task xor --strategy sobol --n-trials 20

# Run a full campaign (multiple tasks)
python -m hyperopt.cli --campaign

# Generate Analysis Report
python -m hyperopt.cli --report --task mnist
```

### Understanding the Report
*   **Pareto Frontier**: Shows optimal trade-off curves (Accuracy vs Time).
*   **Matched Pairs**: Direct comparison of trials with similar resources.
*   **Statistical Verdict**: "Significantly Better", "Tie", or "Worse".

---

## 6. How to Contribute

### Where to Start?
1.  Check `ROADMAP.md` for open "Phase 2" tasks.
2.  Pick a "Stretch Claim" from `README.md` (e.g., O(1) memory verification).
3.  Run the experiment and document results.

### Debugging
*   **OOM?**: Reduce batch size or `d_model`.
*   **Divergence?**: Reduce `beta` (try 0.1-0.2) or `lr`.
*   **Slow?**: Use `--compile` or `--rapid` flag.

### Documentation
*   Update `RESULTS.md` with any new significant findings.
*   Keep `ROADMAP.md` updated as you complete tasks.

---

## 7. Evidence-Based Validation (The "Undeniable Results" System)

To produce results that are **complete, clear, and undeniable**, use the `ValidationPipeline`.

### How it Works
1.  **`StatisticalVerdict`**: Auto-computes p-values (Mann-Whitney U or Welch's t), Bootstrap 95% Confidence Intervals, and Cohen's d effect size.
2.  **`EvidenceArchiver`**: Stores raw data in JSON format with SHA-256 checksums to prove data integrity.
3.  **`ReportGenerator`**: Creates publication-ready Markdown reports with Executive Summary and detailed tables.

### Usage
```python
from hyperopt.validation import ValidationPipeline

pipeline = ValidationPipeline()

# Prepare your claim data (from hyperopt DB or experiments)
claims = [
    {
        "claim": "EqProp > BP on Adversarial Robustness",
        "eqprop": [0.92, 0.91, 0.93, 0.90, 0.88],  # Raw accuracy under attack
        "baseline": [0.75, 0.78, 0.72, 0.80, 0.76]
    },
    # ... more claims
]

# Run validation
verdicts = pipeline.validate_claims(claims)

# Generate a publication-ready report
pipeline.generate_report(verdicts, title="Adversarial Robustness Study")
```
**Output**: A Markdown file in `reports/` with:
*   **Executive Summary Table**: Claim, Verdict, p-value, Effect Size.
*   **Detailed Breakdowns**: Per-claim statistics.
*   **Methodology Section**: Documents the statistical methods used.
*   **Archived Evidence**: Raw data saved with checksums in `evidence_archive/`.

---

*Happy Discovering!*
