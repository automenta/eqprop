"""
Evidence-Based Validation Pipeline.

This module provides tools to produce complete, clear, undeniable, 
evidence-based results for the TorEqProp project.

Components:
    - ValidationPipeline: Runs statistical tests and produces verdicts.
    - EvidenceArchiver: Stores raw data with checksums for reproducibility.
    - ReportGenerator: Creates publication-ready markdown/HTML reports.
"""

import json
import hashlib
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
import scipy.stats as stats

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class StatisticalVerdict:
    """A rigorous statistical conclusion."""
    claim: str                      # The claim being tested (e.g., "EqProp > BP on Robustness")
    eqprop_data: List[float]
    baseline_data: List[float]
    
    # Core Statistics
    eqprop_mean: float = 0.0
    eqprop_std: float = 0.0
    eqprop_n: int = 0
    baseline_mean: float = 0.0
    baseline_std: float = 0.0
    baseline_n: int = 0
    
    # Significance
    p_value: float = 1.0
    test_type: str = ""             # e.g., "Mann-Whitney U", "Welch's t-test"
    ci_lower: float = 0.0           # 95% CI on effect size
    ci_upper: float = 0.0
    
    # Effect Size
    cohens_d: float = 0.0
    effect_size_category: str = ""  # "negligible", "small", "medium", "large"
    
    # Verdict
    is_significant: bool = False
    verdict: str = ""               # "VALIDATED", "REJECTED", "INCONCLUSIVE"
    
    def __post_init__(self):
        self._compute()
    
    def _compute(self):
        """Compute all statistics from raw data."""
        eq = np.array(self.eqprop_data)
        bl = np.array(self.baseline_data)
        
        self.eqprop_n = len(eq)
        self.baseline_n = len(bl)
        self.eqprop_mean = float(np.mean(eq))
        self.eqprop_std = float(np.std(eq, ddof=1)) if len(eq) > 1 else 0.0
        self.baseline_mean = float(np.mean(bl))
        self.baseline_std = float(np.std(bl, ddof=1)) if len(bl) > 1 else 0.0
        
        # Choose test based on normality and sample size
        if self.eqprop_n >= 30 and self.baseline_n >= 30:
            # Large sample: Welch's t-test
            _, self.p_value = stats.ttest_ind(eq, bl, equal_var=False)
            self.test_type = "Welch's t-test"
        else:
            # Small sample: Mann-Whitney U (non-parametric)
            try:
                _, self.p_value = stats.mannwhitneyu(eq, bl, alternative='two-sided')
                self.test_type = "Mann-Whitney U"
            except ValueError:
                self.p_value = 1.0
                self.test_type = "N/A (Insufficient Data)"
        
        # Cohen's d effect size
        pooled_std = np.sqrt(((self.eqprop_n - 1) * self.eqprop_std**2 + 
                              (self.baseline_n - 1) * self.baseline_std**2) /
                             (self.eqprop_n + self.baseline_n - 2))
        if pooled_std > 0:
            self.cohens_d = (self.eqprop_mean - self.baseline_mean) / pooled_std
        else:
            self.cohens_d = 0.0
        
        # Categorize effect size
        d_abs = abs(self.cohens_d)
        if d_abs < 0.2:
            self.effect_size_category = "negligible"
        elif d_abs < 0.5:
            self.effect_size_category = "small"
        elif d_abs < 0.8:
            self.effect_size_category = "medium"
        else:
            self.effect_size_category = "large"
        
        # Bootstrap 95% Confidence Interval on the difference of means
        self.ci_lower, self.ci_upper = self._bootstrap_ci(eq, bl)
        
        # Determine significance and verdict
        alpha = 0.05
        self.is_significant = self.p_value < alpha
        
        if self.is_significant:
            if self.cohens_d > 0:
                self.verdict = "VALIDATED (EqProp > Baseline)"
            else:
                self.verdict = "REJECTED (Baseline > EqProp)"
        else:
            if d_abs < 0.2:
                self.verdict = "TIE (No Meaningful Difference)"
            else:
                self.verdict = "INCONCLUSIVE (Effect observed, but not statistically significant)"

    def _bootstrap_ci(self, eq: np.ndarray, bl: np.ndarray, n_bootstrap: int = 10000, alpha: float = 0.05) -> Tuple[float, float]:
        """Compute bootstrap confidence interval on difference of means."""
        diffs = []
        for _ in range(n_bootstrap):
            eq_sample = np.random.choice(eq, size=len(eq), replace=True)
            bl_sample = np.random.choice(bl, size=len(bl), replace=True)
            diffs.append(np.mean(eq_sample) - np.mean(bl_sample))
        
        lower = np.percentile(diffs, 100 * alpha / 2)
        upper = np.percentile(diffs, 100 * (1 - alpha / 2))
        return float(lower), float(upper)

    def to_markdown(self) -> str:
        """Generate a markdown summary of the verdict."""
        return f"""
### Claim: {self.claim}

| Metric | EqProp | Baseline |
|--------|--------|----------|
| Mean | {self.eqprop_mean:.4f} | {self.baseline_mean:.4f} |
| Std Dev | {self.eqprop_std:.4f} | {self.baseline_std:.4f} |
| N | {self.eqprop_n} | {self.baseline_n} |

**Statistical Test**: {self.test_type}
**p-value**: {self.p_value:.4f} ({"Significant" if self.is_significant else "Not Significant"} at α=0.05)

**Effect Size (Cohen's d)**: {self.cohens_d:.2f} ({self.effect_size_category})
**95% CI on Difference**: [{self.ci_lower:.4f}, {self.ci_upper:.4f}]

---

## 🏆 VERDICT: **{self.verdict}**
"""


# =============================================================================
# EVIDENCE ARCHIVER
# =============================================================================

class EvidenceArchiver:
    """Archives raw experimental data with checksums for reproducibility."""
    
    def __init__(self, archive_dir: str = "evidence_archive"):
        self.archive_dir = Path(archive_dir)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.archive_dir / "manifest.json"
        self.manifest = self._load_manifest()
    
    def _load_manifest(self) -> Dict:
        if self.manifest_path.exists():
            with open(self.manifest_path, 'r') as f:
                return json.load(f)
        return {"entries": []}
    
    def _save_manifest(self):
        with open(self.manifest_path, 'w') as f:
            json.dump(self.manifest, f, indent=2)
    
    def archive(self, claim_id: str, data: Dict[str, Any], metadata: Optional[Dict] = None) -> str:
        """
        Archive data associated with a claim.
        Returns: Path to the archived JSON file.
        """
        timestamp = datetime.now().isoformat()
        filename = f"{claim_id}_{timestamp.replace(':', '-')}.json"
        filepath = self.archive_dir / filename
        
        content = {
            "claim_id": claim_id,
            "timestamp": timestamp,
            "data": data,
            "metadata": metadata or {},
        }
        
        json_str = json.dumps(content, indent=2)
        checksum = hashlib.sha256(json_str.encode()).hexdigest()
        content["checksum"] = checksum
        
        with open(filepath, 'w') as f:
            json.dump(content, f, indent=2)
        
        self.manifest["entries"].append({
            "claim_id": claim_id,
            "filename": filename,
            "checksum": checksum,
            "timestamp": timestamp,
        })
        self._save_manifest()
        
        return str(filepath)
    
    def verify(self, filepath: str) -> bool:
        """Verify that an archived file has not been tampered with."""
        with open(filepath, 'r') as f:
            content = json.load(f)
        
        stored_checksum = content.pop("checksum", None)
        json_str = json.dumps(content, indent=2)
        computed_checksum = hashlib.sha256(json_str.encode()).hexdigest()
        
        return stored_checksum == computed_checksum


# =============================================================================
# REPORT GENERATOR
# =============================================================================

class ReportGenerator:
    """Generates publication-ready reports in Markdown."""
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate(self, verdicts: List[StatisticalVerdict], title: str = "Experimental Report") -> str:
        """Generate a full report from a list of verdicts."""
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        report = f"""# {title}

> **Generated**: {timestamp}
> **Total Claims**: {len(verdicts)}

---

## Executive Summary

| Claim | Verdict | p-value | Effect Size |
|-------|---------|---------|-------------|
"""
        for v in verdicts:
            status_emoji = "✅" if "VALIDATED" in v.verdict else ("❌" if "REJECTED" in v.verdict else "❔")
            report += f"| {v.claim[:40]}... | {status_emoji} {v.verdict.split('(')[0].strip()} | {v.p_value:.4f} | {v.cohens_d:.2f} ({v.effect_size_category}) |\n"
        
        report += "\n---\n\n## Detailed Results\n\n"
        
        for v in verdicts:
            report += v.to_markdown()
            report += "\n---\n\n"
        
        report += """
## Methodology

All results were validated using:
1.  **Non-parametric tests** (Mann-Whitney U) for small samples, Welch's t-test for large samples.
2.  **Bootstrap Confidence Intervals** (10,000 resamples) for effect size estimation.
3.  **Cohen's d** for standardized effect size, categorized by standard thresholds.
4.  **SHA-256 Checksums** on raw data via `EvidenceArchiver` for integrity.

---

*Report generated by TorEqProp Validation Pipeline.*
"""
        
        filepath = self.output_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(filepath, 'w') as f:
            f.write(report)
        
        print(f"📄 Report saved to: {filepath}")
        return str(filepath)


# =============================================================================
# VALIDATION PIPELINE (Main Entry Point)
# =============================================================================

class ValidationPipeline:
    """
    Orchestrates the full evidence-based validation workflow.
    
    Usage:
        pipeline = ValidationPipeline()
        verdicts = pipeline.validate_claims([
            {"claim": "EqProp > BP on Robustness", "eqprop": [0.9, 0.92, ...], "baseline": [0.8, 0.81, ...]},
            ...
        ])
        pipeline.generate_report(verdicts)
    """
    
    def __init__(self, archive_dir: str = "evidence_archive", report_dir: str = "reports"):
        self.archiver = EvidenceArchiver(archive_dir)
        self.reporter = ReportGenerator(report_dir)
    
    def validate_claims(self, claims_data: List[Dict]) -> List[StatisticalVerdict]:
        """
        Takes a list of claim dictionaries and produces verdicts.
        
        Each dict should have:
            - "claim": str (descriptive name)
            - "eqprop": List[float] (raw data)
            - "baseline": List[float] (raw data)
        """
        verdicts = []
        for item in claims_data:
            claim = item["claim"]
            eq_data = item["eqprop"]
            bl_data = item["baseline"]
            
            print(f"🔬 Validating: {claim}...")
            
            verdict = StatisticalVerdict(
                claim=claim,
                eqprop_data=eq_data,
                baseline_data=bl_data
            )
            verdicts.append(verdict)
            
            # Archive raw data
            claim_id = claim.replace(" ", "_").lower()[:30]
            self.archiver.archive(claim_id, {"eqprop": eq_data, "baseline": bl_data}, 
                                  {"verdict": verdict.verdict, "p_value": verdict.p_value})
            
            print(f"   -> {verdict.verdict}")
        
        return verdicts
    
    def generate_report(self, verdicts: List[StatisticalVerdict], title: str = "TorEqProp Validation Report") -> str:
        """Generate the final report."""
        return self.reporter.generate(verdicts, title)

