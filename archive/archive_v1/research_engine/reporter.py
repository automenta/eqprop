"""
Report generation for the research engine.

Generates publication-ready markdown reports with:
- Executive summaries
- Statistical verdicts
- Parameter importance analysis
- Visualizations
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from .collector import ResultCollector, Trial
from .analyzer import ParameterAnalyzer
from .config import ResearchConfig, DEFAULT_CONFIG


@dataclass
class StatisticalVerdict:
    """A rigorous statistical conclusion."""
    claim: str
    eqprop_data: List[float]
    baseline_data: List[float]
    
    # Computed fields
    eqprop_mean: float = 0.0
    eqprop_std: float = 0.0
    eqprop_n: int = 0
    baseline_mean: float = 0.0
    baseline_std: float = 0.0
    baseline_n: int = 0
    
    p_value: float = 1.0
    test_type: str = ""
    cohens_d: float = 0.0
    effect_size_category: str = ""
    is_significant: bool = False
    verdict: str = ""
    
    def __post_init__(self):
        self._compute()
    
    def _compute(self):
        """Compute all statistics from raw data."""
        eq = np.array(self.eqprop_data)
        bl = np.array(self.baseline_data)
        
        self.eqprop_n = len(eq)
        self.baseline_n = len(bl)
        self.eqprop_mean = float(np.mean(eq)) if len(eq) else 0.0
        self.eqprop_std = float(np.std(eq, ddof=1)) if len(eq) > 1 else 0.0
        self.baseline_mean = float(np.mean(bl)) if len(bl) else 0.0
        self.baseline_std = float(np.std(bl, ddof=1)) if len(bl) > 1 else 0.0
        
        if len(eq) < 2 or len(bl) < 2:
            self.test_type = "N/A (Insufficient Data)"
            self.verdict = "INCONCLUSIVE (Need more samples)"
            return
        
        # Statistical test
        if HAS_SCIPY:
            if self.eqprop_n >= 30 and self.baseline_n >= 30:
                _, self.p_value = stats.ttest_ind(eq, bl, equal_var=False)
                self.test_type = "Welch's t-test"
            else:
                try:
                    _, self.p_value = stats.mannwhitneyu(eq, bl, alternative='two-sided')
                    self.test_type = "Mann-Whitney U"
                except ValueError:
                    self.p_value = 1.0
                    self.test_type = "N/A (Test failed)"
        else:
            self.test_type = "N/A (scipy not available)"
            self.p_value = 1.0
        
        # Cohen's d effect size
        pooled_std = np.sqrt(((self.eqprop_n - 1) * self.eqprop_std**2 + 
                              (self.baseline_n - 1) * self.baseline_std**2) /
                             max(1, self.eqprop_n + self.baseline_n - 2))
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
        
        # Verdict
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
                self.verdict = "INCONCLUSIVE (Effect observed, not significant)"
    
    def to_markdown(self) -> str:
        """Generate markdown summary."""
        emoji = "✅" if "VALIDATED" in self.verdict else ("❌" if "REJECTED" in self.verdict else "❔")
        
        return f"""
### {emoji} {self.claim}

| Metric | EqProp | Baseline |
|--------|--------|----------|
| Mean | {self.eqprop_mean:.4f} | {self.baseline_mean:.4f} |
| Std Dev | {self.eqprop_std:.4f} | {self.baseline_std:.4f} |
| N | {self.eqprop_n} | {self.baseline_n} |

**Test**: {self.test_type} | **p-value**: {self.p_value:.4f} | **Cohen's d**: {self.cohens_d:.2f} ({self.effect_size_category})

**Verdict**: **{self.verdict}**
"""


class EvidenceArchiver:
    """Archives raw data with checksums for reproducibility."""
    
    def __init__(self, archive_dir: Path):
        self.archive_dir = archive_dir
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
    
    def archive(
        self,
        claim_id: str,
        data: Dict[str, Any],
        metadata: Optional[Dict] = None,
    ) -> str:
        """Archive data with checksum. Returns filepath."""
        timestamp = datetime.now().isoformat()
        filename = f"{claim_id}_{timestamp.replace(':', '-').replace('.', '_')}.json"
        filepath = self.archive_dir / filename
        
        content = {
            "claim_id": claim_id,
            "timestamp": timestamp,
            "data": data,
            "metadata": metadata or {},
        }
        
        json_str = json.dumps(content, indent=2, default=str)
        checksum = hashlib.sha256(json_str.encode()).hexdigest()
        content["checksum"] = checksum
        
        with open(filepath, 'w') as f:
            json.dump(content, f, indent=2, default=str)
        
        self.manifest["entries"].append({
            "claim_id": claim_id,
            "filename": filename,
            "checksum": checksum,
            "timestamp": timestamp,
        })
        self._save_manifest()
        
        return str(filepath)
    
    def verify(self, filepath: str) -> bool:
        """Verify data integrity."""
        with open(filepath, 'r') as f:
            content = json.load(f)
        
        stored_checksum = content.pop("checksum", None)
        json_str = json.dumps(content, indent=2, default=str)
        computed = hashlib.sha256(json_str.encode()).hexdigest()
        
        return stored_checksum == computed


class ResearchReporter:
    """Generate comprehensive research reports."""
    
    def __init__(
        self,
        collector: Optional[ResultCollector] = None,
        config: ResearchConfig = DEFAULT_CONFIG,
    ):
        self.collector = collector or ResultCollector(config.output_dir)
        self.config = config
        self.output_dir = config.output_dir / "reports"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.evidence_dir = config.output_dir / "evidence"
        self.archiver = EvidenceArchiver(self.evidence_dir)
    
    def generate_live_summary(self, stats: Dict[str, Any]) -> str:
        """Quick summary for dashboard updates."""
        return f"""## Live Research Summary

- **Experiments**: {stats.get('completed', 0)}/{stats.get('total', 0)}
- **EqProp wins**: {stats.get('eqprop_wins', 0)} | **BP wins**: {stats.get('bp_wins', 0)}
- **Success rate**: {stats.get('success_rate', 0) * 100:.1f}%
- **Time elapsed**: {stats.get('elapsed_minutes', 0):.1f} min
"""
    
    def generate_final_report(
        self,
        analyzer: Optional[ParameterAnalyzer] = None,
        title: str = "TorEqProp Research Report",
    ) -> Path:
        """Generate comprehensive final report."""
        trials = self.collector.get_trials(status="complete")
        
        if analyzer is None:
            analyzer = ParameterAnalyzer(self.collector, self.config)
        
        sections = [
            self._generate_header(title, len(trials)),
            self._generate_executive_summary(trials),
            self._generate_methodology(),
            self._generate_results_table(trials),
            analyzer.parameter_importance_report(trials),
            self._generate_statistical_verdicts(trials),
            self._generate_best_configs(trials),
            self._generate_conclusions(trials),
        ]
        
        report = "\n\n---\n\n".join(sections)
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"report_{timestamp}.md"
        report_path.write_text(report)
        
        # Also save as latest
        latest_path = self.output_dir / "latest_report.md"
        latest_path.write_text(report)
        
        # Archive evidence
        self._archive_evidence(trials)
        
        return report_path
    
    def _generate_header(self, title: str, n_trials: int) -> str:
        """Generate report header."""
        return f"""# {title}

> **Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M")}
> **Total Experiments**: {n_trials}
> **Framework**: TorEqProp Unified Research Engine v1.0
"""
    
    def _generate_executive_summary(self, trials: List[Trial]) -> str:
        """Generate executive summary."""
        stats = self.collector.get_statistics()
        
        # Get best results per task
        tasks = set(t.task for t in trials)
        
        md = "## Executive Summary\n\n"
        
        # Win/loss summary
        md += f"### Overall Comparison\n\n"
        md += f"- **EqProp Wins**: {stats.get('eqprop_wins', 0)} tasks\n"
        md += f"- **BP Wins**: {stats.get('bp_wins', 0)} tasks\n"
        md += f"- **Ties**: {stats.get('ties', 0)} tasks\n\n"
        
        # Best results table
        md += "### Best Results by Task\n\n"
        md += "| Task | Best EqProp | Best BP | Winner |\n"
        md += "|------|-------------|---------|--------|\n"
        
        for task in sorted(tasks):
            eq_trials = [t for t in trials if t.task == task and t.algorithm == "eqprop"]
            bp_trials = [t for t in trials if t.task == task and t.algorithm == "bp"]
            
            eq_best = max((t.performance for t in eq_trials), default=0)
            bp_best = max((t.performance for t in bp_trials), default=0)
            
            if eq_best > bp_best:
                winner = "🔋 EqProp"
            elif bp_best > eq_best:
                winner = "⚡ BP"
            else:
                winner = "➖ Tie"
            
            md += f"| {task} | {eq_best:.4f} | {bp_best:.4f} | {winner} |\n"
        
        return md
    
    def _generate_methodology(self) -> str:
        """Generate methodology section."""
        return """## Methodology

### Experimental Setup
- **Progressive Validation**: Experiments start with fast micro-tasks and promote promising configurations
- **Time Budget**: Each preliminary experiment limited to ~10 seconds
- **Fair Comparison**: Matched parameter counts and training budgets

### Statistical Analysis
- **Significance Testing**: Mann-Whitney U (small samples) or Welch's t-test (large samples)
- **Effect Size**: Cohen's d with standard thresholds (small: 0.2, medium: 0.5, large: 0.8)
- **Parameter Importance**: ANOVA-style F-statistic ranking

### Evidence Integrity
- All raw data archived with SHA-256 checksums
- Reproducible through saved configurations and random seeds
"""
    
    def _generate_results_table(self, trials: List[Trial]) -> str:
        """Generate detailed results table."""
        md = "## Detailed Results\n\n"
        
        # Group by task and algorithm
        tasks = sorted(set(t.task for t in trials))
        
        for task in tasks:
            md += f"### {task.upper()}\n\n"
            md += "| Algorithm | Performance | Time (s) | d_model | lr | Config |\n"
            md += "|-----------|-------------|----------|---------|-----|--------|\n"
            
            task_trials = sorted(
                [t for t in trials if t.task == task],
                key=lambda t: -t.performance
            )[:10]  # Top 10
            
            for trial in task_trials:
                algo = "🔋 EqProp" if trial.algorithm == "eqprop" else "⚡ BP"
                config_str = f"β={trial.config.get('beta', 'N/A')}" if trial.algorithm == "eqprop" else ""
                
                md += f"| {algo} | {trial.performance:.4f} | {trial.cost.wall_time_seconds:.1f} | "
                md += f"{trial.config.get('d_model', '-')} | {trial.config.get('lr', '-')} | {config_str} |\n"
            
            md += "\n"
        
        return md
    
    def _generate_statistical_verdicts(self, trials: List[Trial]) -> str:
        """Generate rigorous statistical comparisons."""
        md = "## Statistical Verdicts\n\n"
        
        tasks = sorted(set(t.task for t in trials))
        
        for task in tasks:
            eq_perfs = [t.performance for t in trials 
                       if t.task == task and t.algorithm == "eqprop"]
            bp_perfs = [t.performance for t in trials 
                       if t.task == task and t.algorithm == "bp"]
            
            if len(eq_perfs) >= 2 and len(bp_perfs) >= 2:
                verdict = StatisticalVerdict(
                    claim=f"EqProp vs BP on {task}",
                    eqprop_data=eq_perfs,
                    baseline_data=bp_perfs,
                )
                md += verdict.to_markdown()
                md += "\n"
        
        return md
    
    def _generate_best_configs(self, trials: List[Trial]) -> str:
        """Generate best configuration recommendations."""
        md = "## Recommended Configurations\n\n"
        
        # Best EqProp configs
        eq_trials = [t for t in trials if t.algorithm == "eqprop" and t.performance > 0]
        if eq_trials:
            best_eq = max(eq_trials, key=lambda t: t.performance)
            md += "### Best EqProp Configuration\n\n"
            md += f"- **Task**: {best_eq.task}\n"
            md += f"- **Performance**: {best_eq.performance:.4f}\n"
            md += "- **Config**:\n"
            for key, value in best_eq.config.items():
                if not key.startswith("_"):
                    md += f"  - `{key}`: {value}\n"
            md += "\n"
        
        # Best BP configs
        bp_trials = [t for t in trials if t.algorithm == "bp" and t.performance > 0]
        if bp_trials:
            best_bp = max(bp_trials, key=lambda t: t.performance)
            md += "### Best BP Configuration\n\n"
            md += f"- **Task**: {best_bp.task}\n"
            md += f"- **Performance**: {best_bp.performance:.4f}\n"
            md += "- **Config**:\n"
            for key, value in best_bp.config.items():
                if not key.startswith("_"):
                    md += f"  - `{key}`: {value}\n"
            md += "\n"
        
        return md
    
    def _generate_conclusions(self, trials: List[Trial]) -> str:
        """Generate conclusions section."""
        stats = self.collector.get_statistics()
        
        md = "## Conclusions\n\n"
        
        if stats.get("eqprop_wins", 0) > stats.get("bp_wins", 0):
            md += "**Key Finding**: EqProp outperforms BP on more tasks in this evaluation.\n\n"
        elif stats.get("bp_wins", 0) > stats.get("eqprop_wins", 0):
            md += "**Key Finding**: BP outperforms EqProp on more tasks in this evaluation.\n\n"
        else:
            md += "**Key Finding**: EqProp and BP show comparable performance across tasks.\n\n"
        
        md += "### Next Steps\n\n"
        md += "1. Extend training for promising configurations\n"
        md += "2. Increase sample sizes for statistical significance\n"
        md += "3. Explore underrepresented parameter regions\n"
        md += "4. Validate on larger-scale tasks\n"
        
        md += f"\n---\n\n*Report generated by TorEqProp Unified Research Engine*\n"
        md += f"*Checksum: {self.collector.generate_checksum()[:16]}...*\n"
        
        return md
    
    def _archive_evidence(self, trials: List[Trial]):
        """Archive raw trial data for reproducibility."""
        self.archiver.archive(
            "full_results",
            {"trials": [t.to_dict() for t in trials]},
            {
                "n_trials": len(trials),
                "timestamp": datetime.now().isoformat(),
            }
        )
