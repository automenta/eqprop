"""
Parameter sensitivity analysis for the research engine.

Provides ANOVA-style importance analysis, sensitivity curves,
and heatmap generation for understanding hyperparameter effects.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

# Optional dependencies for visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from .collector import ResultCollector, Trial
from .config import ResearchConfig, DEFAULT_CONFIG


@dataclass
class ParameterImportance:
    """Importance analysis result for a parameter."""
    parameter: str
    f_statistic: float
    p_value: float
    importance_rank: int
    best_value: Any
    best_performance: float
    value_performance: Dict[Any, float]  # value -> avg performance


@dataclass
class SensitivityResult:
    """Sensitivity analysis for a parameter."""
    parameter: str
    values: List[Any]
    mean_performances: List[float]
    std_performances: List[float]
    correlation: float  # Correlation with performance
    monotonic: bool  # Is the relationship monotonic?


class ParameterAnalyzer:
    """Analyze hyperparameter importance and sensitivity."""
    
    def __init__(
        self,
        collector: Optional[ResultCollector] = None,
        config: ResearchConfig = DEFAULT_CONFIG,
    ):
        self.collector = collector or ResultCollector(config.output_dir)
        self.config = config
        self.output_dir = config.output_dir / "analysis"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_analyzed_params(self) -> List[str]:
        """Get list of parameters to analyze."""
        return ["d_model", "lr", "beta", "damping", "max_iters", "epochs"]
    
    def sensitivity_analysis(
        self,
        trials: Optional[List[Trial]] = None,
        task: Optional[str] = None,
        algorithm: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Compute variance explained by each hyperparameter using ANOVA.
        
        Returns dict of parameter -> F-statistic (higher = more important)
        """
        if trials is None:
            trials = self.collector.get_trials(
                status="complete",
                task=task,
                algorithm=algorithm,
            )
        
        if len(trials) < 5:
            return {}
        
        importances = {}
        
        for param in self.get_analyzed_params():
            groups = self._group_by_param(trials, param)
            
            if len(groups) < 2:
                continue
            
            # Need at least 2 samples per group for ANOVA
            valid_groups = [g for g in groups.values() if len(g) >= 2]
            if len(valid_groups) < 2:
                continue
            
            if HAS_SCIPY:
                try:
                    f_stat, p_value = stats.f_oneway(*valid_groups)
                    if not np.isnan(f_stat):
                        importances[param] = float(f_stat)
                except Exception:
                    continue
            else:
                # Simple variance ratio without scipy
                group_means = [np.mean(g) for g in valid_groups]
                between_var = np.var(group_means)
                within_var = np.mean([np.var(g) for g in valid_groups])
                if within_var > 0:
                    importances[param] = between_var / within_var
        
        return dict(sorted(importances.items(), key=lambda x: -x[1]))
    
    def _group_by_param(
        self,
        trials: List[Trial],
        param: str,
    ) -> Dict[Any, List[float]]:
        """Group trial performances by parameter value."""
        groups: Dict[Any, List[float]] = defaultdict(list)
        
        for trial in trials:
            if param in trial.config:
                value = trial.config[param]
                # Discretize continuous values
                if isinstance(value, float):
                    value = round(value, 4)
                groups[value].append(trial.performance)
        
        return groups
    
    def get_best_value_per_param(
        self,
        trials: Optional[List[Trial]] = None,
    ) -> Dict[str, Tuple[Any, float]]:
        """Get best performing value for each parameter."""
        if trials is None:
            trials = self.collector.get_trials(status="complete")
        
        results = {}
        
        for param in self.get_analyzed_params():
            groups = self._group_by_param(trials, param)
            
            if not groups:
                continue
            
            # Find value with best mean performance
            best_value = None
            best_mean = -1
            
            for value, perfs in groups.items():
                mean_perf = np.mean(perfs)
                if mean_perf > best_mean:
                    best_mean = mean_perf
                    best_value = value
            
            if best_value is not None:
                results[param] = (best_value, best_mean)
        
        return results
    
    def detailed_sensitivity(
        self,
        param: str,
        trials: Optional[List[Trial]] = None,
    ) -> Optional[SensitivityResult]:
        """Get detailed sensitivity analysis for a parameter."""
        if trials is None:
            trials = self.collector.get_trials(status="complete")
        
        groups = self._group_by_param(trials, param)
        
        if len(groups) < 2:
            return None
        
        values = sorted(groups.keys(), key=lambda x: float(x) if isinstance(x, (int, float)) else 0)
        means = [np.mean(groups[v]) for v in values]
        stds = [np.std(groups[v]) for v in values]
        
        # Correlation (for numeric params)
        correlation = 0.0
        try:
            if all(isinstance(v, (int, float)) for v in values):
                all_values = []
                all_perfs = []
                for v in values:
                    for p in groups[v]:
                        all_values.append(float(v))
                        all_perfs.append(p)
                if len(all_values) > 2:
                    correlation = float(np.corrcoef(all_values, all_perfs)[0, 1])
        except Exception:
            pass
        
        # Check monotonicity
        monotonic = True
        for i in range(len(means) - 1):
            if (means[i+1] - means[i]) * (means[1] - means[0]) < 0:
                monotonic = False
                break
        
        return SensitivityResult(
            parameter=param,
            values=values,
            mean_performances=means,
            std_performances=stds,
            correlation=correlation if not np.isnan(correlation) else 0.0,
            monotonic=monotonic,
        )
    
    def generate_heatmap(
        self,
        x_param: str,
        y_param: str,
        task: Optional[str] = None,
        algorithm: Optional[str] = None,
        trials: Optional[List[Trial]] = None,
    ) -> Optional[Path]:
        """
        Generate heatmap of performance for two parameters.
        
        Returns path to saved figure or None if visualization unavailable.
        """
        if not HAS_MATPLOTLIB:
            print("⚠️ matplotlib not available, skipping heatmap generation")
            return None
        
        if trials is None:
            trials = self.collector.get_trials(
                status="complete",
                task=task,
                algorithm=algorithm,
            )
        
        if len(trials) < 5:
            return None
        
        # Create pivot table
        pivot_data: Dict[Tuple[Any, Any], List[float]] = defaultdict(list)
        
        for trial in trials:
            x_val = trial.config.get(x_param)
            y_val = trial.config.get(y_param)
            
            if x_val is not None and y_val is not None:
                pivot_data[(x_val, y_val)].append(trial.performance)
        
        if len(pivot_data) < 4:
            return None
        
        # Get unique values
        x_values = sorted(set(k[0] for k in pivot_data.keys()))
        y_values = sorted(set(k[1] for k in pivot_data.keys()))
        
        # Build matrix
        matrix = np.zeros((len(y_values), len(x_values)))
        for i, y_val in enumerate(y_values):
            for j, x_val in enumerate(x_values):
                perfs = pivot_data.get((x_val, y_val), [])
                matrix[i, j] = np.mean(perfs) if perfs else np.nan
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if HAS_SEABORN:
            import seaborn as sns  # Re-import for use
            mask = np.isnan(matrix)
            sns.heatmap(
                matrix,
                xticklabels=[f"{v:.4g}" if isinstance(v, float) else str(v) for v in x_values],
                yticklabels=[f"{v:.4g}" if isinstance(v, float) else str(v) for v in y_values],
                annot=True,
                fmt=".3f",
                cmap="RdYlGn",
                ax=ax,
                mask=mask,
            )
        else:
            im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto")
            ax.set_xticks(range(len(x_values)))
            ax.set_yticks(range(len(y_values)))
            ax.set_xticklabels([f"{v:.4g}" if isinstance(v, float) else str(v) for v in x_values])
            ax.set_yticklabels([f"{v:.4g}" if isinstance(v, float) else str(v) for v in y_values])
            plt.colorbar(im, ax=ax)
        
        ax.set_xlabel(x_param)
        ax.set_ylabel(y_param)
        
        title = f"Performance: {x_param} vs {y_param}"
        if task:
            title += f" ({task})"
        if algorithm:
            title += f" - {algorithm}"
        ax.set_title(title)
        
        # Save
        filename = f"heatmap_{x_param}_{y_param}"
        if task:
            filename += f"_{task}"
        if algorithm:
            filename += f"_{algorithm}"
        filename += ".png"
        
        path = self.output_dir / filename
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        
        return path
    
    def generate_all_heatmaps(
        self,
        task: Optional[str] = None,
        algorithm: Optional[str] = None,
    ) -> List[Path]:
        """Generate heatmaps for all parameter pairs."""
        paths = []
        params = self.get_analyzed_params()
        
        for i, x_param in enumerate(params):
            for y_param in params[i+1:]:
                path = self.generate_heatmap(
                    x_param, y_param,
                    task=task,
                    algorithm=algorithm,
                )
                if path:
                    paths.append(path)
        
        return paths
    
    def parameter_importance_report(
        self,
        trials: Optional[List[Trial]] = None,
    ) -> str:
        """Generate markdown report of parameter importance."""
        if trials is None:
            trials = self.collector.get_trials(status="complete")
        
        importance = self.sensitivity_analysis(trials)
        best_values = self.get_best_value_per_param(trials)
        
        if not importance:
            return "## Parameter Importance Analysis\n\nInsufficient data for analysis.\n"
        
        md = "## Parameter Importance Analysis\n\n"
        md += f"*Based on {len(trials)} completed experiments*\n\n"
        
        md += "### ANOVA Importance Ranking\n\n"
        md += "| Rank | Parameter | F-statistic | Best Value | Best Perf | Importance |\n"
        md += "|------|-----------|-------------|------------|-----------|------------|\n"
        
        max_f = max(importance.values()) if importance else 1
        
        for rank, (param, f_stat) in enumerate(importance.items(), 1):
            best = best_values.get(param, (None, 0))
            bars = "█" * int(f_stat / max_f * 10)
            
            best_val_str = f"{best[0]:.4g}" if isinstance(best[0], float) else str(best[0])
            
            md += f"| {rank} | `{param}` | {f_stat:.2f} | {best_val_str} | {best[1]:.3f} | {bars} |\n"
        
        md += "\n### Interpretation\n\n"
        
        if importance:
            top_param = list(importance.keys())[0]
            md += f"- **Most important parameter**: `{top_param}` (F={importance[top_param]:.2f})\n"
            
            if len(importance) > 1:
                second_param = list(importance.keys())[1]
                md += f"- **Second most important**: `{second_param}` (F={importance[second_param]:.2f})\n"
        
        md += "\n*Higher F-statistic indicates the parameter explains more variance in performance.*\n"
        
        return md
    
    def save_analysis_json(self, trials: Optional[List[Trial]] = None) -> Path:
        """Save full analysis as JSON for external tools."""
        if trials is None:
            trials = self.collector.get_trials(status="complete")
        
        analysis = {
            "n_trials": len(trials),
            "importance": self.sensitivity_analysis(trials),
            "best_values": {
                k: {"value": v[0], "performance": v[1]}
                for k, v in self.get_best_value_per_param(trials).items()
            },
            "sensitivity": {},
        }
        
        for param in self.get_analyzed_params():
            result = self.detailed_sensitivity(param, trials)
            if result:
                analysis["sensitivity"][param] = {
                    "values": [str(v) for v in result.values],
                    "means": result.mean_performances,
                    "stds": result.std_performances,
                    "correlation": result.correlation,
                    "monotonic": result.monotonic,
                }
        
        path = self.output_dir / "parameter_analysis.json"
        with open(path, "w") as f:
            json.dump(analysis, f, indent=2)
        
        return path
