#!/usr/bin/env python3
"""
TorEqProp Scientific Research Engine

Unified system for rigorous hyperparameter optimization:
- Multi-objective scoring (accuracy + speed + efficiency)
- Progressive validation (micro → small → medium → large)
- Hyperparameter importance analysis (ANOVA, sensitivity)
- Rich TUI dashboard with parameter space visualization

Usage:
    python research_engine.py                  # Interactive TUI
    python research_engine.py --quick          # 5-min rapid mode
    python research_engine.py --campaign       # Full progressive campaign
    python research_engine.py --analyze        # Hyperparameter importance
"""

import argparse
import json
import random
import signal
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple

import numpy as np

# Rich TUI
try:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
    from rich.text import Text
    from rich.align import Align
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "rich", "-q"])
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
    from rich.text import Text
    from rich.align import Align

# Core components
from hyperopt_engine import (
    HyperOptEngine, HyperOptTrial, HyperOptDB,
    EqPropSearchSpace, BaselineSearchSpace, CostAwareEvaluator
)
from statistics import StatisticalAnalyzer


# =============================================================================
# MULTI-OBJECTIVE SCORING
# =============================================================================

@dataclass
class CompositeScore:
    """Multi-objective scoring with configurable weights."""
    accuracy: float
    time_seconds: float
    parameters: int
    
    # Weights (accuracy dominant)
    w_acc: float = 0.70
    w_speed: float = 0.15
    w_eff: float = 0.15
    
    @property
    def speed_score(self) -> float:
        """Inverse time, normalized."""
        return 1.0 / max(1.0, self.time_seconds / 60)  # Per-minute scale
    
    @property
    def efficiency_score(self) -> float:
        """Accuracy per million parameters."""
        return self.accuracy / max(1, self.parameters / 1e6)
    
    @property
    def composite(self) -> float:
        """Weighted composite score."""
        return (self.w_acc * self.accuracy + 
                self.w_speed * min(1.0, self.speed_score) +
                self.w_eff * min(1.0, self.efficiency_score))
    
    def __str__(self):
        return f"acc={self.accuracy:.3f} time={self.time_seconds:.1f}s params={self.parameters} → {self.composite:.3f}"


@dataclass
class ExperimentResult:
    """Rich experiment result with full metadata."""
    trial_id: str
    algorithm: str  # "eqprop" | "bp"
    
    # Task info
    task: str
    tier: str  # "micro" | "small" | "medium" | "large"
    epochs: int
    
    # Config
    config: Dict[str, Any]
    
    # Metrics
    accuracy: float
    time_seconds: float
    parameters: int
    memory_mb: float = 0.0
    
    # Scores
    composite_score: float = 0.0
    pareto_rank: int = 0
    
    # Lineage
    promoted_from: Optional[str] = None
    promoted_to: Optional[str] = None
    
    # Status
    status: str = "complete"
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


# =============================================================================
# PROGRESSIVE VALIDATION TIERS
# =============================================================================

@dataclass
class TaskTier:
    """Definition of a validation tier."""
    name: str
    tasks: List[Tuple[str, int]]  # (task_name, epochs)
    promotion_threshold: float
    max_time_per_trial: float  # seconds


class ProgressiveValidator:
    """Progressive validation: test cheap before expensive."""
    
    TIERS = {
        "micro": TaskTier(
            name="micro",
            tasks=[("xor", 10), ("xor3", 10)],
            promotion_threshold=0.6,
            max_time_per_trial=60
        ),
        "small": TaskTier(
            name="small", 
            tasks=[("mnist", 3), ("fashion", 3)],
            promotion_threshold=0.7,
            max_time_per_trial=300
        ),
        "medium": TaskTier(
            name="medium",
            tasks=[("cartpole", 50), ("mnist", 10)],
            promotion_threshold=0.8,
            max_time_per_trial=600
        ),
        "large": TaskTier(
            name="large",
            tasks=[("cartpole", 200), ("mnist", 50)],
            promotion_threshold=0.9,
            max_time_per_trial=3600
        )
    }
    
    def __init__(self):
        self.tier_results: Dict[str, List[ExperimentResult]] = defaultdict(list)
        self.promoted_configs: Dict[str, List[Dict]] = defaultdict(list)
    
    def get_tier(self, name: str) -> TaskTier:
        return self.TIERS[name]
    
    def should_promote(self, config: Dict, tier: str, results: List[ExperimentResult]) -> bool:
        """Check if config should be promoted to next tier."""
        if not results:
            return False
        
        tier_def = self.TIERS[tier]
        avg_accuracy = np.mean([r.accuracy for r in results])
        return avg_accuracy >= tier_def.promotion_threshold
    
    def get_next_tier(self, current: str) -> Optional[str]:
        tiers = list(self.TIERS.keys())
        idx = tiers.index(current)
        if idx < len(tiers) - 1:
            return tiers[idx + 1]
        return None


# =============================================================================
# HYPERPARAMETER IMPORTANCE ANALYSIS
# =============================================================================

class HyperparameterAnalyzer:
    """Analyze which hyperparameters matter most."""
    
    def __init__(self, results: List[ExperimentResult]):
        self.results = [r for r in results if r.status == "complete"]
    
    def anova_importance(self) -> Dict[str, float]:
        """Compute variance explained by each hyperparameter."""
        if len(self.results) < 10:
            return {}
        
        importance = {}
        
        # Get all config keys
        all_keys = set()
        for r in self.results:
            all_keys.update(r.config.keys())
        
        total_variance = np.var([r.accuracy for r in self.results])
        if total_variance == 0:
            return {}
        
        for key in all_keys:
            # Group by parameter value
            groups = defaultdict(list)
            for r in self.results:
                val = r.config.get(key, "missing")
                groups[str(val)].append(r.accuracy)
            
            if len(groups) < 2:
                continue
            
            # Between-group variance
            group_means = [np.mean(g) for g in groups.values()]
            between_var = np.var(group_means) * len(groups)
            
            # Importance = fraction of variance explained
            importance[key] = min(1.0, between_var / total_variance)
        
        return dict(sorted(importance.items(), key=lambda x: -x[1]))
    
    def sensitivity_analysis(self, param: str) -> Dict[str, float]:
        """Analyze sensitivity to a specific parameter."""
        groups = defaultdict(list)
        for r in self.results:
            val = r.config.get(param, "missing")
            groups[str(val)].append(r.accuracy)
        
        return {k: np.mean(v) for k, v in groups.items()}
    
    def best_config_per_param(self, param: str) -> Tuple[str, float]:
        """Find best value for a parameter."""
        sensitivity = self.sensitivity_analysis(param)
        if not sensitivity:
            return ("unknown", 0.0)
        best = max(sensitivity.items(), key=lambda x: x[1])
        return best


# =============================================================================
# PARETO ANALYSIS
# =============================================================================

class ParetoAnalyzer:
    """Multi-objective Pareto frontier analysis."""
    
    @staticmethod
    def is_dominated(a: ExperimentResult, b: ExperimentResult) -> bool:
        """Check if 'a' is dominated by 'b' (b is better on all objectives)."""
        # Higher accuracy is better, lower time is better, lower params is better
        better_acc = b.accuracy >= a.accuracy
        better_time = b.time_seconds <= a.time_seconds
        better_params = b.parameters <= a.parameters
        
        strictly_better = (b.accuracy > a.accuracy or 
                          b.time_seconds < a.time_seconds or
                          b.parameters < a.parameters)
        
        return better_acc and better_time and better_params and strictly_better
    
    @staticmethod
    def compute_frontier(results: List[ExperimentResult]) -> List[ExperimentResult]:
        """Compute Pareto frontier."""
        frontier = []
        for r in results:
            is_dominated = any(ParetoAnalyzer.is_dominated(r, other) 
                              for other in results if other != r)
            if not is_dominated:
                frontier.append(r)
        return frontier
    
    @staticmethod
    def assign_ranks(results: List[ExperimentResult]) -> List[ExperimentResult]:
        """Assign Pareto ranks to all results."""
        remaining = results.copy()
        rank = 0
        
        while remaining:
            frontier = ParetoAnalyzer.compute_frontier(remaining)
            for r in frontier:
                r.pareto_rank = rank
            remaining = [r for r in remaining if r not in frontier]
            rank += 1
        
        return results


# =============================================================================
# UNIFIED RESEARCH ENGINE
# =============================================================================

class ResearchEngine:
    """Unified scientific research engine."""
    
    def __init__(self, output_dir: str = "research_results"):
        self.console = Console()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Core components
        self.engine = HyperOptEngine()
        self.stats = StatisticalAnalyzer()
        self.validator = ProgressiveValidator()
        
        # Results storage
        self.results: List[ExperimentResult] = []
        self.findings: List[str] = []
        
        # Stats
        self.start_time = time.time()
        self.experiments_run = 0
        self.experiments_success = 0
        
        # Control
        self.running = True
        signal.signal(signal.SIGINT, self._handle_shutdown)
    
    def _handle_shutdown(self, signum, frame):
        self.console.print("\n[yellow]Shutting down gracefully...[/yellow]")
        self.running = False
        self._save_results()
        self._generate_report()
        sys.exit(0)
    
    def run_experiment(self, task: str, algorithm: str, epochs: int,
                       config: Dict, tier: str) -> ExperimentResult:
        """Run a single experiment with full tracking."""
        trial = HyperOptTrial(
            trial_id=f"sci_{task}_{algorithm}_{int(time.time()*1000)}",
            algorithm=algorithm,
            config=config,
            task=task,
            seed=random.randint(0, 9999)
        )
        
        trial = self.engine.evaluator.evaluate(trial, epochs=epochs, show_progress=False)
        
        self.experiments_run += 1
        
        # Create result
        result = ExperimentResult(
            trial_id=trial.trial_id,
            algorithm=algorithm,
            task=task,
            tier=tier,
            epochs=epochs,
            config=config,
            accuracy=trial.performance,
            time_seconds=trial.cost.wall_time_seconds if trial.cost else 0,
            parameters=trial.cost.param_count if trial.cost else 0,
            status=trial.status,
            error=trial.error
        )
        
        if trial.status == "complete":
            self.experiments_success += 1
            
            # Compute composite score
            score = CompositeScore(
                accuracy=result.accuracy,
                time_seconds=result.time_seconds,
                parameters=result.parameters
            )
            result.composite_score = score.composite
            
            self.engine.db.add_trial(trial)
        
        self.results.append(result)
        return result
    
    def run_progressive_campaign(self, max_configs: int = 20):
        """Run full progressive validation campaign."""
        self.console.print(Panel(
            "[bold magenta]PROGRESSIVE VALIDATION CAMPAIGN[/bold magenta]\n"
            "Testing cheap before expensive",
            border_style="magenta"
        ))
        
        # Sample configs
        eqprop_configs = [self.engine.eqprop_space.sample(random.Random()) 
                         for _ in range(max_configs)]
        bp_configs = [self.engine.baseline_space.sample(random.Random())
                     for _ in range(max_configs)]
        
        tier_order = ["micro", "small", "medium"]
        
        for tier_name in tier_order:
            if not self.running:
                break
                
            tier = self.validator.get_tier(tier_name)
            self.console.print(f"\n[bold cyan]═══ TIER: {tier_name.upper()} ═══[/bold cyan]")
            
            tier_results_eq = []
            tier_results_bp = []
            
            for task, epochs in tier.tasks:
                if not self.running:
                    break
                
                self.console.print(f"\n[yellow]Task: {task} ({epochs} epochs)[/yellow]")
                
                # Run EqProp configs
                for i, config in enumerate(eqprop_configs[:5]):  # Top 5 for speed
                    if not self.running:
                        break
                    result = self.run_experiment(task, "eqprop", epochs, config, tier_name)
                    tier_results_eq.append(result)
                    self._print_result(result)
                
                # Run BP configs
                for config in bp_configs[:5]:
                    if not self.running:
                        break
                    result = self.run_experiment(task, "bp", epochs, config, tier_name)
                    tier_results_bp.append(result)
                    self._print_result(result)
            
            # Analyze tier results
            self._analyze_tier(tier_name, tier_results_eq, tier_results_bp)
            
            # Promote top configs
            promoted_eq = self._get_top_configs(tier_results_eq, n=3)
            promoted_bp = self._get_top_configs(tier_results_bp, n=3)
            
            eqprop_configs = promoted_eq + eqprop_configs[5:]
            bp_configs = promoted_bp + bp_configs[5:]
            
            self.console.print(f"[green]Promoted {len(promoted_eq)} EqProp, {len(promoted_bp)} BP configs[/green]")
        
        self._generate_report()
    
    def _print_result(self, result: ExperimentResult):
        """Print single result."""
        icon = "🔋" if result.algorithm == "eqprop" else "⚡"
        if result.status == "complete":
            self.console.print(f"  {icon} {result.accuracy:.3f} ({result.time_seconds:.1f}s)")
        else:
            self.console.print(f"  {icon} [red]FAILED[/red]")
    
    def _get_top_configs(self, results: List[ExperimentResult], n: int) -> List[Dict]:
        """Get top N configs by composite score."""
        successful = [r for r in results if r.status == "complete"]
        sorted_results = sorted(successful, key=lambda x: x.composite_score, reverse=True)
        return [r.config for r in sorted_results[:n]]
    
    def _analyze_tier(self, tier: str, eq_results: List[ExperimentResult], 
                      bp_results: List[ExperimentResult]):
        """Analyze results for a tier."""
        eq_success = [r for r in eq_results if r.status == "complete"]
        bp_success = [r for r in bp_results if r.status == "complete"]
        
        if eq_success and bp_success:
            eq_acc = [r.accuracy for r in eq_success]
            bp_acc = [r.accuracy for r in bp_success]
            
            comparison = self.stats.compare(eq_acc, bp_acc, "EqProp", "BP")
            
            self.console.print(f"\n[bold]Tier {tier} Summary:[/bold]")
            self.console.print(f"  EqProp: {np.mean(eq_acc):.3f} ± {np.std(eq_acc):.3f}")
            self.console.print(f"  BP:     {np.mean(bp_acc):.3f} ± {np.std(bp_acc):.3f}")
            self.console.print(f"  p-value: {comparison.p_value:.4f}")
            
            if comparison.is_significant:
                winner = "EqProp" if comparison.algo1_mean > comparison.algo2_mean else "BP"
                finding = f"{winner} wins tier {tier} (p={comparison.p_value:.3f})"
                self.findings.append(finding)
                self.console.print(f"  [bold green]🎯 {finding}[/bold green]")
    
    def analyze_importance(self):
        """Run hyperparameter importance analysis."""
        self.console.print(Panel(
            "[bold magenta]HYPERPARAMETER IMPORTANCE ANALYSIS[/bold magenta]",
            border_style="magenta"
        ))
        
        eq_results = [r for r in self.results if r.algorithm == "eqprop"]
        bp_results = [r for r in self.results if r.algorithm == "bp"]
        
        for name, results in [("EqProp", eq_results), ("BP", bp_results)]:
            if len(results) < 10:
                self.console.print(f"[yellow]{name}: Need more data (have {len(results)}, need 10+)[/yellow]")
                continue
            
            analyzer = HyperparameterAnalyzer(results)
            importance = analyzer.anova_importance()
            
            self.console.print(f"\n[bold cyan]{name} Hyperparameter Importance:[/bold cyan]")
            
            table = Table(show_header=True)
            table.add_column("Parameter", style="cyan")
            table.add_column("Importance", justify="right")
            table.add_column("Bar", justify="left")
            
            for param, imp in list(importance.items())[:10]:
                bar = "█" * int(imp * 20) + "░" * (20 - int(imp * 20))
                table.add_row(param, f"{imp:.1%}", bar)
            
            self.console.print(table)
    
    def _save_results(self):
        """Save results to JSON."""
        results_file = self.output_dir / "results.json"
        with open(results_file, "w") as f:
            json.dump([r.to_dict() for r in self.results], f, indent=2)
    
    def _generate_report(self):
        """Generate comprehensive report."""
        report_path = self.output_dir / "report.md"
        
        elapsed = (time.time() - self.start_time) / 60
        
        eq_results = [r for r in self.results if r.algorithm == "eqprop" and r.status == "complete"]
        bp_results = [r for r in self.results if r.algorithm == "bp" and r.status == "complete"]
        
        with open(report_path, "w") as f:
            f.write(f"""# 🔬 Scientific Research Report

> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

| Metric | Value |
|--------|-------|
| Runtime | {elapsed:.1f} minutes |
| Experiments | {self.experiments_run} |
| Success Rate | {self.experiments_success/max(1,self.experiments_run)*100:.0f}% |

## Results by Algorithm

| Algorithm | Count | Mean Accuracy | Best |
|-----------|-------|---------------|------|
| EqProp | {len(eq_results)} | {np.mean([r.accuracy for r in eq_results]) if eq_results else 0:.3f} | {max([r.accuracy for r in eq_results]) if eq_results else 0:.3f} |
| BP | {len(bp_results)} | {np.mean([r.accuracy for r in bp_results]) if bp_results else 0:.3f} | {max([r.accuracy for r in bp_results]) if bp_results else 0:.3f} |

## Discoveries

""")
            for finding in self.findings:
                f.write(f"- 🎯 {finding}\n")
            
            if not self.findings:
                f.write("*No significant findings yet. Run more experiments.*\n")
            
            # Hyperparameter importance
            if len(eq_results) >= 10:
                analyzer = HyperparameterAnalyzer(eq_results)
                importance = analyzer.anova_importance()
                
                f.write("\n## EqProp Hyperparameter Importance\n\n")
                f.write("| Parameter | Importance |\n|-----------|------------|\n")
                for param, imp in list(importance.items())[:5]:
                    f.write(f"| {param} | {imp:.1%} |\n")
        
        self.console.print(f"\n[green]📄 Report saved: {report_path}[/green]")
    
    def run_quick(self, minutes: float = 5):
        """Quick validation mode."""
        self.console.print(Panel(
            f"[bold green]QUICK MODE - {minutes} minutes[/bold green]",
            border_style="green"
        ))
        
        deadline = time.time() + minutes * 60
        tier = self.validator.get_tier("micro")
        
        while self.running and time.time() < deadline:
            for task, epochs in tier.tasks:
                if time.time() >= deadline:
                    break
                
                for algo in ["eqprop", "bp"]:
                    if time.time() >= deadline:
                        break
                    
                    if algo == "eqprop":
                        config = self.engine.eqprop_space.sample(random.Random())
                    else:
                        config = self.engine.baseline_space.sample(random.Random())
                    
                    result = self.run_experiment(task, algo, epochs, config, "micro")
                    self._print_result(result)
        
        self.analyze_importance()
        self._generate_report()


def main():
    parser = argparse.ArgumentParser(description="Scientific Research Engine")
    parser.add_argument("--quick", action="store_true", help="Quick 5-minute mode")
    parser.add_argument("--campaign", action="store_true", help="Full progressive campaign")
    parser.add_argument("--analyze", action="store_true", help="Analyze existing results")
    parser.add_argument("--minutes", type=float, default=5, help="Minutes for quick mode")
    args = parser.parse_args()
    
    engine = ResearchEngine()
    
    if args.analyze:
        engine.analyze_importance()
    elif args.campaign:
        engine.run_progressive_campaign()
    else:
        engine.run_quick(minutes=args.minutes)


if __name__ == "__main__":
    main()
