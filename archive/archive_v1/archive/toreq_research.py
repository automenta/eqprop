#!/usr/bin/env python3
"""
TorEqProp Research Dashboard

ONE unified interface for hyperparameter optimization.
Shows EVERYTHING: configs, scores, parameter importance, recommendations.
Gets actionable results in MINUTES that guide future investment.

Usage:
    python toreq_research.py           # Default 5 minutes
    python toreq_research.py 10        # Run for 10 minutes
    python toreq_research.py --patience high  # Extended exploration
"""

import argparse
import random
import signal
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
import json

import numpy as np

# Rich TUI
try:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.text import Text
    from rich import box
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "rich", "-q"])
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.text import Text
    from rich import box

from hyperopt_engine import (
    HyperOptEngine, HyperOptTrial, EqPropSearchSpace, BaselineSearchSpace
)
from statistics import StatisticalAnalyzer

console = Console()


@dataclass
class TrialResult:
    """Complete trial result with all details."""
    algorithm: str
    task: str
    accuracy: float
    time_sec: float
    config: Dict[str, Any]
    composite: float = 0.0


class TorEqResearch:
    """
    THE research dashboard. One interface, all insights.
    
    Core purpose: Demonstrate TorEqProp's potential vs baselines
    through rigorous, transparent hyperparameter comparison.
    """
    
    # Task ladder: fast → slow
    TASKS = [
        ("xor", 10, "micro"),      # ~10s
        ("xor3", 10, "micro"),     # ~15s  
        ("mnist", 3, "small"),     # ~60s
        ("cartpole", 30, "small"), # ~90s
    ]
    
    def __init__(self, patience: str = "normal"):
        self.engine = HyperOptEngine()
        self.stats = StatisticalAnalyzer()
        
        # All results
        self.results: List[TrialResult] = []
        self.best_per_task: Dict[str, Dict[str, TrialResult]] = defaultdict(dict)
        
        # Parameter tracking for importance
        self.param_performance: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        
        # Stats
        self.start_time = time.time()
        self.running = True
        
        # Patience controls speed vs depth
        self.patience = patience
        self.trials_per_config = {"low": 1, "normal": 2, "high": 3}[patience]
        
        signal.signal(signal.SIGINT, self._shutdown)
    
    def _shutdown(self, *args):
        console.print("\n[yellow]Saving and exiting...[/yellow]")
        self.running = False
    
    def run(self, minutes: float = 5):
        """Main research loop."""
        deadline = time.time() + minutes * 60
        
        console.print(Panel(
            f"[bold magenta]TorEqProp Research Dashboard[/bold magenta]\n"
            f"Time budget: {minutes} min | Patience: {self.patience}",
            border_style="magenta"
        ))
        
        round_num = 0
        while self.running and time.time() < deadline:
            round_num += 1
            elapsed = (time.time() - self.start_time) / 60
            remaining = minutes - elapsed
            
            console.print(f"\n[bold cyan]═══ Round {round_num} | {elapsed:.1f}m elapsed, {remaining:.1f}m remaining ═══[/bold cyan]")
            
            for task, epochs, tier in self.TASKS:
                if not self.running or time.time() >= deadline:
                    break
                
                # Run both algorithms
                eq_result = self._run_trial("eqprop", task, epochs)
                bp_result = self._run_trial("bp", task, epochs)
                
                # Show comparison
                self._show_comparison(task, eq_result, bp_result)
                
                # Early analysis after each task
                if len(self.results) >= 4:
                    self._show_current_insights()
        
        # Final report
        self._final_report()
    
    def _run_trial(self, algo: str, task: str, epochs: int) -> Optional[TrialResult]:
        """Run a single trial, showing full config."""
        
        # Sample config
        if algo == "eqprop":
            config = self.engine.eqprop_space.sample(random.Random())
        else:
            config = self.engine.baseline_space.sample(random.Random())
        
        # Show what we're running
        config_str = self._format_config(algo, config)
        console.print(f"\n{'🔋' if algo == 'eqprop' else '⚡'} [bold]{algo.upper()}[/bold] on {task}")
        console.print(f"   Config: {config_str}")
        
        trial = HyperOptTrial(
            trial_id=f"{algo}_{task}_{int(time.time()*1000)}",
            algorithm=algo,
            config=config,
            task=task,
            seed=random.randint(0, 9999)
        )
        
        start = time.time()
        trial = self.engine.evaluator.evaluate(trial, epochs=epochs, show_progress=False)
        elapsed = time.time() - start
        
        if trial.status != "complete":
            console.print(f"   [red]FAILED[/red]: {trial.error}")
            return None
        
        # Create result
        result = TrialResult(
            algorithm=algo,
            task=task,
            accuracy=trial.performance,
            time_sec=elapsed,
            config=config
        )
        result.composite = 0.7 * result.accuracy + 0.3 * (1 / max(1, elapsed/60))
        
        self.results.append(result)
        self.engine.db.add_trial(trial)
        
        # Track for importance analysis
        for param, value in config.items():
            self.param_performance[param][str(value)].append(result.accuracy)
        
        # Update best
        if task not in self.best_per_task[algo] or result.accuracy > self.best_per_task[algo][task].accuracy:
            self.best_per_task[algo][task] = result
        
        console.print(f"   Result: [{'green' if result.accuracy > 0.6 else 'yellow'}]{result.accuracy:.3f}[/] in {elapsed:.1f}s")
        
        return result
    
    def _format_config(self, algo: str, config: Dict) -> str:
        """Format config for display - show key params."""
        if algo == "eqprop":
            key_params = ["beta", "damping", "d_model", "attention_type", "lr"]
        else:
            key_params = ["lr", "optimizer", "d_model"]
        
        parts = []
        for p in key_params:
            if p in config:
                v = config[p]
                if isinstance(v, float):
                    parts.append(f"{p}={v:.4g}")
                else:
                    parts.append(f"{p}={v}")
        return ", ".join(parts)
    
    def _show_comparison(self, task: str, eq: Optional[TrialResult], bp: Optional[TrialResult]):
        """Show head-to-head comparison."""
        if not eq or not bp:
            return
        
        diff = eq.accuracy - bp.accuracy
        winner = "🔋 EqProp" if diff > 0 else "⚡ BP" if diff < 0 else "TIE"
        color = "green" if diff > 0 else "red" if diff < 0 else "yellow"
        
        console.print(f"   [bold {color}]→ {winner}[/] ({diff:+.1%})")
    
    def _show_current_insights(self):
        """Show actionable insights from current data."""
        eq_results = [r for r in self.results if r.algorithm == "eqprop"]
        bp_results = [r for r in self.results if r.algorithm == "bp"]
        
        if not eq_results or not bp_results:
            return
        
        # Overall comparison
        eq_mean = np.mean([r.accuracy for r in eq_results])
        bp_mean = np.mean([r.accuracy for r in bp_results])
        
        # Parameter importance (simple variance analysis)
        importance = {}
        for param, values in self.param_performance.items():
            if len(values) >= 2:
                means = [np.mean(perfs) for perfs in values.values() if len(perfs) > 0]
                if len(means) >= 2:
                    importance[param] = np.std(means)
        
        # Display insights panel
        insights = []
        
        # Overall verdict
        if eq_mean > bp_mean * 1.05:
            insights.append(f"✅ EqProp leading: {eq_mean:.1%} vs BP {bp_mean:.1%}")
        elif bp_mean > eq_mean * 1.05:
            insights.append(f"⚠️ BP leading: {bp_mean:.1%} vs EqProp {eq_mean:.1%}")
        else:
            insights.append(f"🔄 Close race: EqProp {eq_mean:.1%} vs BP {bp_mean:.1%}")
        
        # Top parameter insights
        if importance:
            sorted_imp = sorted(importance.items(), key=lambda x: -x[1])[:3]
            insights.append(f"📊 Key params: {', '.join(p for p,_ in sorted_imp)}")
            
            # Best value for top param
            top_param = sorted_imp[0][0]
            best_val = max(self.param_performance[top_param].items(), 
                          key=lambda x: np.mean(x[1]) if x[1] else 0)
            insights.append(f"   Best {top_param}={best_val[0]} (avg {np.mean(best_val[1]):.1%})")
        
        console.print(Panel("\n".join(insights), title="Current Insights", border_style="cyan"))
    
    def _final_report(self):
        """Generate comprehensive final report."""
        elapsed = (time.time() - self.start_time) / 60
        
        console.print("\n" + "=" * 70)
        console.print("[bold magenta]FINAL RESEARCH REPORT[/bold magenta]")
        console.print("=" * 70)
        
        # Summary stats
        eq_results = [r for r in self.results if r.algorithm == "eqprop"]
        bp_results = [r for r in self.results if r.algorithm == "bp"]
        
        console.print(f"\n⏱️  Runtime: {elapsed:.1f} minutes")
        console.print(f"🧪 Total experiments: {len(self.results)}")
        
        # Scoreboard
        table = Table(title="Scoreboard", box=box.ROUNDED)
        table.add_column("Task")
        table.add_column("EqProp Best", style="green")
        table.add_column("BP Best", style="yellow")
        table.add_column("Winner")
        
        tasks = set(r.task for r in self.results)
        eq_wins = bp_wins = 0
        
        for task in sorted(tasks):
            eq_best = self.best_per_task["eqprop"].get(task)
            bp_best = self.best_per_task["bp"].get(task)
            
            eq_score = f"{eq_best.accuracy:.3f}" if eq_best else "-"
            bp_score = f"{bp_best.accuracy:.3f}" if bp_best else "-"
            
            if eq_best and bp_best:
                if eq_best.accuracy > bp_best.accuracy:
                    winner = "🔋 EqProp"
                    eq_wins += 1
                elif bp_best.accuracy > eq_best.accuracy:
                    winner = "⚡ BP"
                    bp_wins += 1
                else:
                    winner = "Tie"
            else:
                winner = "-"
            
            table.add_row(task, eq_score, bp_score, winner)
        
        console.print(table)
        console.print(f"\n[bold]Overall: EqProp {eq_wins} wins, BP {bp_wins} wins[/bold]")
        
        # Parameter importance
        if len(eq_results) >= 3:
            console.print("\n[bold cyan]EqProp Parameter Importance:[/bold cyan]")
            
            imp_table = Table(box=box.SIMPLE)
            imp_table.add_column("Parameter")
            imp_table.add_column("Best Value")
            imp_table.add_column("Impact")
            
            importance = {}
            best_vals = {}
            for param, values in self.param_performance.items():
                if len(values) >= 2:
                    means = {k: np.mean(v) for k, v in values.items() if v}
                    if means:
                        importance[param] = np.std(list(means.values()))
                        best_vals[param] = max(means.items(), key=lambda x: x[1])
            
            for param in sorted(importance, key=lambda x: -importance[x])[:6]:
                val, score = best_vals[param]
                bar = "█" * int(importance[param] * 50)
                imp_table.add_row(param, f"{val}", bar[:20])
            
            console.print(imp_table)
        
        # Best configs
        console.print("\n[bold cyan]Best EqProp Configurations:[/bold cyan]")
        for task, result in self.best_per_task["eqprop"].items():
            console.print(f"\n{task} ({result.accuracy:.3f}):")
            for k, v in sorted(result.config.items()):
                if isinstance(v, float):
                    console.print(f"  {k}: {v:.4g}")
                else:
                    console.print(f"  {k}: {v}")
        
        # Recommendation
        console.print("\n[bold green]💡 RECOMMENDATIONS:[/bold green]")
        
        if eq_wins > bp_wins:
            console.print("✅ EqProp shows promise! Invest more time in:")
            # Find best performing task
            best_task = max(self.best_per_task["eqprop"].items(), 
                           key=lambda x: x[1].accuracy)
            console.print(f"   - Extended training on {best_task[0]}")
            console.print(f"   - Scale up d_model for better capacity")
        else:
            console.print("⚠️ BP currently leads. Consider:")
            console.print("   - Tuning beta/damping parameters")
            console.print("   - Trying different attention types")
        
        # Save to file
        self._save_report(elapsed)
    
    def _save_report(self, elapsed: float):
        """Save report to markdown."""
        Path("research_results").mkdir(exist_ok=True)
        
        with open("research_results/report.md", "w") as f:
            f.write(f"""# TorEqProp Research Report

> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
> Runtime: {elapsed:.1f} minutes | Experiments: {len(self.results)}

## Results Summary

| Task | EqProp Best | BP Best | Winner |
|------|-------------|---------|--------|
""")
            for task in sorted(set(r.task for r in self.results)):
                eq = self.best_per_task["eqprop"].get(task)
                bp = self.best_per_task["bp"].get(task)
                eq_s = f"{eq.accuracy:.3f}" if eq else "-"
                bp_s = f"{bp.accuracy:.3f}" if bp else "-"
                w = "EqProp" if eq and bp and eq.accuracy > bp.accuracy else "BP" if bp and eq and bp.accuracy > eq.accuracy else "-"
                f.write(f"| {task} | {eq_s} | {bp_s} | {w} |\n")
            
            f.write("\n## Best EqProp Configurations\n\n")
            for task, result in self.best_per_task["eqprop"].items():
                f.write(f"### {task} ({result.accuracy:.3f})\n```yaml\n")
                for k, v in sorted(result.config.items()):
                    f.write(f"{k}: {v}\n")
                f.write("```\n\n")
            
            f.write("## All Trials\n\n")
            f.write("| # | Algo | Task | Accuracy | Time | Key Config |\n")
            f.write("|---|------|------|----------|------|------------|\n")
            for i, r in enumerate(self.results, 1):
                cfg = self._format_config(r.algorithm, r.config)
                f.write(f"| {i} | {r.algorithm} | {r.task} | {r.accuracy:.3f} | {r.time_sec:.1f}s | {cfg} |\n")
        
        console.print(f"\n[green]📄 Report saved: research_results/report.md[/green]")


def main():
    parser = argparse.ArgumentParser(description="TorEqProp Research Dashboard")
    parser.add_argument("minutes", type=float, nargs="?", default=5, 
                       help="Minutes to run (default: 5)")
    parser.add_argument("--patience", choices=["low", "normal", "high"], 
                       default="normal", help="Exploration depth")
    args = parser.parse_args()
    
    console.print("""
[bold magenta]╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   🔬 TorEqProp vs Baselines: Scientific Comparison 🔬        ║
║                                                               ║
║   Goal: Demonstrate TorEqProp potential through rigorous     ║
║         hyperparameter exploration with full transparency    ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝[/bold magenta]
""")
    
    research = TorEqResearch(patience=args.patience)
    research.run(minutes=args.minutes)


if __name__ == "__main__":
    main()
