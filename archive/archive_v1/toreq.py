#!/usr/bin/env python3
"""
TorEqProp - THE research system

One clean interface for rigorous hyperparameter optimization.
Demonstrates TorEqProp vs baselines with full transparency.

Usage:
    python toreq.py                    # Quick 3-min validation
    python toreq.py 10                 # Run 10 minutes
    python toreq.py --deep             # Deep exploration mode
"""

import argparse
import random
import signal
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Any
import json
import logging

import numpy as np

# Suppress Optuna logging to clean up output
import optuna
optuna.logging.set_verbosity(optuna.logging.WARN)

# Progress bars
try:
    from tqdm import tqdm
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "tqdm", "-q"])
    from tqdm import tqdm

# Rich for nice output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "rich", "-q"])
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel

from hyperopt_engine import OptunaHyperoptEngine
from statistics import StatisticalAnalyzer

console = Console()


# =============================================================================
# CORE RESEARCH SYSTEM
# =============================================================================

@dataclass
class Result:
    """Trial result."""
    algorithm: str
    task: str
    accuracy: float
    time: float
    config: Dict[str, Any]


class TorEq:
    """THE core research system."""
    
    def __init__(self, breadth: str = "normal"):
        # Use in-memory DB for breadth='low' (simulated by minutes < 5 in main) or file for persistence
        if breadth == "low":
             self.engine = OptunaHyperoptEngine(storage_url="sqlite:///toreq_hyperopt_smoke.db", study_name="toreq_smoke")
        else:
             self.engine = OptunaHyperoptEngine(storage_url="sqlite:///toreq_hyperopt.db", study_name="toreq_main")
             
        self.stats = StatisticalAnalyzer()
        self.breadth = breadth
        
        self.results: List[Result] = []
        
        self.running = True
        signal.signal(signal.SIGINT, lambda *args: setattr(self, 'running', False))
    
    def run(self, minutes: float = 3):
        """Main research loop."""
        console.print(Panel(
            f"[bold magenta]TorEqProp Research (Optuna-Powered)[/bold magenta]\n"
            f"Time: {minutes}min | Breadth: {self.breadth}",
            title="🔬 Scientific Comparison"
        ))
        
        deadline = time.time() + minutes * 60
        # Focus on key tasks
        tasks = ["mnist", "cartpole"] if self.breadth != "low" else ["xor", "mnist"]
        
        round_num = 0
        while self.running and time.time() < deadline:
            round_num += 1
            console.print(f"\n[bold cyan]Round {round_num}[/bold cyan]")
            
            for task in tasks:
                if not self.running or time.time() >= deadline:
                    break
                
                # Determine epochs based on task
                epochs = 3
                if task == "mnist": epochs = 3
                elif task == "cartpole": epochs = 20
                elif task == "xor": epochs = 10
                
                # Run both algorithms
                console.print(f"\n[yellow]{task}[/yellow] ({epochs} epochs)")
                
                # EqProp Step
                # Transfer Seeding Strategy
                seed_params = None
                if task == "fashion":
                    seed_params = self.engine.get_best_params("mnist", "eqprop")
                    if seed_params: console.print(f"  🌱 Seeding from MNIST")
                elif task == "cifar10":
                     seed_params = self.engine.get_best_params("fashion", "eqprop") or self.engine.get_best_params("mnist", "eqprop")
                     if seed_params: console.print(f"  🌱 Seeding from Fashion/MNIST")

                eq_study = self.engine.run_study(n_trials=1, task=task, algorithm="eqprop", epochs=epochs, seed_params=seed_params)
                eq_trial = eq_study.trials[-1] if eq_study.trials else None
                
                # BP Step
                seed_params_bp = None
                if task == "fashion":
                     seed_params_bp = self.engine.get_best_params("mnist", "bp")
                
                bp_study = self.engine.run_study(n_trials=1, task=task, algorithm="bp", epochs=epochs, seed_params=seed_params_bp)
                bp_trial = bp_study.trials[-1] if bp_study.trials else None
                
                # Display Comparison
                if eq_trial and bp_trial and eq_trial.state == optuna.trial.TrialState.COMPLETE and bp_trial.state == optuna.trial.TrialState.COMPLETE:
                    eq_val = eq_trial.value or 0.0
                    bp_val = bp_trial.value or 0.0
                    diff = eq_val - bp_val
                    
                    winner = "🔋 EqProp" if diff > 0 else "⚡ BP"
                    console.print(f"  Result: EqProp={eq_val:.1%} | BP={bp_val:.1%}")
                    console.print(f"  → {winner} ({diff:+.1%})")
                    
                    # Store results
                    self.results.append(Result("eqprop", task, eq_val, 0.0, eq_trial.params))
                    self.results.append(Result("bp", task, bp_val, 0.0, bp_trial.params))
                else:
                    console.print("[red]Trial failed or incomplete[/red]")

            # Show insights
            if len(self.results) >= 4:
                self._show_insights()
        
        self._final_report()
    
    def _show_insights(self):
        """Show current insights based on Best Found So Far."""
        # Group by task
        tasks = set(r.task for r in self.results)
        
        insights = []
        for task in tasks:
            eq_best = max([r.accuracy for r in self.results if r.algorithm == "eqprop" and r.task == task], default=0)
            bp_best = max([r.accuracy for r in self.results if r.algorithm == "bp" and r.task == task], default=0)
            
            if eq_best > bp_best:
                insights.append(f"[{task}] ✅ EqProp leading: {eq_best:.1%} vs {bp_best:.1%}")
            elif bp_best > eq_best:
                insights.append(f"[{task}] ⚠️ BP leading: {bp_best:.1%} vs {eq_best:.1%}")
            
        if insights:
            console.print(Panel("\n".join(insights), title="Validation Insights", border_style="cyan"))
    
    def _final_report(self):
        """Final comprehensive report."""
        console.print("\n" + "="*60)
        console.print("[bold]FINAL REPORT[/bold]")
        console.print("="*60)
        
        # Scoreboard
        table = Table(title="Best Results by Task")
        table.add_column("Task")
        table.add_column("EqProp Best", style="green")
        table.add_column("BP Best", style="yellow")
        table.add_column("Winner")
        
        tasks = sorted(set(r.task for r in self.results))
        eq_wins = 0
        bp_wins = 0
        
        for task in tasks:
            eq_best_r = max([r for r in self.results if r.algorithm == "eqprop" and r.task == task],
                         key=lambda x: x.accuracy, default=None)
            bp_best_r = max([r for r in self.results if r.algorithm == "bp" and r.task == task],
                         key=lambda x: x.accuracy, default=None)
            
            eq_s = f"{eq_best_r.accuracy:.3f}" if eq_best_r else "-"
            bp_s = f"{bp_best_r.accuracy:.3f}" if bp_best_r else "-"
            
            winner = "-"
            if eq_best_r and bp_best_r:
                if eq_best_r.accuracy > bp_best_r.accuracy:
                    winner = "🔋 EqProp"
                    eq_wins += 1
                elif bp_best_r.accuracy > eq_best_r.accuracy:
                    winner = "⚡ BP"
                    bp_wins += 1
                else:
                    winner = "Tie"
            
            table.add_row(task, eq_s, bp_s, winner)
        
        console.print(table)
        console.print(f"\n[bold]Score: EqProp {eq_wins}, BP {bp_wins}[/bold]")
        
        # Best Configs
        console.print("\n[bold cyan]Best Configurations:[/bold cyan]")
        for task in tasks:
            eq_best_r = max([r for r in self.results if r.algorithm == "eqprop" and r.task == task],
                         key=lambda x: x.accuracy, default=None)
            if eq_best_r:
                console.print(f"[green]{task} (EqProp)[/green]: {eq_best_r.config}")


def main():
    parser = argparse.ArgumentParser(description="TorEqProp Research")
    parser.add_argument("minutes", type=float, nargs="?", default=3)
    parser.add_argument("--deep", action="store_true", help="Deep exploration")
    args = parser.parse_args()
    
    breadth = "high" if args.deep else "low" if args.minutes < 5 else "normal"
    
    console.print("""[bold magenta]
╔════════════════════════════════════════════════════╗
║  🔬 TorEqProp vs Baselines: Scientific Analysis   ║
╚════════════════════════════════════════════════════╝[/bold magenta]
""")
    
    toreq = TorEq(breadth=breadth)
    toreq.run(minutes=args.minutes)


if __name__ == "__main__":
    main()
