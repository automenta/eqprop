#!/usr/bin/env python3
"""
TorEqProp Rapid Research Dashboard

A fast, visually exciting TUI dashboard for autonomous research.
Gets tangible results in MINUTES with rich live feedback.

Features:
- Live updating dashboard with progress bars
- Real-time statistics and findings
- Fast micro-experiments for quick validation
- Colorful, engaging display

Usage:
    python rapid_dashboard.py              # Start dashboard
    python rapid_dashboard.py --turbo      # Ultra-fast mode
"""

import argparse
import random
import sys
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any

# Check for rich library
try:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
    from rich.text import Text
    from rich.align import Align
    from rich.style import Style
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    print("Installing rich library for TUI...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "rich", "-q"])
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
    from rich.text import Text
    from rich.align import Align
    from rich.style import Style

import numpy as np

# Import core components
from hyperopt_engine import (
    HyperOptEngine, HyperOptTrial, EqPropSearchSpace, BaselineSearchSpace
)
from statistics import StatisticalAnalyzer


@dataclass
class LiveStats:
    """Real-time statistics for dashboard."""
    experiments_run: int = 0
    experiments_success: int = 0
    experiments_failed: int = 0
    
    eqprop_wins: int = 0
    bp_wins: int = 0
    ties: int = 0
    
    best_eqprop: Dict[str, float] = field(default_factory=dict)
    best_bp: Dict[str, float] = field(default_factory=dict)
    
    current_task: str = ""
    current_algo: str = ""
    current_status: str = "Starting..."
    
    findings: List[str] = field(default_factory=list)
    recent_results: List[str] = field(default_factory=list)
    
    start_time: float = field(default_factory=time.time)


class RapidDashboard:
    """Fast, visual research dashboard."""
    
    # Fast experiment configurations
    RAPID_TASKS = [
        ("xor", 1),      # task, epochs
        ("xor3", 1),
        ("mnist", 1),
        ("cartpole", 1),
    ]
    
    def __init__(self, turbo_mode: bool = False):
        self.console = Console()
        self.engine = HyperOptEngine()
        self.stats = StatisticalAnalyzer()
        self.live_stats = LiveStats()
        self.turbo = turbo_mode
        self.running = True
        
        # Reduce model size for speed
        self.engine.eqprop_space = EqPropSearchSpace(
            beta=[0.2, 0.22, 0.25],
            damping=[0.8, 0.9],
            max_iters=[10, 20],
            tol=[1e-4],
            attention_type=["linear"],
            symmetric=[False],
            update_mode=["mse_proxy"],
            d_model=[32, 64] if turbo_mode else [64, 128],
            lr=[1e-3, 2e-3],
        )
        self.engine.baseline_space = BaselineSearchSpace(
            lr=[1e-3, 2e-3, 5e-3],
            optimizer=["adam"],
            d_model=[32, 64] if turbo_mode else [64, 128],
            weight_decay=[0, 1e-4],
            scheduler=["none"],
        )
    
    def make_header(self) -> Panel:
        """Create animated header."""
        elapsed = time.time() - self.live_stats.start_time
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        
        # Animated elements
        frames = ["🔬", "🧪", "⚗️", "🔭"]
        frame = frames[int(elapsed) % len(frames)]
        
        title = Text()
        title.append(f" {frame} ", style="bold white on blue")
        title.append(" TorEqProp RAPID RESEARCH ", style="bold white on magenta")
        title.append(f" {frame} ", style="bold white on blue")
        
        status = f"⏱️ {mins:02d}:{secs:02d} | "
        status += f"🧪 {self.live_stats.experiments_run} experiments | "
        status += f"✅ {self.live_stats.experiments_success} success | "
        status += f"❌ {self.live_stats.experiments_failed} failed"
        
        content = Align.center(title)
        
        return Panel(
            content,
            subtitle=status,
            border_style="bright_magenta",
            padding=(0, 2)
        )
    
    def make_scoreboard(self) -> Panel:
        """Create EqProp vs BP scoreboard."""
        table = Table(show_header=True, header_style="bold cyan", box=None)
        table.add_column("", justify="center", width=20)
        table.add_column("🔋 EqProp", justify="center", style="green", width=15)
        table.add_column("⚡ BP", justify="center", style="yellow", width=15)
        
        table.add_row(
            "🏆 Wins",
            str(self.live_stats.eqprop_wins),
            str(self.live_stats.bp_wins)
        )
        
        # Best scores per task
        all_tasks = set(self.live_stats.best_eqprop.keys()) | set(self.live_stats.best_bp.keys())
        for task in sorted(all_tasks)[:4]:
            eq = self.live_stats.best_eqprop.get(task, 0)
            bp = self.live_stats.best_bp.get(task, 0)
            
            eq_style = "bold green" if eq > bp else "dim"
            bp_style = "bold yellow" if bp > eq else "dim"
            
            table.add_row(
                f"📊 {task}",
                Text(f"{eq:.3f}", style=eq_style) if eq > 0 else "-",
                Text(f"{bp:.3f}", style=bp_style) if bp > 0 else "-"
            )
        
        return Panel(
            table,
            title="⚔️ SCOREBOARD",
            border_style="cyan"
        )
    
    def make_activity(self) -> Panel:
        """Create current activity panel."""
        content = []
        
        # Current experiment
        if self.live_stats.current_task:
            algo_icon = "🔋" if self.live_stats.current_algo == "eqprop" else "⚡"
            content.append(Text(f"{algo_icon} Running: {self.live_stats.current_task}", style="bold"))
            content.append(Text(f"   {self.live_stats.current_status}", style="dim"))
        
        # Recent results
        content.append(Text("\n📊 Recent:", style="bold cyan"))
        for result in self.live_stats.recent_results[-5:]:
            content.append(Text(f"   {result}"))
        
        return Panel(
            "\n".join(str(c) for c in content) if content else "Starting...",
            title="⚡ ACTIVITY",
            border_style="yellow"
        )
    
    def make_findings(self) -> Panel:
        """Create findings panel."""
        if not self.live_stats.findings:
            content = Text("🔍 Running experiments...\n   Findings will appear here!", style="dim italic")
        else:
            content = "\n".join(f"🎯 {f}" for f in self.live_stats.findings[-5:])
        
        return Panel(
            content,
            title="🔬 DISCOVERIES",
            border_style="green"
        )
    
    def make_progress_bars(self) -> Panel:
        """Create animated progress visualization."""
        lines = []
        
        # Overall progress
        total_planned = len(self.RAPID_TASKS) * 2 * 3  # tasks * algos * seeds
        done = self.live_stats.experiments_run
        pct = min(100, int(done / total_planned * 100))
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        lines.append(f"📈 Progress: [{bar}] {pct}%")
        
        # Speed indicator
        elapsed = time.time() - self.live_stats.start_time
        if elapsed > 0:
            rate = self.live_stats.experiments_run / (elapsed / 60)
            lines.append(f"🚀 Speed: {rate:.1f} experiments/min")
        
        # EqProp vs BP bar
        total_wins = self.live_stats.eqprop_wins + self.live_stats.bp_wins + 1
        eq_pct = int(self.live_stats.eqprop_wins / total_wins * 20)
        bp_pct = int(self.live_stats.bp_wins / total_wins * 20)
        versus_bar = "🟢" * eq_pct + "⚪" * (20 - eq_pct - bp_pct) + "🟡" * bp_pct
        lines.append(f"⚔️ EqProp vs BP: [{versus_bar}]")
        
        return Panel(
            "\n".join(lines),
            title="📊 METRICS",
            border_style="magenta"
        )
    
    def make_layout(self) -> Layout:
        """Build the full dashboard layout."""
        layout = Layout()
        
        layout.split(
            Layout(name="header", size=4),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=6)
        )
        
        layout["main"].split_row(
            Layout(name="left", ratio=1),
            Layout(name="right", ratio=1)
        )
        
        layout["header"].update(self.make_header())
        layout["left"].split(
            Layout(self.make_scoreboard(), name="scoreboard"),
            Layout(self.make_findings(), name="findings")
        )
        layout["right"].update(self.make_activity())
        layout["footer"].update(self.make_progress_bars())
        
        return layout
    
    def run_experiment(self, task: str, algorithm: str, epochs: int):
        """Run a single fast experiment."""
        self.live_stats.current_task = task
        self.live_stats.current_algo = algorithm
        self.live_stats.current_status = "Initializing..."
        
        # Sample config
        if algorithm == "eqprop":
            config = self.engine.eqprop_space.sample(random.Random())
        else:
            config = self.engine.baseline_space.sample(random.Random())
        
        trial = HyperOptTrial(
            trial_id=f"rapid_{task}_{algorithm}_{int(time.time()*1000)}",
            algorithm=algorithm,
            config=config,
            task=task,
            seed=random.randint(0, 9999)
        )
        
        self.live_stats.current_status = "Training..."
        
        # Run the trial
        trial = self.engine.evaluator.evaluate(trial, epochs=epochs, show_progress=False)
        
        self.live_stats.experiments_run += 1
        
        if trial.status == "complete":
            self.live_stats.experiments_success += 1
            self.engine.db.add_trial(trial)
            
            # Update bests
            icon = "🔋" if algorithm == "eqprop" else "⚡"
            result_str = f"{icon} {task}: {trial.performance:.3f}"
            self.live_stats.recent_results.append(result_str)
            
            if algorithm == "eqprop":
                if trial.performance > self.live_stats.best_eqprop.get(task, 0):
                    self.live_stats.best_eqprop[task] = trial.performance
            else:
                if trial.performance > self.live_stats.best_bp.get(task, 0):
                    self.live_stats.best_bp[task] = trial.performance
            
            # Check for winner update
            self._update_wins()
            
            # Check for findings
            self._check_findings(task)
            
            return True
        else:
            self.live_stats.experiments_failed += 1
            self.live_stats.current_status = f"Failed: {trial.error}"
            return False
    
    def _update_wins(self):
        """Update win counts."""
        self.live_stats.eqprop_wins = 0
        self.live_stats.bp_wins = 0
        self.live_stats.ties = 0
        
        all_tasks = set(self.live_stats.best_eqprop.keys()) | set(self.live_stats.best_bp.keys())
        for task in all_tasks:
            eq = self.live_stats.best_eqprop.get(task, 0)
            bp = self.live_stats.best_bp.get(task, 0)
            if eq > bp:
                self.live_stats.eqprop_wins += 1
            elif bp > eq:
                self.live_stats.bp_wins += 1
            else:
                self.live_stats.ties += 1
    
    def _check_findings(self, task: str):
        """Check for significant findings."""
        eq_trials = self.engine.db.get_trials(algorithm="eqprop", task=task, status="complete")
        bp_trials = self.engine.db.get_trials(algorithm="bp", task=task, status="complete")
        
        if len(eq_trials) >= 2 and len(bp_trials) >= 2:
            eq_perfs = [t.performance for t in eq_trials[-5:]]
            bp_perfs = [t.performance for t in bp_trials[-5:]]
            
            result = self.stats.compare(eq_perfs, bp_perfs, "EqProp", "BP")
            
            if result.is_significant:
                winner = "EqProp" if result.algo1_mean > result.algo2_mean else "BP"
                finding = f"{winner} wins on {task} (p={result.p_value:.3f})"
                if finding not in self.live_stats.findings:
                    self.live_stats.findings.append(finding)
    
    def run(self, max_minutes: float = 5):
        """Run the rapid research dashboard."""
        deadline = time.time() + max_minutes * 60
        
        with Live(self.make_layout(), refresh_per_second=4, console=self.console) as live:
            cycle = 0
            while self.running and time.time() < deadline:
                cycle += 1
                
                for task, epochs in self.RAPID_TASKS:
                    if time.time() >= deadline:
                        break
                    
                    for algo in ["eqprop", "bp"]:
                        if time.time() >= deadline:
                            break
                        
                        try:
                            self.run_experiment(task, algo, epochs)
                        except Exception as e:
                            self.live_stats.experiments_failed += 1
                            self.live_stats.current_status = f"Error: {str(e)[:30]}"
                        
                        # Update display
                        live.update(self.make_layout())
            
            # Final update
            self.live_stats.current_status = "Complete!"
            live.update(self.make_layout())
            time.sleep(1)
        
        # Print summary
        self._print_summary()
    
    def _print_summary(self):
        """Print final summary."""
        self.console.print("\n")
        self.console.print(Panel(
            f"""
[bold green]🎉 RESEARCH COMPLETE![/bold green]

⏱️  Time: {(time.time() - self.live_stats.start_time)/60:.1f} minutes
🧪  Experiments: {self.live_stats.experiments_run}
✅  Success rate: {self.live_stats.experiments_success / max(1, self.live_stats.experiments_run) * 100:.0f}%

[bold cyan]SCOREBOARD:[/bold cyan]
   🔋 EqProp wins: {self.live_stats.eqprop_wins}
   ⚡ BP wins: {self.live_stats.bp_wins}

[bold green]DISCOVERIES:[/bold green]
{chr(10).join("   🎯 " + f for f in self.live_stats.findings) or "   (Run longer for statistically significant findings)"}
            """,
            title="📊 FINAL RESULTS",
            border_style="green"
        ))


def main():
    parser = argparse.ArgumentParser(description="Rapid Research Dashboard")
    parser.add_argument("--turbo", action="store_true", help="Ultra-fast mode (smaller models)")
    parser.add_argument("--minutes", type=float, default=5, help="Minutes to run (default: 5)")
    args = parser.parse_args()
    
    Console().print("""
[bold magenta]
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║     🚀 RAPID RESEARCH DASHBOARD 🚀                                ║
║                                                                   ║
║     Get results in MINUTES, not hours!                            ║
║     Watch discoveries happen in real-time!                        ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
[/bold magenta]
    """)
    
    dashboard = RapidDashboard(turbo_mode=args.turbo)
    dashboard.run(max_minutes=args.minutes)


if __name__ == "__main__":
    main()
