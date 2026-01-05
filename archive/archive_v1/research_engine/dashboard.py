"""
Live TUI dashboard for the research engine.

Provides real-time visualization of:
- Current experiment parameters (full transparency)
- Progress across tiers
- Scoreboard (EqProp vs BP)
- Recent results and discoveries
"""

import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from pathlib import Path

# Rich library for TUI
try:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
    from rich.text import Text
    from rich.align import Align
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    print("⚠️ Rich library not available. Install with: pip install rich")

from .collector import ResultCollector, Trial
from .config import ResearchConfig, TIERS, TIER_ORDER, DEFAULT_CONFIG


@dataclass
class DashboardState:
    """Current state for dashboard display."""
    # Experiment state
    current_task: str = ""
    current_algorithm: str = ""
    current_config: Dict[str, Any] = field(default_factory=dict)
    current_status: str = "Initializing..."
    
    # Statistics
    experiments_run: int = 0
    experiments_success: int = 0
    experiments_failed: int = 0
    experiments_timeout: int = 0
    
    eqprop_wins: int = 0
    bp_wins: int = 0
    ties: int = 0
    
    best_eqprop: Dict[str, float] = field(default_factory=dict)
    best_bp: Dict[str, float] = field(default_factory=dict)
    
    # Recent activity
    recent_results: List[str] = field(default_factory=list)
    discoveries: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Timing
    start_time: float = field(default_factory=time.time)
    deadline: float = 0.0
    
    # Tier progress
    tier_progress: Dict[str, Dict[str, int]] = field(default_factory=dict)


class ResearchDashboard:
    """Live TUI dashboard with full parameter transparency."""
    
    def __init__(
        self,
        collector: Optional[ResultCollector] = None,
        config: ResearchConfig = DEFAULT_CONFIG,
    ):
        if not HAS_RICH:
            raise ImportError("Rich library required for dashboard. Install with: pip install rich")
        
        self.collector = collector or ResultCollector(config.output_dir)
        self.config = config
        self.console = Console()
        self.state = DashboardState()
        self.refresh_rate = config.dashboard_refresh_rate
        
        self._live: Optional[Live] = None
    
    def update_experiment(
        self,
        task: str,
        algorithm: str,
        config: Dict[str, Any],
        status: str = "Running...",
    ):
        """Update current experiment being run."""
        self.state.current_task = task
        self.state.current_algorithm = algorithm
        self.state.current_config = config
        self.state.current_status = status
    
    def record_result(self, trial: Trial):
        """Record a completed trial result."""
        self.state.experiments_run += 1
        
        if trial.status == "complete":
            self.state.experiments_success += 1
            
            # Update bests
            if trial.algorithm == "eqprop":
                current_best = self.state.best_eqprop.get(trial.task, 0)
                if trial.performance > current_best:
                    self.state.best_eqprop[trial.task] = trial.performance
            else:
                current_best = self.state.best_bp.get(trial.task, 0)
                if trial.performance > current_best:
                    self.state.best_bp[trial.task] = trial.performance
            
            # Add to recent
            icon = "🔋" if trial.algorithm == "eqprop" else "⚡"
            result_str = f"{icon} {trial.task}: {trial.performance:.3f} ({trial.cost.wall_time_seconds:.1f}s)"
            self.state.recent_results.append(result_str)
            if len(self.state.recent_results) > 10:
                self.state.recent_results.pop(0)
            
            # Update wins
            self._update_wins()
            
        elif trial.status == "failed":
            self.state.experiments_failed += 1
        elif trial.status == "timeout":
            self.state.experiments_timeout += 1
    
    def add_discovery(self, discovery: str):
        """Add a significant discovery."""
        if discovery not in self.state.discoveries:
            self.state.discoveries.append(discovery)
            if len(self.state.discoveries) > 5:
                self.state.discoveries.pop(0)
    
    def add_warning(self, warning: str):
        """Add a warning message."""
        self.state.warnings.append(warning)
        if len(self.state.warnings) > 3:
            self.state.warnings.pop(0)
    
    def _update_wins(self):
        """Update win counts from best scores."""
        self.state.eqprop_wins = 0
        self.state.bp_wins = 0
        self.state.ties = 0
        
        all_tasks = set(self.state.best_eqprop.keys()) | set(self.state.best_bp.keys())
        for task in all_tasks:
            eq = self.state.best_eqprop.get(task, 0)
            bp = self.state.best_bp.get(task, 0)
            if eq > bp:
                self.state.eqprop_wins += 1
            elif bp > eq:
                self.state.bp_wins += 1
            else:
                self.state.ties += 1
    
    def _make_header(self) -> Panel:
        """Create animated header panel."""
        elapsed = time.time() - self.state.start_time
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        
        # Remaining time
        if self.state.deadline > 0:
            remaining = max(0, self.state.deadline - time.time())
            rem_mins = int(remaining // 60)
            rem_secs = int(remaining % 60)
            time_str = f"⏱️ {mins:02d}:{secs:02d} | ⏳ {rem_mins:02d}:{rem_secs:02d} remaining"
        else:
            time_str = f"⏱️ {mins:02d}:{secs:02d}"
        
        # Animated icon
        frames = ["🔬", "🧪", "⚗️", "🔭"]
        frame = frames[int(elapsed) % len(frames)]
        
        title = Text()
        title.append(f" {frame} ", style="bold white on blue")
        title.append(" TorEqProp RESEARCH ENGINE ", style="bold white on magenta")
        title.append(f" {frame} ", style="bold white on blue")
        
        status = f"{time_str} | 🧪 {self.state.experiments_run} experiments | "
        status += f"✅ {self.state.experiments_success} | ❌ {self.state.experiments_failed}"
        if self.state.experiments_timeout > 0:
            status += f" | ⏰ {self.state.experiments_timeout}"
        
        return Panel(
            Align.center(title),
            subtitle=status,
            border_style="bright_magenta",
            padding=(0, 2),
        )
    
    def _make_parameters_panel(self) -> Panel:
        """Create current parameters panel (full transparency)."""
        table = Table(show_header=True, header_style="bold cyan", box=box.SIMPLE)
        table.add_column("Parameter", style="dim")
        table.add_column("Value", style="bold")
        
        if self.state.current_config:
            for key, value in sorted(self.state.current_config.items()):
                if not key.startswith("_"):
                    val_str = f"{value:.4g}" if isinstance(value, float) else str(value)
                    table.add_row(key, val_str)
        else:
            table.add_row("(waiting)", "-")
        
        return Panel(
            table,
            title=f"⚙️ {self.state.current_algorithm.upper() if self.state.current_algorithm else 'CONFIG'}",
            border_style="cyan",
        )
    
    def _make_scoreboard(self) -> Panel:
        """Create EqProp vs BP scoreboard."""
        table = Table(show_header=True, header_style="bold", box=box.SIMPLE)
        table.add_column("", justify="center", width=15)
        table.add_column("🔋 EqProp", justify="center", style="green", width=12)
        table.add_column("⚡ BP", justify="center", style="yellow", width=12)
        
        table.add_row(
            "🏆 Wins",
            str(self.state.eqprop_wins),
            str(self.state.bp_wins),
        )
        
        # Best per task
        all_tasks = sorted(set(self.state.best_eqprop.keys()) | set(self.state.best_bp.keys()))
        for task in all_tasks[:5]:
            eq = self.state.best_eqprop.get(task, 0)
            bp = self.state.best_bp.get(task, 0)
            
            eq_style = "bold green" if eq > bp else "dim"
            bp_style = "bold yellow" if bp > eq else "dim"
            
            table.add_row(
                f"📊 {task}",
                Text(f"{eq:.3f}", style=eq_style) if eq > 0 else Text("-", style="dim"),
                Text(f"{bp:.3f}", style=bp_style) if bp > 0 else Text("-", style="dim"),
            )
        
        return Panel(
            table,
            title="⚔️ SCOREBOARD",
            border_style="yellow",
        )
    
    def _make_activity_panel(self) -> Panel:
        """Create current activity panel."""
        lines = []
        
        # Current experiment
        if self.state.current_task:
            algo_icon = "🔋" if self.state.current_algorithm == "eqprop" else "⚡"
            lines.append(Text(f"{algo_icon} {self.state.current_task}", style="bold"))
            lines.append(Text(f"   {self.state.current_status}", style="dim"))
        
        # Recent results
        lines.append(Text("\n📊 Recent:", style="bold cyan"))
        for result in self.state.recent_results[-5:]:
            lines.append(Text(f"   {result}"))
        
        # Warnings
        if self.state.warnings:
            lines.append(Text("\n⚠️ Warnings:", style="bold yellow"))
            for warning in self.state.warnings[-2:]:
                lines.append(Text(f"   {warning}", style="yellow"))
        
        content = "\n".join(str(line) for line in lines) if lines else "Starting..."
        
        return Panel(
            content,
            title="⚡ ACTIVITY",
            border_style="green",
        )
    
    def _make_discoveries_panel(self) -> Panel:
        """Create discoveries panel."""
        if not self.state.discoveries:
            content = Text("🔍 Running experiments...\n   Findings will appear here!", style="dim italic")
        else:
            content = "\n".join(f"🎯 {d}" for d in self.state.discoveries[-5:])
        
        return Panel(
            content,
            title="🔬 DISCOVERIES",
            border_style="magenta",
        )
    
    def _make_progress_panel(self) -> Panel:
        """Create progress visualization."""
        lines = []
        
        # Overall progress
        elapsed = time.time() - self.state.start_time
        if elapsed > 0:
            rate = self.state.experiments_run / (elapsed / 60)
            lines.append(f"🚀 Speed: {rate:.1f} experiments/min")
        
        # Success rate
        if self.state.experiments_run > 0:
            success_rate = self.state.experiments_success / self.state.experiments_run * 100
            lines.append(f"✅ Success rate: {success_rate:.0f}%")
        
        # EqProp vs BP bar
        total_wins = self.state.eqprop_wins + self.state.bp_wins + 1
        eq_pct = int(self.state.eqprop_wins / total_wins * 20)
        bp_pct = int(self.state.bp_wins / total_wins * 20)
        neutral = 20 - eq_pct - bp_pct
        versus_bar = "🟢" * eq_pct + "⚪" * neutral + "🟡" * bp_pct
        lines.append(f"⚔️ EqProp vs BP: [{versus_bar}]")
        
        # Tier progress
        if self.state.tier_progress:
            lines.append("\n📈 Tier Progress:")
            for tier_name in TIER_ORDER:
                if tier_name in self.state.tier_progress:
                    prog = self.state.tier_progress[tier_name]
                    pct = prog.get("percent", 0)
                    bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
                    lines.append(f"   {tier_name}: [{bar}] {pct}%")
        
        return Panel(
            "\n".join(lines) if lines else "Starting...",
            title="📊 METRICS",
            border_style="blue",
        )
    
    def render(self) -> Layout:
        """Build the full dashboard layout."""
        layout = Layout()
        
        layout.split(
            Layout(name="header", size=4),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=8),
        )
        
        layout["main"].split_row(
            Layout(name="left", ratio=1),
            Layout(name="center", ratio=1),
            Layout(name="right", ratio=1),
        )
        
        layout["header"].update(self._make_header())
        layout["left"].update(self._make_parameters_panel())
        layout["center"].update(self._make_scoreboard())
        layout["right"].split(
            Layout(self._make_activity_panel()),
            Layout(self._make_discoveries_panel()),
        )
        layout["footer"].update(self._make_progress_panel())
        
        return layout
    
    def start(self, deadline: float = 0.0):
        """Start live dashboard."""
        self.state.start_time = time.time()
        self.state.deadline = deadline
        self._live = Live(
            self.render(),
            refresh_per_second=1 / self.refresh_rate,
            console=self.console,
        )
        self._live.start()
    
    def update(self):
        """Update dashboard display."""
        if self._live:
            self._live.update(self.render())
    
    def stop(self):
        """Stop live dashboard."""
        if self._live:
            self._live.stop()
            self._live = None
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
    
    def print_summary(self):
        """Print final summary after dashboard stops."""
        elapsed = time.time() - self.state.start_time
        
        self.console.print("\n")
        self.console.print(Panel(
            f"""
[bold green]🎉 RESEARCH COMPLETE![/bold green]

⏱️  Time: {elapsed/60:.1f} minutes
🧪  Experiments: {self.state.experiments_run}
✅  Success rate: {self.state.experiments_success / max(1, self.state.experiments_run) * 100:.0f}%

[bold cyan]SCOREBOARD:[/bold cyan]
   🔋 EqProp wins: {self.state.eqprop_wins}
   ⚡ BP wins: {self.state.bp_wins}

[bold green]DISCOVERIES:[/bold green]
{chr(10).join("   🎯 " + d for d in self.state.discoveries) or "   (Run longer for findings)"}
            """,
            title="📊 FINAL RESULTS",
            border_style="green",
        ))


class SimpleDashboard:
    """Fallback dashboard without Rich (simple print statements)."""
    
    def __init__(self, collector=None, config=None):
        self.state = DashboardState()
        self.collector = collector
    
    def update_experiment(self, task, algorithm, config, status="Running..."):
        self.state.current_task = task
        self.state.current_algorithm = algorithm
        self.state.current_config = config
        print(f"[{algorithm}] {task}: {status}")
    
    def record_result(self, trial):
        self.state.experiments_run += 1
        if trial.status == "complete":
            self.state.experiments_success += 1
            print(f"  ✅ {trial.performance:.4f} in {trial.cost.wall_time_seconds:.1f}s")
        else:
            self.state.experiments_failed += 1
            print(f"  ❌ {trial.status}: {trial.error}")
    
    def add_discovery(self, discovery):
        print(f"🎯 Discovery: {discovery}")
    
    def add_warning(self, warning):
        print(f"⚠️ {warning}")
    
    def start(self, deadline=0.0):
        print("=" * 60)
        print("TorEqProp Research Engine (Simple Mode)")
        print("=" * 60)
    
    def update(self):
        pass
    
    def stop(self):
        pass
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.stop()
    
    def print_summary(self):
        print("\n" + "=" * 60)
        print("RESEARCH COMPLETE")
        print(f"Experiments: {self.state.experiments_run}")
        print(f"Success: {self.state.experiments_success}")
        print("=" * 60)


def create_dashboard(use_rich: bool = True, **kwargs):
    """Factory function to create appropriate dashboard."""
    if use_rich and HAS_RICH:
        return ResearchDashboard(**kwargs)
    return SimpleDashboard(**kwargs)
