#!/usr/bin/env python3
"""
TUI Dashboard for Validation Engine.

Provides real-time visualization of:
- Overall progress
- Current experiment status
- Validated results table
- Fairness indicators

Uses the 'rich' library for beautiful terminal output.
"""

import time
from datetime import datetime
from typing import Dict, Optional, List
from pathlib import Path

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.live import Live
    from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
    from rich.text import Text
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Warning: 'rich' library not installed. Install with: pip install rich")


class ValidationDashboard:
    """Rich TUI dashboard for validation progress."""
    
    def __init__(self):
        if not RICH_AVAILABLE:
            raise ImportError("rich library required for dashboard")
        
        self.console = Console()
        self.start_time = datetime.now()
        
        # State
        self.current_experiment: Optional[Dict] = None
        self.current_output: List[str] = []
        self.max_output_lines = 8
        self.results: Dict = {}
        self.progress: Dict = {}
        self.validated: Dict = {}
    
    def update_experiment(self, exp_config: Dict):
        """Update current experiment being run."""
        self.current_experiment = exp_config
        self.current_output = []
    
    def add_output_line(self, line: str):
        """Add output line from current experiment."""
        self.current_output.append(line.rstrip())
        if len(self.current_output) > self.max_output_lines:
            self.current_output = self.current_output[-self.max_output_lines:]
    
    def update_progress(self, progress: Dict):
        """Update overall progress."""
        self.progress = progress
    
    def update_validated(self, validated: Dict):
        """Update validated results."""
        self.validated = validated
    
    def _create_header(self) -> Panel:
        """Create header panel."""
        elapsed = datetime.now() - self.start_time
        elapsed_str = str(elapsed).split('.')[0]
        
        header_text = Text()
        header_text.append("TorEqProp Validation Engine v1.0", style="bold cyan")
        header_text.append(f"\n\nStarted: {self.start_time.strftime('%H:%M:%S')}")
        header_text.append(f"  │  Elapsed: {elapsed_str}")
        header_text.append(f"  │  Status: ", style="dim")
        header_text.append("RUNNING", style="bold green")
        
        return Panel(header_text, title="[bold white]Dashboard[/]", border_style="blue")
    
    def _create_progress_panel(self) -> Panel:
        """Create progress panel."""
        if not self.progress:
            return Panel("Loading...", title="Progress")
        
        content = Text()
        
        overall = self.progress.get("overall_progress", 0)
        completed = self.progress.get("completed", 0)
        total = self.progress.get("total", 0)
        
        # Overall progress bar
        bar_width = 30
        filled = int(overall * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        content.append(f"Overall: [{bar}] {overall:.0%}\n", style="bold")
        content.append(f"         {completed}/{total} experiments\n\n")
        
        # Per-environment progress
        for env, p in self.progress.get("environments", {}).items():
            pct = p["completed"] / p["total"] if p["total"] > 0 else 0
            filled = int(pct * 20)
            bar = "█" * filled + "░" * (20 - filled)
            
            status = "✅" if pct == 1.0 else "🔄" if pct > 0 else "⏳"
            content.append(f"{status} {env:18} [{bar}] {p['completed']}/{p['total']}\n")
        
        return Panel(content, title="[bold white]Progress[/]", border_style="green")
    
    def _create_current_panel(self) -> Panel:
        """Create current experiment panel."""
        if not self.current_experiment:
            return Panel("Waiting for next experiment...", title="Current Experiment")
        
        exp = self.current_experiment
        content = Text()
        content.append(f"Environment: ", style="dim")
        content.append(f"{exp['environment']}\n", style="bold cyan")
        content.append(f"Algorithm:   ", style="dim")
        content.append(f"{exp['algorithm'].upper()}\n", style="bold yellow")
        content.append(f"Seed:        ", style="dim")
        content.append(f"{exp['seed']}\n\n", style="bold")
        
        # Latest output
        content.append("─" * 40 + "\n", style="dim")
        for line in self.current_output[-self.max_output_lines:]:
            # Truncate long lines
            display_line = line[:60] + "..." if len(line) > 60 else line
            content.append(f"{display_line}\n", style="white")
        
        return Panel(content, title="[bold white]Current Experiment[/]", border_style="yellow")
    
    def _create_results_table(self) -> Table:
        """Create validated results table."""
        table = Table(title="Validated Results", box=box.ROUNDED)
        
        table.add_column("Environment", style="cyan")
        table.add_column("EqProp", justify="right")
        table.add_column("BP", justify="right")
        table.add_column("Δ", justify="right")
        table.add_column("Status", justify="center")
        
        for env, result in sorted(self.validated.items()):
            improvement = f"+{result.improvement_pct:.0f}%{result.significance_stars}"
            
            if result.is_breakthrough:
                status = "[bold green]✅ VALIDATED[/]"
            elif result.is_significant:
                status = "[green]✅ Significant[/]"
            else:
                status = "[yellow]⏳ Pending[/]"
            
            table.add_row(
                env,
                f"{result.algo1_mean:.0f}±{result.algo1_std:.0f}",
                f"{result.algo2_mean:.0f}±{result.algo2_std:.0f}",
                improvement,
                status
            )
        
        if not self.validated:
            table.add_row("", "[dim]Collecting data...[/]", "", "", "")
        
        return table
    
    def render(self) -> Layout:
        """Render complete dashboard layout."""
        layout = Layout()
        
        layout.split_column(
            Layout(self._create_header(), size=5),
            Layout(name="main")
        )
        
        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        
        layout["left"].split_column(
            Layout(self._create_progress_panel(), size=12),
            Layout(self._create_current_panel())
        )
        
        layout["right"].update(Panel(self._create_results_table(), title="[bold white]Results[/]"))
        
        return layout
    
    def print_static(self):
        """Print a static snapshot of the dashboard."""
        self.console.print(self._create_header())
        self.console.print(self._create_progress_panel())
        if self.current_experiment:
            self.console.print(self._create_current_panel())
        self.console.print(self._create_results_table())


class SimpleDashboard:
    """Fallback simple dashboard without rich."""
    
    def __init__(self):
        self.current_experiment = None
        self.progress = {}
        self.validated = {}
    
    def update_experiment(self, exp_config: Dict):
        self.current_experiment = exp_config
        print(f"\n{'='*60}")
        print(f"🚀 Running: {exp_config['experiment_id']}")
        print(f"   Env: {exp_config['environment']}, Algo: {exp_config['algorithm']}, Seed: {exp_config['seed']}")
    
    def add_output_line(self, line: str):
        print(line, end="")
    
    def update_progress(self, progress: Dict):
        self.progress = progress
        pct = progress.get("overall_progress", 0) * 100
        completed = progress.get("completed", 0)
        total = progress.get("total", 0)
        print(f"\n📊 Progress: {completed}/{total} ({pct:.0f}%)")
    
    def update_validated(self, validated: Dict):
        self.validated = validated
        if validated:
            print("\n📈 Validated Results:")
            for env, result in validated.items():
                print(f"   {env}: +{result.improvement_pct:.0f}% {'***' if result.is_breakthrough else ''}")
    
    def print_static(self):
        print(f"\n{'='*60}")
        print("Validation Status")
        print(f"{'='*60}")
        if self.progress:
            pct = self.progress.get("overall_progress", 0) * 100
            print(f"Progress: {pct:.0f}%")


def create_dashboard(use_rich: bool = True):
    """Factory function to create appropriate dashboard."""
    if use_rich and RICH_AVAILABLE:
        return ValidationDashboard()
    return SimpleDashboard()


# Self-test
if __name__ == "__main__":
    print("Testing Dashboard...")
    
    if RICH_AVAILABLE:
        dashboard = ValidationDashboard()
        
        # Mock data
        dashboard.update_progress({
            "overall_progress": 0.45,
            "completed": 36,
            "total": 80,
            "environments": {
                "CartPole-v1": {"completed": 20, "total": 20},
                "Acrobot-v1": {"completed": 16, "total": 20},
                "MountainCar-v0": {"completed": 0, "total": 20},
                "LunarLander-v2": {"completed": 0, "total": 20}
            }
        })
        
        dashboard.update_experiment({
            "experiment_id": "acrobot_v1_eqprop_seed7",
            "environment": "Acrobot-v1",
            "algorithm": "eqprop",
            "seed": 7
        })
        
        dashboard.add_output_line("Episode 50: Reward=-89, Avg(100)=-123.4")
        dashboard.add_output_line("Episode 60: Reward=-78, Avg(100)=-118.2")
        
        dashboard.print_static()
        print("\n✅ Dashboard test complete!")
    else:
        print("⚠️ Rich not available, testing simple dashboard")
        dashboard = SimpleDashboard()
        dashboard.update_progress({"overall_progress": 0.5, "completed": 40, "total": 80})
        print("✅ Simple dashboard test complete!")
