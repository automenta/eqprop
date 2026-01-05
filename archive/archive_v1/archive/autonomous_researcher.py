#!/usr/bin/env python3
"""
TorEqProp Autonomous Research System

Turn-key automation:
    INPUT:  Time + Electricity
    OUTPUT: Beneficial Research

Features:
    - Runs indefinitely, continuously discovering insights
    - Intelligent experiment prioritization
    - Auto-documentation of findings
    - Crash recovery and checkpointing
    - Resource monitoring
    - Progressive result updates to RESULTS.md

Usage:
    python autonomous_researcher.py                     # Run indefinitely
    python autonomous_researcher.py --hours 8           # Run for 8 hours
    python autonomous_researcher.py --until-discovery   # Stop on breakthrough
    python autonomous_researcher.py --resume            # Resume from checkpoint
"""

import argparse
import json
import os
import signal
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
import random
import hashlib

import numpy as np

# Import existing infrastructure
from hyperopt_engine import (
    HyperOptEngine, HyperOptTrial, HyperOptDB, 
    EqPropSearchSpace, BaselineSearchSpace,
    CostAwareEvaluator, ParetoAnalyzer, TrialMatcher
)
from hyperopt.validation import ValidationPipeline, StatisticalVerdict
from statistics import StatisticalAnalyzer


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ResearchFinding:
    """A significant research finding."""
    timestamp: str
    category: str  # "performance", "robustness", "efficiency", "novel"
    title: str
    description: str
    evidence: Dict[str, Any]
    significance: float  # 0-1, how important
    validated: bool = False
    
    def to_markdown(self) -> str:
        icon = {"performance": "🎯", "robustness": "🛡️", 
                "efficiency": "⚡", "novel": "🔬"}.get(self.category, "📊")
        return f"""### {icon} {self.title}
**Category**: {self.category.title()} | **Significance**: {self.significance:.0%}

{self.description}

**Evidence**: p-value={self.evidence.get('p_value', 'N/A')}, effect_size={self.evidence.get('effect_size', 'N/A')}
"""


@dataclass
class ResearchState:
    """Complete state for checkpoint/recovery."""
    started_at: str
    last_update: str
    total_runtime_hours: float = 0.0
    experiments_completed: int = 0
    experiments_failed: int = 0
    findings: List[Dict] = field(default_factory=list)
    best_eqprop_performance: Dict[str, float] = field(default_factory=dict)
    best_baseline_performance: Dict[str, float] = field(default_factory=dict)
    current_phase: str = "exploration"
    prioritization_weights: Dict[str, float] = field(default_factory=dict)
    # Track failed tasks to avoid re-selecting
    failed_task_counts: Dict[str, int] = field(default_factory=dict)
    blocked_tasks: List[str] = field(default_factory=list)
    
    def save(self, path: Path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "ResearchState":
        with open(path) as f:
            data = json.load(f)
        # Handle legacy checkpoints without new fields
        if "failed_task_counts" not in data:
            data["failed_task_counts"] = {}
        if "blocked_tasks" not in data:
            data["blocked_tasks"] = []
        return cls(**data)
    
    def record_failure(self, task: str):
        """Record a task failure. Block task after 3 consecutive failures."""
        self.failed_task_counts[task] = self.failed_task_counts.get(task, 0) + 1
        if self.failed_task_counts[task] >= 3 and task not in self.blocked_tasks:
            self.blocked_tasks.append(task)
    
    def record_success(self, task: str):
        """Reset failure count on success."""
        self.failed_task_counts[task] = 0


@dataclass  
class ExperimentPriority:
    """Priority score for an experiment."""
    task: str
    algorithm: str
    config: Dict[str, Any]
    priority_score: float
    expected_information_gain: float
    risk: float  # probability of failure/crash


# =============================================================================
# INTELLIGENT EXPERIMENT SELECTION
# =============================================================================

class ExperimentSelector:
    """Intelligently selects next experiments based on research state."""
    
    # Task definitions with expected value - ONLY WORKING TASKS
    TASKS = {
        # Quick validation tasks
        "xor": {"category": "micro", "difficulty": 0.1, "time_min": 0.5},
        "xor3": {"category": "micro", "difficulty": 0.2, "time_min": 0.5},
        
        # Core classification (proven to work)
        "mnist": {"category": "classification", "difficulty": 0.5, "time_min": 5},
        "fashion": {"category": "classification", "difficulty": 0.6, "time_min": 5},
        
        # Reinforcement learning (high value target - proven working)
        "cartpole": {"category": "rl", "difficulty": 0.4, "time_min": 3},
        "acrobot": {"category": "rl", "difficulty": 0.6, "time_min": 5},
    }
    
    def __init__(self, state: ResearchState, db: HyperOptDB):
        self.state = state
        self.db = db
        self.rng = random.Random()
    
    def select_next_experiments(self, n: int = 5, 
                                 time_budget_minutes: float = 30) -> List[ExperimentPriority]:
        """Select the next experiments to run based on expected value."""
        candidates = []
        
        for task, info in self.TASKS.items():
            # Skip blocked tasks
            if task in self.state.blocked_tasks:
                continue
            
            # Skip if time exceeds remaining budget
            if info["time_min"] > time_budget_minutes:
                continue
            
            for algorithm in ["eqprop", "bp"]:
                priority = self._compute_priority(task, algorithm, info)
                candidates.append(priority)
        
        # Sort by priority score (highest first)
        candidates.sort(key=lambda x: x.priority_score, reverse=True)
        
        # Add exploration: 20% chance of random selection
        selected = []
        for c in candidates[:n]:
            if self.rng.random() < 0.2:
                # Random exploration
                random_candidate = self.rng.choice(candidates)
                selected.append(random_candidate)
            else:
                selected.append(c)
        
        return selected
    
    def _compute_priority(self, task: str, algorithm: str, 
                          info: Dict) -> ExperimentPriority:
        """Compute priority score for an experiment."""
        
        # Get existing trials for this task/algorithm
        existing = self.db.get_trials(algorithm=algorithm, task=task, status="complete")
        n_existing = len(existing)
        
        # Base priority factors
        novelty = 1.0 / (1 + n_existing * 0.5)  # Less explored = higher priority
        difficulty_bonus = 1.0 - info["difficulty"] * 0.3  # Easier = faster feedback
        
        # Category bonuses based on research state
        category_weights = {
            "rl": 1.5,  # RL shows promise (+88% result)
            "classification": 1.0,
            "algorithmic": 1.2,  # Novel contribution
            "micro": 0.8,  # Quick feedback
        }
        category_bonus = category_weights.get(info["category"], 1.0)
        
        # EqProp gets slight priority (subject of research)
        algo_bonus = 1.2 if algorithm == "eqprop" else 1.0
        
        # Information gain: prioritize underexplored configurations
        # Sample config and check if similar configs exist
        config = self._sample_novel_config(algorithm, existing)
        
        # Final score
        priority_score = novelty * difficulty_bonus * category_bonus * algo_bonus
        
        return ExperimentPriority(
            task=task,
            algorithm=algorithm,
            config=config,
            priority_score=priority_score,
            expected_information_gain=novelty,
            risk=info["difficulty"] * 0.3
        )
    
    def _sample_novel_config(self, algorithm: str, 
                             existing: List[HyperOptTrial]) -> Dict[str, Any]:
        """Sample a configuration that differs from existing trials."""
        if algorithm == "eqprop":
            space = EqPropSearchSpace()
        else:
            space = BaselineSearchSpace()
        
        # Sample several and pick most different from existing
        best_config = None
        best_novelty = 0
        
        for _ in range(10):
            config = space.sample(self.rng)
            novelty = self._config_novelty(config, existing)
            if novelty > best_novelty:
                best_novelty = novelty
                best_config = config
        
        return best_config or space.sample(self.rng)
    
    def _config_novelty(self, config: Dict, existing: List[HyperOptTrial]) -> float:
        """Compute how novel a config is relative to existing trials."""
        if not existing:
            return 1.0
        
        # Compare key hyperparameters
        novelty = 0.0
        for trial in existing:
            diff = 0
            for key in ["d_model", "lr", "beta", "damping"]:
                if key in config and key in trial.config:
                    diff += abs(config[key] - trial.config[key]) / max(config[key], 1)
            novelty += diff / 4
        
        return novelty / len(existing)


# =============================================================================
# AUTONOMOUS RESEARCH RUNNER
# =============================================================================

class AutonomousResearcher:
    """The main autonomous research system."""
    
    def __init__(self, 
                 output_dir: str = "autonomous_research",
                 checkpoint_interval_minutes: float = 10):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_path = self.output_dir / "checkpoint.json"
        self.findings_path = self.output_dir / "findings.md"
        self.log_path = self.output_dir / "researcher.log"
        
        self.checkpoint_interval = checkpoint_interval_minutes * 60
        self.last_checkpoint = time.time()
        
        # Core components
        self.engine = HyperOptEngine()
        self.validation = ValidationPipeline(
            archive_dir=str(self.output_dir / "evidence"),
            report_dir=str(self.output_dir / "reports")
        )
        self.stats = StatisticalAnalyzer()
        
        # State
        self.state = self._load_or_create_state()
        self.selector = ExperimentSelector(self.state, self.engine.db)
        
        # Control
        self.running = True
        self.start_time = time.time()
        
        # Signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
    
    def _load_or_create_state(self) -> ResearchState:
        """Load existing state or create new."""
        if self.checkpoint_path.exists():
            try:
                return ResearchState.load(self.checkpoint_path)
            except Exception as e:
                self.log(f"Failed to load checkpoint: {e}")
        
        return ResearchState(
            started_at=datetime.now().isoformat(),
            last_update=datetime.now().isoformat()
        )
    
    def _handle_shutdown(self, signum, frame):
        """Handle graceful shutdown."""
        self.log("\n🛑 Shutdown signal received, saving state...")
        self.running = False
        self._checkpoint()
        self.log("✅ State saved. Exiting gracefully.")
        sys.exit(0)
    
    def log(self, message: str):
        """Log message to file and console."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        print(line, flush=True)
        
        with open(self.log_path, "a") as f:
            f.write(line + "\n")
    
    def run(self, 
            max_hours: Optional[float] = None,
            stop_on_discovery: bool = False):
        """Main research loop."""
        
        self.log("=" * 70)
        self.log("🔬 AUTONOMOUS RESEARCH SYSTEM STARTED")
        self.log("=" * 70)
        self.log(f"Output directory: {self.output_dir}")
        self.log(f"Max runtime: {max_hours or 'indefinite'} hours")
        self.log(f"Previous runtime: {self.state.total_runtime_hours:.2f} hours")
        self.log(f"Previous experiments: {self.state.experiments_completed}")
        self.log("=" * 70)
        
        deadline = None
        if max_hours:
            deadline = time.time() + max_hours * 3600
        
        cycle = 0
        while self.running:
            cycle += 1
            
            # Check time limit
            if deadline and time.time() > deadline:
                self.log("⏰ Time limit reached.")
                break
            
            # Update runtime
            self.state.total_runtime_hours = (time.time() - self.start_time) / 3600 + \
                                              self.state.total_runtime_hours
            
            self.log(f"\n{'='*70}")
            self.log(f"🔄 Research Cycle {cycle}")
            self.log(f"   Runtime: {self.state.total_runtime_hours:.2f}h")
            self.log(f"   Experiments: {self.state.experiments_completed}")
            self.log(f"   Findings: {len(self.state.findings)}")
            self.log("=" * 70)
            
            try:
                # Select next experiments
                priorities = self.selector.select_next_experiments(n=3)
                
                for priority in priorities:
                    if not self.running:
                        break
                    
                    self.log(f"\n🧪 Running: {priority.task} ({priority.algorithm})")
                    self.log(f"   Priority: {priority.priority_score:.2f}")
                    
                    # Run experiment
                    finding = self._run_experiment(priority)
                    
                    if finding:
                        self.state.findings.append(asdict(finding))
                        self._update_findings_doc()
                        
                        if stop_on_discovery and finding.significance > 0.8:
                            self.log("🎉 BREAKTHROUGH! Stopping as requested.")
                            self.running = False
                            break
                    
                    # Checkpoint periodically
                    if time.time() - self.last_checkpoint > self.checkpoint_interval:
                        self._checkpoint()
                
                # Periodic analysis
                if cycle % 5 == 0:
                    self._run_analysis_cycle()
                
                # Update documentation
                if cycle % 10 == 0:
                    self._update_results_md()
                
            except Exception as e:
                self.log(f"❌ Error in cycle {cycle}: {e}")
                self.log(traceback.format_exc())
                self.state.experiments_failed += 1
                time.sleep(5)  # Brief pause before retry
        
        # Final save
        self._checkpoint()
        self._generate_final_report()
        
        self.log("\n" + "=" * 70)
        self.log("🏁 AUTONOMOUS RESEARCH COMPLETE")
        self.log(f"   Total runtime: {self.state.total_runtime_hours:.2f} hours")
        self.log(f"   Experiments: {self.state.experiments_completed}")
        self.log(f"   Findings: {len(self.state.findings)}")
        self.log("=" * 70)
    
    def _run_experiment(self, priority: ExperimentPriority) -> Optional[ResearchFinding]:
        """Run a single experiment and check for findings."""
        
        trial = HyperOptTrial(
            trial_id=f"auto_{priority.task}_{priority.algorithm}_{int(time.time())}",
            algorithm=priority.algorithm,
            config=priority.config,
            task=priority.task,
            seed=random.randint(0, 9999)
        )
        
        # Run via evaluator
        trial = self.engine.evaluator.evaluate(trial, epochs=3, show_progress=True)
        
        if trial.status == "complete":
            self.state.experiments_completed += 1
            self.state.record_success(priority.task)
            self.engine.db.add_trial(trial)
            
            # Update best performance tracking
            task_key = priority.task
            if priority.algorithm == "eqprop":
                current_best = self.state.best_eqprop_performance.get(task_key, 0)
                if trial.performance > current_best:
                    self.state.best_eqprop_performance[task_key] = trial.performance
            else:
                current_best = self.state.best_baseline_performance.get(task_key, 0)
                if trial.performance > current_best:
                    self.state.best_baseline_performance[task_key] = trial.performance
            
            # Check for significant finding
            return self._check_for_finding(priority.task)
        else:
            self.state.experiments_failed += 1
            self.state.record_failure(priority.task)
            if priority.task in self.state.blocked_tasks:
                self.log(f"   ⛔ Task '{priority.task}' blocked after repeated failures")
            else:
                self.log(f"   ❌ Failed: {trial.error}")
            return None
    
    def _check_for_finding(self, task: str) -> Optional[ResearchFinding]:
        """Check if current results constitute a significant finding."""
        
        eq_trials = self.engine.db.get_trials(
            algorithm="eqprop", task=task, status="complete")
        bp_trials = self.engine.db.get_trials(
            algorithm="bp", task=task, status="complete")
        
        if len(eq_trials) < 3 or len(bp_trials) < 3:
            return None  # Need more data
        
        eq_perfs = [t.performance for t in eq_trials]
        bp_perfs = [t.performance for t in bp_trials]
        
        result = self.stats.compare(eq_perfs, bp_perfs, "EqProp", "BP")
        
        # Significant finding if p < 0.05 and meaningful effect
        if result.is_significant and abs(result.cohens_d) > 0.5:
            if result.algo1_mean > result.algo2_mean:
                title = f"EqProp outperforms BP on {task}"
                description = f"EqProp achieves {result.algo1_mean:.4f} vs BP's {result.algo2_mean:.4f} ({result.improvement_pct:+.1f}%)"
                significance = min(1.0, abs(result.cohens_d) / 2)
            else:
                title = f"BP outperforms EqProp on {task}"
                description = f"BP achieves {result.algo2_mean:.4f} vs EqProp's {result.algo1_mean:.4f}"
                significance = min(1.0, abs(result.cohens_d) / 2) * 0.8  # Slightly lower for negative
            
            self.log(f"   🎯 FINDING: {title}")
            
            return ResearchFinding(
                timestamp=datetime.now().isoformat(),
                category="performance",
                title=title,
                description=description,
                evidence={
                    "p_value": result.p_value,
                    "effect_size": result.cohens_d,
                    "eqprop_mean": result.algo1_mean,
                    "baseline_mean": result.algo2_mean,
                    "n_eqprop": len(eq_trials),
                    "n_baseline": len(bp_trials)
                },
                significance=significance,
                validated=True
            )
        
        return None
    
    def _run_analysis_cycle(self):
        """Periodic deeper analysis."""
        self.log("\n📊 Running analysis cycle...")
        
        # Find tasks with enough data for analysis
        all_trials = self.engine.db.get_trials(status="complete")
        tasks = set(t.task for t in all_trials)
        
        for task in tasks:
            eq = self.engine.db.get_trials(algorithm="eqprop", task=task, status="complete")
            bp = self.engine.db.get_trials(algorithm="bp", task=task, status="complete")
            
            if len(eq) >= 5 and len(bp) >= 5:
                # Generate validated comparison
                claims = [{
                    "claim": f"EqProp vs BP on {task}",
                    "eqprop": [t.performance for t in eq],
                    "baseline": [t.performance for t in bp]
                }]
                
                try:
                    verdicts = self.validation.validate_claims(claims)
                    self.validation.generate_report(
                        verdicts, 
                        title=f"{task.upper()} Analysis ({datetime.now().strftime('%Y%m%d')})"
                    )
                    self.log(f"   ✅ Generated report for {task}")
                except Exception as e:
                    self.log(f"   ⚠️ Analysis failed for {task}: {e}")
    
    def _checkpoint(self):
        """Save current state."""
        self.state.last_update = datetime.now().isoformat()
        self.state.save(self.checkpoint_path)
        self.last_checkpoint = time.time()
        self.log("💾 Checkpoint saved")
    
    def _update_findings_doc(self):
        """Update findings markdown file."""
        with open(self.findings_path, "w") as f:
            f.write("# Research Findings\n\n")
            f.write(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
            f.write(f"Total findings: {len(self.state.findings)}\n\n")
            f.write("---\n\n")
            
            for finding_dict in sorted(
                self.state.findings, 
                key=lambda x: x.get("significance", 0), 
                reverse=True
            ):
                finding = ResearchFinding(**finding_dict)
                f.write(finding.to_markdown())
                f.write("\n---\n\n")
    
    def _update_results_md(self):
        """Update main RESULTS.md with latest findings."""
        results_path = Path("RESULTS.md")
        
        if not results_path.exists():
            return
        
        # Read current results
        with open(results_path) as f:
            content = f.read()
        
        # Find/update autonomous research section
        section_marker = "## 🤖 Autonomous Research Findings"
        if section_marker not in content:
            # Append new section
            new_section = self._generate_results_section()
            content = content.rstrip() + "\n\n" + new_section
        else:
            # Update existing section
            parts = content.split(section_marker)
            if len(parts) == 2:
                # Find next section
                next_section = parts[1].find("\n## ")
                if next_section != -1:
                    after_section = parts[1][next_section:]
                else:
                    after_section = ""
                
                new_section = self._generate_results_section()
                content = parts[0] + new_section + after_section
        
        # Write updated results
        with open(results_path, "w") as f:
            f.write(content)
        
        self.log("📝 Updated RESULTS.md")
    
    def _generate_results_section(self) -> str:
        """Generate autonomous research section for RESULTS.md."""
        lines = [
            "## 🤖 Autonomous Research Findings",
            "",
            f"*Auto-generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
            f"*Runtime: {self.state.total_runtime_hours:.1f} hours | Experiments: {self.state.experiments_completed}*",
            "",
        ]
        
        # Best results per task
        if self.state.best_eqprop_performance or self.state.best_baseline_performance:
            lines.append("### Best Results by Task")
            lines.append("")
            lines.append("| Task | EqProp | BP | Winner |")
            lines.append("|------|--------|-----|--------|")
            
            all_tasks = set(self.state.best_eqprop_performance.keys()) | \
                        set(self.state.best_baseline_performance.keys())
            
            for task in sorted(all_tasks):
                eq = self.state.best_eqprop_performance.get(task, 0)
                bp = self.state.best_baseline_performance.get(task, 0)
                winner = "🔋 EqProp" if eq > bp else "⚡ BP" if bp > eq else "Tie"
                lines.append(f"| {task} | {eq:.4f} | {bp:.4f} | {winner} |")
            
            lines.append("")
        
        # Key findings
        if self.state.findings:
            lines.append("### Key Findings")
            lines.append("")
            
            for finding_dict in sorted(
                self.state.findings[:5], 
                key=lambda x: x.get("significance", 0), 
                reverse=True
            ):
                finding = ResearchFinding(**finding_dict)
                lines.append(f"- **{finding.title}**: {finding.description}")
            
            lines.append("")
        
        lines.append("---")
        lines.append("")
        
        return "\n".join(lines)
    
    def _generate_final_report(self):
        """Generate comprehensive final report with full analysis."""
        report_path = self.output_dir / "final_report.md"
        
        # Gather all trials from database
        all_trials = self.engine.db.get_trials(status="complete")
        eq_trials = [t for t in all_trials if t.algorithm == "eqprop"]
        bp_trials = [t for t in all_trials if t.algorithm == "bp"]
        
        # Organize by task
        tasks = sorted(set(t.task for t in all_trials))
        
        with open(report_path, "w") as f:
            # Header
            f.write(f"""# 🔬 TorEqProp Autonomous Research Report

> **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
> **System**: Autonomous Research System v1.0

---

## 📊 Executive Summary

| Metric | Value |
|--------|-------|
| **Total Runtime** | {self.state.total_runtime_hours:.2f} hours |
| **Experiments Completed** | {self.state.experiments_completed} |
| **Experiments Failed** | {self.state.experiments_failed} |
| **Success Rate** | {self.state.experiments_completed / max(1, self.state.experiments_completed + self.state.experiments_failed) * 100:.1f}% |
| **Tasks Explored** | {len(tasks)} |
| **Significant Findings** | {len(self.state.findings)} |

---

## 🎯 Key Results by Task

""")
            # Per-task analysis
            for task in tasks:
                task_eq = [t for t in eq_trials if t.task == task]
                task_bp = [t for t in bp_trials if t.task == task]
                
                f.write(f"### {task.upper()}\n\n")
                
                if task_eq and task_bp:
                    # Best configs
                    best_eq = max(task_eq, key=lambda t: t.performance)
                    best_bp = max(task_bp, key=lambda t: t.performance)
                    
                    f.write(f"| Algorithm | Best Score | Trials | Avg Time |\n")
                    f.write(f"|-----------|------------|--------|----------|\n")
                    
                    eq_avg_time = np.mean([t.cost.wall_time_seconds for t in task_eq]) if task_eq else 0
                    bp_avg_time = np.mean([t.cost.wall_time_seconds for t in task_bp]) if task_bp else 0
                    
                    f.write(f"| 🔋 EqProp | **{best_eq.performance:.4f}** | {len(task_eq)} | {eq_avg_time:.1f}s |\n")
                    f.write(f"| ⚡ BP | **{best_bp.performance:.4f}** | {len(task_bp)} | {bp_avg_time:.1f}s |\n\n")
                    
                    # Statistical comparison
                    if len(task_eq) >= 2 and len(task_bp) >= 2:
                        eq_perfs = [t.performance for t in task_eq]
                        bp_perfs = [t.performance for t in task_bp]
                        
                        try:
                            result = self.stats.compare(eq_perfs, bp_perfs, "EqProp", "BP")
                            
                            winner = "🔋 EqProp" if result.algo1_mean > result.algo2_mean else "⚡ BP"
                            sig = "✅ Significant" if result.is_significant else "❌ Not significant"
                            
                            f.write(f"**Statistical Analysis**:\n")
                            f.write(f"- Winner: {winner} ({result.improvement_pct:+.2f}%)\n")
                            f.write(f"- p-value: {result.p_value:.4f} ({sig})\n")
                            f.write(f"- Effect size (Cohen's d): {result.cohens_d:.2f}\n\n")
                        except:
                            pass
                    
                    # Best EqProp config details
                    f.write(f"**Best EqProp Configuration**:\n```yaml\n")
                    for k, v in sorted(best_eq.config.items()):
                        f.write(f"{k}: {v}\n")
                    f.write(f"```\n\n")
                    
                elif task_eq:
                    best_eq = max(task_eq, key=lambda t: t.performance)
                    f.write(f"- EqProp only: Best = {best_eq.performance:.4f} ({len(task_eq)} trials)\n\n")
                elif task_bp:
                    best_bp = max(task_bp, key=lambda t: t.performance)
                    f.write(f"- BP only: Best = {best_bp.performance:.4f} ({len(task_bp)} trials)\n\n")
                
                f.write("---\n\n")
            
            # Findings section
            f.write("## 🔍 Significant Findings\n\n")
            
            if self.state.findings:
                for i, finding_dict in enumerate(sorted(
                    self.state.findings, 
                    key=lambda x: x.get("significance", 0), 
                    reverse=True
                ), 1):
                    finding = ResearchFinding(**finding_dict)
                    f.write(f"### Finding {i}: {finding.title}\n\n")
                    f.write(f"**Category**: {finding.category.title()}\n")
                    f.write(f"**Significance**: {finding.significance:.0%}\n")
                    f.write(f"**Timestamp**: {finding.timestamp}\n\n")
                    f.write(f"{finding.description}\n\n")
                    f.write(f"**Evidence**:\n")
                    for k, v in finding.evidence.items():
                        if isinstance(v, float):
                            f.write(f"- {k}: {v:.4f}\n")
                        else:
                            f.write(f"- {k}: {v}\n")
                    f.write("\n")
            else:
                f.write("*No statistically significant findings yet. Continue running to accumulate more data.*\n\n")
            
            # Overall comparison
            f.write("## 📈 Overall Comparison\n\n")
            
            if eq_trials and bp_trials:
                all_eq_perfs = [t.performance for t in eq_trials]
                all_bp_perfs = [t.performance for t in bp_trials]
                
                f.write(f"| Metric | EqProp | BP |\n")
                f.write(f"|--------|--------|----|\n")
                f.write(f"| Mean Performance | {np.mean(all_eq_perfs):.4f} | {np.mean(all_bp_perfs):.4f} |\n")
                f.write(f"| Std Dev | {np.std(all_eq_perfs):.4f} | {np.std(all_bp_perfs):.4f} |\n")
                f.write(f"| Best | {np.max(all_eq_perfs):.4f} | {np.max(all_bp_perfs):.4f} |\n")
                f.write(f"| Worst | {np.min(all_eq_perfs):.4f} | {np.min(all_bp_perfs):.4f} |\n")
                f.write(f"| Total Trials | {len(eq_trials)} | {len(bp_trials)} |\n\n")
                
                # Wins by task
                eq_wins = 0
                bp_wins = 0
                ties = 0
                for task in tasks:
                    eq_best = self.state.best_eqprop_performance.get(task, 0)
                    bp_best = self.state.best_baseline_performance.get(task, 0)
                    if eq_best > bp_best:
                        eq_wins += 1
                    elif bp_best > eq_best:
                        bp_wins += 1
                    else:
                        ties += 1
                
                f.write(f"**Task Wins**: EqProp {eq_wins} | BP {bp_wins} | Ties {ties}\n\n")
            
            # System performance
            f.write("## ⚙️ System Performance\n\n")
            f.write(f"| Metric | Value |\n")
            f.write(f"|--------|-------|\n")
            f.write(f"| Experiments/Hour | {self.state.experiments_completed / max(0.01, self.state.total_runtime_hours):.1f} |\n")
            f.write(f"| Blocked Tasks | {len(self.state.blocked_tasks)} |\n")
            if self.state.blocked_tasks:
                f.write(f"| Blocked List | {', '.join(self.state.blocked_tasks)} |\n")
            f.write(f"\n")
            
            # Recommendations
            f.write("## 💡 Recommendations\n\n")
            
            recommendations = []
            
            # Analyze which tasks need more exploration
            for task in tasks:
                task_eq = [t for t in eq_trials if t.task == task]
                task_bp = [t for t in bp_trials if t.task == task]
                
                if len(task_eq) < 5 or len(task_bp) < 5:
                    recommendations.append(f"**Run more trials on {task}**: Only {len(task_eq)} EqProp and {len(task_bp)} BP trials. Need ≥5 each for statistical significance.")
                
                if task_eq and task_bp:
                    best_eq = max(t.performance for t in task_eq)
                    best_bp = max(t.performance for t in task_bp)
                    if best_eq > best_bp * 1.1:
                        recommendations.append(f"**Scale up {task}**: EqProp shows {((best_eq/best_bp)-1)*100:.0f}% advantage. Try longer training.")
            
            if not recommendations:
                recommendations.append("**Continue running**: More data will improve statistical power and reveal patterns.")
            
            for rec in recommendations[:5]:
                f.write(f"- {rec}\n")
            
            f.write(f"""
---

## 📋 Appendix: All Trial Configurations

<details>
<summary>Click to expand full trial list</summary>

### EqProp Trials ({len(eq_trials)} total)

| Task | Performance | Time | β | d_model | damping |
|------|-------------|------|---|---------|---------|
""")
            for t in sorted(eq_trials, key=lambda x: x.performance, reverse=True)[:20]:
                beta = t.config.get('beta', 'N/A')
                d_model = t.config.get('d_model', 'N/A')
                damping = t.config.get('damping', 'N/A')
                f.write(f"| {t.task} | {t.performance:.4f} | {t.cost.wall_time_seconds:.1f}s | {beta} | {d_model} | {damping} |\n")

            f.write(f"""
### BP Trials ({len(bp_trials)} total)

| Task | Performance | Time | lr | d_model | optimizer |
|------|-------------|------|----|---------|-----------|
""")
            for t in sorted(bp_trials, key=lambda x: x.performance, reverse=True)[:20]:
                lr = t.config.get('lr', 'N/A')
                d_model = t.config.get('d_model', 'N/A')
                opt = t.config.get('optimizer', 'N/A')
                f.write(f"| {t.task} | {t.performance:.4f} | {t.cost.wall_time_seconds:.1f}s | {lr} | {d_model} | {opt} |\n")

            f.write("""
</details>

---

*Report generated by TorEqProp Autonomous Research System*
*For questions or issues, see the research log at `autonomous_research/researcher.log`*
""")
        
        self.log(f"📄 Final report: {report_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="TorEqProp Autonomous Research System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Turn-key automation:
    INPUT:  Time + Electricity
    OUTPUT: Beneficial Research

Examples:
    python autonomous_researcher.py                 # Run indefinitely
    python autonomous_researcher.py --hours 8       # Run for 8 hours
    python autonomous_researcher.py --resume        # Resume from checkpoint
    python autonomous_researcher.py --status        # Show current status
        """
    )
    
    parser.add_argument("--hours", type=float, default=None,
                       help="Maximum hours to run (default: indefinite)")
    parser.add_argument("--output-dir", type=str, default="autonomous_research",
                       help="Output directory for results")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from last checkpoint")
    parser.add_argument("--until-discovery", action="store_true",
                       help="Stop when significant discovery is made")
    parser.add_argument("--status", action="store_true",
                       help="Show status and exit")
    
    args = parser.parse_args()
    
    # Status mode
    if args.status:
        checkpoint_path = Path(args.output_dir) / "checkpoint.json"
        if checkpoint_path.exists():
            state = ResearchState.load(checkpoint_path)
            print("\n" + "=" * 50)
            print("Autonomous Research Status")
            print("=" * 50)
            print(f"Started: {state.started_at}")
            print(f"Last update: {state.last_update}")
            print(f"Runtime: {state.total_runtime_hours:.2f} hours")
            print(f"Experiments: {state.experiments_completed}")
            print(f"Findings: {len(state.findings)}")
            print("=" * 50)
        else:
            print("No checkpoint found. Run with --help for usage.")
        return
    
    # Create/resume researcher
    researcher = AutonomousResearcher(output_dir=args.output_dir)
    
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║     🔬 TorEqProp AUTONOMOUS RESEARCH SYSTEM 🔬                    ║
    ║                                                                   ║
    ║     INPUT:  Time + Electricity                                    ║
    ║     OUTPUT: Beneficial Research                                   ║
    ║                                                                   ║
    ║     Press Ctrl+C at any time to safely stop and save progress     ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    researcher.run(
        max_hours=args.hours,
        stop_on_discovery=args.until_discovery
    )


if __name__ == "__main__":
    main()
