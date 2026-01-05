"""Discovery Orchestrator - Main experiment controller.

Guides research towards publishable discoveries by:
- Managing experiment campaigns
- Allocating compute to promising trials
- Generating and validating insights
- Producing academic-quality evidence
"""

import time
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
import uuid

from .config import DiscoveryConfig, RAPID_CONFIG
from .database import ExperimentDB, Trial, Insight
from .insights import InsightEngine, ClaimValidation


@dataclass
class CampaignState:
    """Current state of a discovery campaign."""
    start_time: float
    trials_completed: int = 0
    trials_failed: int = 0
    current_phase: str = "initialization"
    best_eqprop_result: float = 0.0
    best_bp_result: float = 0.0
    discoveries: List[str] = None
    
    def __post_init__(self):
        if self.discoveries is None:
            self.discoveries = []


class DiscoveryOrchestrator:
    """Orchestrates TorEqProp discovery campaigns.
    
    Goals:
    1. Find any task where EqProp > BP
    2. Quantify speed vs accuracy tradeoff
    3. Validate β characterization claims
    4. Generate publication-ready evidence
    """
    
    def __init__(self, config: DiscoveryConfig = None, db_path: str = "results/experiments.db"):
        self.config = config or RAPID_CONFIG
        self.db = ExperimentDB(db_path)
        self.insights = InsightEngine(self.db)
        self.state = None
        
        # Training scripts by task category
        self.trainers = {
            "micro": "train_micro.py",
            "classification": "train.py",
            "algorithmic": "train_algorithmic.py",
            "rl": "train_rl.py",
        }
    
    def run_campaign(self, patience_hours: float = None) -> Dict[str, Any]:
        """Run a discovery campaign with time budget.
        
        Args:
            patience_hours: Maximum hours to run (overrides config)
            
        Returns:
            Campaign results summary
        """
        self.state = CampaignState(start_time=time.time())
        max_time = (patience_hours or self.config.max_total_time_hours) * 3600
        
        print("=" * 70)
        print(f"  TorEqProp DISCOVERY CAMPAIGN")
        print("=" * 70)
        print(f"  Campaign: {self.config.name}")
        print(f"  Time budget: {max_time/3600:.1f} hours")
        print(f"  Tasks: {', '.join(self.config.tasks[:5])}...")
        print(f"  Hypotheses to test: {len(self.config.hypotheses)}")
        print("=" * 70)
        print()
        
        try:
            # Phase 1: Rapid micro exploration
            self._phase_micro_exploration()
            
            # Check time
            if self._time_remaining(max_time) < 60:
                return self._finalize_campaign("time_exceeded")
            
            # Phase 2: Targeted comparisons
            self._phase_targeted_comparison()
            
            # Phase 3: Hypothesis validation
            self._phase_hypothesis_validation()
            
        except KeyboardInterrupt:
            print("\n⚠️  Campaign interrupted by user")
            
        return self._finalize_campaign("completed")
    
    def _time_remaining(self, max_time: float) -> float:
        """Get remaining time in seconds."""
        elapsed = time.time() - self.state.start_time
        return max_time - elapsed
    
    def _phase_micro_exploration(self):
        """Phase 1: Rapid exploration on micro tasks."""
        self.state.current_phase = "micro_exploration"
        print("\n" + "=" * 50)
        print("  PHASE 1: Micro Task Exploration")
        print("=" * 50)
        
        micro_tasks = ["xor", "xor3", "majority", "tiny_lm"]
        
        for task in micro_tasks:
            print(f"\n📊 Exploring: {task}")
            
            # Run a few EqProp configs
            for d_model in [16, 32]:
                for beta in [0.20, 0.22, 0.25]:
                    trial = self._run_trial(
                        algorithm="eqprop",
                        task=task,
                        config={"d_model": d_model, "beta": beta, "epochs": 20},
                        seed=42
                    )
                    if trial and trial.performance > self.state.best_eqprop_result:
                        self.state.best_eqprop_result = trial.performance
            
            # Run BP baseline
            for d_model in [16, 32]:
                trial = self._run_trial(
                    algorithm="bp",
                    task=task,
                    config={"d_model": d_model, "epochs": 20},
                    seed=42
                )
                if trial and trial.performance > self.state.best_bp_result:
                    self.state.best_bp_result = trial.performance
            
            # Generate comparison insight
            insight = self.insights.compare_algorithms(task)
            if insight:
                self.db.add_insight(insight)
                print(f"   💡 {insight.title}")
    
    def _phase_targeted_comparison(self):
        """Phase 2: Run matched comparisons on promising tasks."""
        self.state.current_phase = "targeted_comparison"
        print("\n" + "=" * 50)
        print("  PHASE 2: Targeted Comparisons")
        print("=" * 50)
        
        # Find tasks where EqProp showed promise
        summary = self.db.get_summary()
        
        # Run with multiple seeds for statistical validity
        for task in ["xor3", "tiny_lm"]:
            print(f"\n📊 Multi-seed comparison: {task}")
            
            for seed in range(self.config.min_seeds):
                for algo in ["eqprop", "bp"]:
                    self._run_trial(
                        algorithm=algo,
                        task=task,
                        config={"d_model": 32, "epochs": 30},
                        seed=seed
                    )
            
            insight = self.insights.compare_algorithms(task)
            if insight:
                self.db.add_insight(insight)
                print(f"   💡 {insight.title}")
    
    def _phase_hypothesis_validation(self):
        """Phase 3: Validate research hypotheses."""
        self.state.current_phase = "hypothesis_validation"
        print("\n" + "=" * 50)
        print("  PHASE 3: Hypothesis Validation")
        print("=" * 50)
        
        for hypothesis in self.config.hypotheses:
            name = hypothesis["name"]
            print(f"\n🔬 Testing: {hypothesis['description']}")
            
            validation = self.insights.validate_hypothesis(name)
            
            status = "✅ SUPPORTED" if validation.is_supported else "❌ NOT SUPPORTED"
            print(f"   {status} (confidence: {validation.confidence:.0%})")
            
            if validation.skeptic_concerns:
                print("   ⚠️  Skeptic concerns:")
                for concern in validation.skeptic_concerns[:2]:
                    print(f"      - {concern}")
            
            if validation.is_supported:
                self.state.discoveries.append(name)
    
    def _run_trial(self, algorithm: str, task: str, config: Dict[str, Any], 
                   seed: int) -> Optional[Trial]:
        """Run a single trial."""
        trial_id = f"{algorithm[:2]}_{task}_{config.get('d_model', 0)}_{seed}"
        
        # Check if already run
        existing = self.db.get_trial(trial_id)
        if existing and existing.status == "complete":
            return existing
        
        trial = Trial(
            trial_id=trial_id,
            algorithm=algorithm,
            task=task,
            config=config,
            seed=seed,
            status="running"
        )
        self.db.add_trial(trial)
        
        try:
            start_time = time.time()
            
            # Build command
            if task in ["xor", "and", "or", "xor3", "majority", "identity", "tiny_lm"]:
                script = "train_micro.py"
            elif task in ["cartpole", "acrobot", "lunarlander", "mountaincar"]:
                script = "train_rl.py"
            elif task in ["parity", "copy", "addition"]:
                script = "train_algorithmic.py"
            else:
                script = "train.py"
            
            cmd = [
                sys.executable, script,
                "--task", task,
                "--d-model", str(config.get("d_model", 32)),
                "--epochs", str(config.get("epochs", 10)),
                "--seed", str(seed),
            ]
            
            if algorithm == "bp":
                cmd.append("--use-bp")
            elif algorithm == "eqprop":
                cmd.extend(["--beta", str(config.get("beta", 0.22))])
            
            # Run training
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.max_trial_time_seconds,
                cwd=Path(__file__).parent.parent
            )
            
            # Parse output for performance
            output = result.stdout
            performance = self._extract_performance(output)
            
            trial.performance = performance
            trial.wall_time_seconds = time.time() - start_time
            trial.status = "complete"
            trial.completed_at = datetime.now().isoformat()
            
            self.state.trials_completed += 1
            print(f"   ✓ {trial_id}: {performance:.4f} ({trial.wall_time_seconds:.1f}s)")
            
        except subprocess.TimeoutExpired:
            trial.status = "failed"
            trial.error_message = "Timeout"
            self.state.trials_failed += 1
            print(f"   ✗ {trial_id}: Timeout")
            
        except Exception as e:
            trial.status = "failed"
            trial.error_message = str(e)
            self.state.trials_failed += 1
            print(f"   ✗ {trial_id}: {e}")
        
        self.db.add_trial(trial)
        
        # Generate insights
        if trial.status == "complete":
            for insight in self.insights.analyze_trial(trial):
                self.db.add_insight(insight)
        
        return trial if trial.status == "complete" else None
    
    def _extract_performance(self, output: str) -> float:
        """Extract performance metric from training output."""
        import re
        
        # Try different patterns
        patterns = [
            r"Best Test Accuracy: ([\d.]+)",
            r"Test Acc[uracy]*[=:] *([\d.]+)",
            r"performance[=:] *([\d.]+)",
            r"accuracy[=:] *([\d.]+)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                return float(match.group(1))
        
        return 0.0
    
    def _finalize_campaign(self, reason: str) -> Dict[str, Any]:
        """Finalize campaign and generate report."""
        elapsed = time.time() - self.state.start_time
        
        print("\n" + "=" * 70)
        print("  CAMPAIGN COMPLETE")
        print("=" * 70)
        print(f"  Reason: {reason}")
        print(f"  Duration: {elapsed/60:.1f} minutes")
        print(f"  Trials completed: {self.state.trials_completed}")
        print(f"  Trials failed: {self.state.trials_failed}")
        print(f"  Discoveries: {len(self.state.discoveries)}")
        print("=" * 70)
        
        # Generate discoveries markdown
        discoveries_md = self.insights.generate_discoveries_markdown()
        Path("DISCOVERIES.md").write_text(discoveries_md)
        print("\n📄 Generated DISCOVERIES.md")
        
        # Export data
        self.db.export_json("results/campaign_results.json")
        print("📄 Exported results/campaign_results.json")
        
        return {
            "reason": reason,
            "duration_minutes": elapsed / 60,
            "trials_completed": self.state.trials_completed,
            "trials_failed": self.state.trials_failed,
            "discoveries": self.state.discoveries,
            "best_eqprop": self.state.best_eqprop_result,
            "best_bp": self.state.best_bp_result,
        }
    
    def status(self) -> Dict[str, Any]:
        """Get current campaign status."""
        return self.db.get_summary()
    
    def validate_all_hypotheses(self) -> Dict[str, ClaimValidation]:
        """Validate all configured hypotheses."""
        results = {}
        for hypothesis in self.config.hypotheses:
            name = hypothesis["name"]
            results[name] = self.insights.validate_hypothesis(name)
        return results


def main():
    """Entry point for discovery campaign."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="TorEqProp Discovery Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--hours", type=float, default=1.0,
                       help="Maximum hours to run")
    parser.add_argument("--rapid", action="store_true",
                       help="Use rapid configuration (smaller models, fewer trials)")
    parser.add_argument("--status", action="store_true",
                       help="Show database status and exit")
    parser.add_argument("--validate", action="store_true",
                       help="Validate all hypotheses from existing data")
    
    args = parser.parse_args()
    
    config = RAPID_CONFIG if args.rapid else DiscoveryConfig()
    orchestrator = DiscoveryOrchestrator(config)
    
    if args.status:
        summary = orchestrator.status()
        print("\n📊 Discovery Database Status")
        print("=" * 50)
        for key, value in summary.items():
            print(f"  {key}: {value}")
        return
    
    if args.validate:
        print("\n🔬 Validating Hypotheses")
        print("=" * 50)
        results = orchestrator.validate_all_hypotheses()
        for name, validation in results.items():
            status = "✅" if validation.is_supported else "❌"
            print(f"\n{status} {name}")
            print(f"   {validation.evidence_summary}")
        return
    
    # Run campaign
    orchestrator.run_campaign(patience_hours=args.hours)


if __name__ == "__main__":
    main()
