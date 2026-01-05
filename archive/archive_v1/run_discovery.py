#!/usr/bin/env python3
"""
TorEqProp Experiment Orchestrator

Turn-key discovery process with:
- Intelligent resource allocation
- Modular, extensible experiment framework
- Configuration-driven experiment definitions
- Rapid, detailed feedback

Usage:
    python run_discovery.py                    # Full discovery campaign
    python run_discovery.py --phase 1          # Run specific phase
    python run_discovery.py --quick            # Quick validation (1 epoch each)
    python run_discovery.py --dry-run          # Show what would run
    python run_discovery.py --config custom.yaml  # Use custom config
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from src.experiment_framework import (
    Experiment,
    ExperimentResult,
    ExperimentStatus,
    ExperimentBuilder,
    ExperimentRegistry,
    ResultsAggregator,
    create_default_campaign,
    # Concrete experiment types
    ClassificationExperiment,
    AlgorithmicExperiment,
    RLExperiment,
    MemoryProfilingExperiment,
)


# ============================================================================
# Campaign Runner
# ============================================================================

class CampaignRunner:
    """Orchestrate experiment campaigns with tracking and reporting."""
    
    def __init__(self, output_dir: Path, verbose: bool = True):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self.results: List[ExperimentResult] = []
        self.start_time = datetime.now()
        
    def log(self, message: str):
        """Print message if verbose."""
        if self.verbose:
            print(message)
    
    def run_experiment(self, experiment: Experiment, dry_run: bool = False) -> ExperimentResult:
        """Run a single experiment with logging."""
        import sys
        
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.log(f"\n🚀 [{timestamp}] Starting: {experiment.name}")
        self.log(f"   📂 Category: {experiment.category}")
        self.log(f"   ⚡ Priority: {experiment.priority}")
        self.log(f"   ⏱️  Expected: ~{experiment.expected_duration_min} min")
        self.log(f"   🔬 Hypothesis: {experiment.get_hypothesis()}")
        
        if not dry_run:
            cmd = experiment.build_command()
            self.log(f"   💻 Command: {cmd}")
            self.log(f"   🏃 Running...")
            sys.stdout.flush()  # Force output
        
        result = experiment.run(self.output_dir, dry_run=dry_run)
        self.results.append(result)
        
        # Log result
        timestamp = datetime.now().strftime('%H:%M:%S')
        icon = "✅" if result.status == ExperimentStatus.SUCCESS else \
               "❌" if result.status == ExperimentStatus.FAILURE else \
               "⚠️" if result.status == ExperimentStatus.ERROR else "⏭️"
        
        self.log(f"\n{icon} [{timestamp}] {result.name} [{result.status.value.upper()}]")
        if result.metrics:
            for metric, value in result.metrics.items():
                self.log(f"   📈 {metric}: {value:.4f}")
        self.log(f"   ⏰ Duration: {result.duration_sec:.1f}s ({result.duration_sec/60:.1f}min)")
        for insight in result.insights:
            self.log(f"   💡 {insight}")
        if result.log_path:
            self.log(f"   📄 Log: {result.log_path}")
        
        # Save intermediate results
        self._save_results()
        sys.stdout.flush()  # Force output
        
        return result
    
    def run_campaign(self, experiments: List[Experiment], dry_run: bool = False, confirm: bool = False) -> None:
        """Run a full campaign of experiments."""
        total = len(experiments)
        total_time = sum(e.expected_duration_min for e in experiments)
        
        self.log(f"\n{'='*70}")
        self.log(f"🔬 TorEqProp Discovery Campaign")
        self.log(f"{'='*70}")
        self.log(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f"📊 Experiments: {total}")
        self.log(f"⏱️  Estimated time: {total_time:.0f} minutes ({total_time/60:.1f} hours)")
        self.log(f"💾 Output: {self.output_dir}")
        self.log("")
        
        # List experiments
        self.log("📋 Experiments to run:")
        for i, exp in enumerate(experiments, 1):
            self.log(f"  {i}. [{exp.category:^15}] {exp.name:30} ({exp.priority:6}) ~{exp.expected_duration_min}min")
        self.log("")
        
        if dry_run:
            self.log("[DRY RUN] Would run the above experiments.")
            self.log("Use without --dry-run to execute.")
            return
        
        # Optional confirmation
        if confirm:
            self.log("Press Enter to start, or Ctrl+C to cancel...")
            try:
                input()
            except KeyboardInterrupt:
                self.log("\nCancelled.")
                return
        else:
            self.log("🚀 Starting experiments in 2 seconds...")
            import time
            time.sleep(2)
        
        # Run each experiment
        for i, experiment in enumerate(experiments, 1):
            elapsed = (datetime.now() - self.start_time).total_seconds() / 60
            remaining = total_time - elapsed
            progress = (i - 1) / total * 100
            
            self.log(f"\n{'='*70}")
            self.log(f"📍 Progress: [{i}/{total}] ({progress:.0f}% complete)")
            self.log(f"⏰ Elapsed: {elapsed:.1f}min | Remaining: ~{remaining:.0f}min")
            self.log(f"{'='*70}")
            self.run_experiment(experiment, dry_run=False)
        
        # Print summary
        self.print_summary()
    
    def _save_results(self):
        """Save intermediate results."""
        results_file = self.output_dir / "results.json"
        data = {
            "start_time": self.start_time.isoformat(),
            "last_updated": datetime.now().isoformat(),
            "results": [r.to_dict() for r in self.results]
        }
        with open(results_file, "w") as f:
            json.dump(data, f, indent=2)
    
    def print_summary(self):
        """Print and save final summary."""
        aggregator = ResultsAggregator(self.results)
        summary = aggregator.summarize()
        
        duration = (datetime.now() - self.start_time).total_seconds() / 60
        
        self.log(f"\n{'='*70}")
        self.log("📊 DISCOVERY CAMPAIGN SUMMARY")
        self.log(f"{'='*70}")
        self.log(f"Total experiments: {summary.total_experiments}")
        self.log(f"  ✅ Passed: {summary.passed}")
        self.log(f"  ❌ Failed: {summary.failed}")
        self.log(f"  ⚠️  Errors: {summary.errors}")
        self.log(f"  ⏭️  Skipped: {summary.skipped}")
        self.log(f"Total duration: {duration:.1f} minutes")
        self.log("")
        
        # Per-category summary
        self.log("Results by category:")
        for cat, stats in summary.results_by_category.items():
            self.log(f"  {cat}: {stats['passed']}/{stats['passed']+stats['failed']+stats['errors']} passed")
        
        # Best results
        if summary.best_results:
            self.log("\n🏆 Best results:")
            for cat, result in summary.best_results.items():
                metric = result.metrics.get("test_accuracy", result.metrics.get("avg_reward", 0))
                self.log(f"  {cat}: {result.name} ({metric:.4f})")
        
        # Key insights
        if summary.insights:
            self.log("\n💡 Key insights:")
            for insight in summary.insights[:10]:
                self.log(f"  • {insight}")
        
        # Recommendations
        self.log("\n📋 Recommendations:")
        failed = [r for r in self.results if r.status == ExperimentStatus.FAILURE]
        if failed:
            self.log("  Investigate failures:")
            for r in failed[:3]:
                metric_name, _ = r.metadata.get("config", {}).get("success_criteria", ("metric", 0))
                self.log(f"    - {r.name}")
        
        passed = [r for r in self.results if r.status == ExperimentStatus.SUCCESS]
        if passed:
            self.log("  Scale promising directions:")
            for r in sorted(passed, key=lambda x: max(x.metrics.values()) if x.metrics else 0, reverse=True)[:3]:
                self.log(f"    - {r.name}")
        
        self.log(f"{'='*70}\n")
        
        # Save markdown summary
        summary_file = self.output_dir / "summary.md"
        with open(summary_file, "w") as f:
            f.write(aggregator.to_markdown())
        
        self.log(f"Full results saved to: {self.output_dir}/")


# ============================================================================
# Experiment Filtering
# ============================================================================

def filter_experiments(
    experiments: List[Experiment],
    phases: Optional[List[int]] = None,
    categories: Optional[List[str]] = None,
    priority: Optional[str] = None,
    quick_mode: bool = False,
    smoke_test: bool = False
) -> List[Experiment]:
    """Filter and modify experiments based on criteria."""
    
    # Assign phase numbers based on experiment order
    phase_map = {
        "MNIST Rapid": 1, "Fashion Rapid": 1, "CIFAR-10 Rapid": 1, "SVHN Rapid": 1,
        "Parity N=8": 2, "Parity N=12": 2, "Copy Task": 2, "Addition 4-digit": 2,
        "CartPole EqProp": 3, "CartPole BP": 3,
        "MNIST Extended": 4,
        "Memory d=256": 5, "Memory d=1024": 5, "Memory d=2048": 5,
    }
    
    filtered = experiments
    
    # Filter by phase
    if phases:
        filtered = [e for e in filtered if phase_map.get(e.name, 0) in phases]
    
    # Filter by category
    if categories:
        filtered = [e for e in filtered if e.category in categories]
    
    # Filter by priority
    if priority:
        filtered = [e for e in filtered if e.priority == priority]
    
    # Quick mode: reduce epochs
    if quick_mode:
        quick_experiments = []
        for exp in filtered:
            # Create quick version with reduced epochs
            new_config = exp.config.copy()
            new_config["name"] = exp.name + " [QUICK]"
            new_config["epochs"] = 1
            new_config["episodes"] = 100  # For RL
            new_config["expected_time_min"] = 2
            new_config["success_threshold"] = exp.config.get("success_threshold", 0.5) * 0.5
            
            quick_experiments.append(ExperimentBuilder.from_dict(new_config))
        filtered = quick_experiments
    
    # Smoke test mode: ultra-minimal for infrastructure verification
    if smoke_test:
        smoke_experiments = []
        for exp in filtered:
            new_config = exp.config.copy()
            new_config["name"] = exp.name + " [SMOKE]"
            
            # Minimal model dimensions (except for memory profiling where we test different sizes)
            exp_type = new_config.get("type", "")
            if exp_type != "memory":
                # Only override d_model for non-memory experiments
                new_config["d_model"] = 32
                new_config["n_heads"] = 2
                new_config["d_ff"] = 64
            
            new_config["max_iters"] = 5
            
            # Minimal training
            new_config["epochs"] = 1
            new_config["batch_size"] = 32
            new_config["episodes"] = 10  # For RL
            
            # Reduce time estimates
            new_config["expected_time_min"] = 0.5
            new_config["success_threshold"] = 0.1  # Just needs to run
            
            smoke_experiments.append(ExperimentBuilder.from_dict(new_config))
        filtered = smoke_experiments
    
    return filtered


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="TorEqProp Discovery Campaign",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_discovery.py                     # Full campaign
  python run_discovery.py --phase 1           # Phase 1 only (dataset sweep)
  python run_discovery.py --phase 1 2 3       # Phases 1-3 (rapid exploration)
  python run_discovery.py --smoke-test        # Smoke test all phases (<5min)
  python run_discovery.py --quick             # Quick validation (1 epoch each)
  python run_discovery.py --dry-run           # Preview experiments
  python run_discovery.py --category rl       # RL experiments only
  python run_discovery.py --priority HIGH     # HIGH priority only
  python run_discovery.py --config custom.yaml # Use custom config
        """
    )
    
    # Filtering options
    parser.add_argument("--phase", type=int, nargs="+", 
                        help="Run specific phase(s): 1=datasets, 2=algorithmic, 3=RL, 4=accuracy, 5=memory")
    parser.add_argument("--category", type=str, nargs="+",
                        choices=["classification", "algorithmic", "rl", "memory"],
                        help="Filter by experiment category")
    parser.add_argument("--priority", choices=["HIGH", "MEDIUM", "LOW"],
                        help="Filter by priority level")
    parser.add_argument("--quick", action="store_true",
                        help="Quick validation mode (1 epoch each)")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Ultra-minimal smoke test (d=32, 1 epoch, <5min total)")
    
    # Configuration
    parser.add_argument("--config", type=str,
                        help="Path to custom experiment config (YAML or JSON)")
    parser.add_argument("--output-dir", type=str, default="logs/discovery",
                        help="Output directory for results")
    
    # Execution options
    parser.add_argument("--dry-run", action="store_true",
                        help="Show experiments without running")
    parser.add_argument("--quiet", action="store_true",
                        help="Reduce output verbosity")
    
    # List available options
    parser.add_argument("--list-experiments", action="store_true",
                        help="List all configured experiments and exit")
    parser.add_argument("--list-types", action="store_true",
                        help="List registered experiment types and exit")
    
    args = parser.parse_args()
    
    # List modes
    if args.list_types:
        print("Registered experiment types:")
        for exp_type in ExperimentRegistry.list_experiment_types():
            print(f"  - {exp_type}")
        return
    
    # Load experiments
    if args.config:
        config_path = Path(args.config)
        if config_path.suffix in [".yaml", ".yml"]:
            experiments = ExperimentBuilder.from_yaml(config_path)
        else:
            experiments = ExperimentBuilder.from_json(config_path)
    else:
        experiments = create_default_campaign()
    
    if args.list_experiments:
        print("Configured experiments:")
        for i, exp in enumerate(experiments, 1):
            print(f"  {i}. [{exp.category}] {exp.name} ({exp.priority})")
        return
    
    # Filter experiments
    experiments = filter_experiments(
        experiments,
        phases=args.phase,
        categories=args.category,
        priority=args.priority,
        quick_mode=args.quick,
        smoke_test=args.smoke_test
    )
    
    if not experiments:
        print("No experiments match the specified criteria.")
        return
    
    # Create output directory with timestamp
    output_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Run campaign
    runner = CampaignRunner(output_dir, verbose=not args.quiet)
    
    # Enable line buffering for real-time output
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    
    runner.run_campaign(experiments, dry_run=args.dry_run, confirm=False)


if __name__ == "__main__":
    main()
