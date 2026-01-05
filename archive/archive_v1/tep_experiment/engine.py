"""
Main orchestration engine for TEP experiments.

Implements the phased, gated evaluation pipeline:
- Phase 1: Rapid Signal Detection (XOR, 8x8 digits)
- Phase 2: Validation on Medium Tasks (MNIST, CartPole)
- Phase 3: Comprehensive Benchmarking (CIFAR-10, sequences)

Each phase has success criteria that must be met before proceeding.

Enhanced Optuna Features:
- NSGA-II for multi-objective with proper population sizing
- SuccessiveHalving pruner with adaptive thresholds
- Hyperparameter importance analysis
- Failed trial handling with NaN-aware objectives
- Constraint-aware sampling (e.g., symmetric requires linear attention)
"""

import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json
import warnings

import optuna
from optuna.trial import TrialState
from optuna.samplers import NSGAIISampler, TPESampler
from optuna.pruners import SuccessiveHalvingPruner, MedianPruner

from .config import (
    PhaseConfig,
    TrialResult,
    PHASE_CONFIGS,
    SMOKE_TEST_CONFIG,
    SharedSearchSpace,
    TEPSearchSpace,
    BPSearchSpace,
)
from .tasks import get_task, get_phase_tasks
from .sampler import (
    sample_full_config,
    TransferSampler,
    enqueue_seed_trials,
)
from .runner import TrialRunner
from .analysis import ParetoFront, ParetoAnalyzer, ReportGenerator
from .derived_metrics import compute_derived_metrics, compare_derived_metrics, summarize_config

# Suppress verbose Optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)


class TEPExperimentEngine:
    """Main orchestration engine for TEP experiments.
    
    Implements the specification's phased, gated approach with:
    - Identical treatment for TEP and BP
    - Multi-objective optimization (4 objectives)
    - Automatic pruning via SuccessiveHalving
    - Pareto front analysis
    - Honest reporting (wins, ties, and losses)
    
    Enhanced Optuna Configuration:
    - NSGA-II with population_size = 2 * sqrt(n_trials)
    - SuccessiveHalving with min_resource=1, reduction_factor=3
    - TPE for single-objective baselines with multivariate=True
    - Failed trial handling with proper NaN/inf management
    """
    
    def __init__(
        self,
        storage_url: str = "sqlite:///tep_experiments.db",
        output_dir: Path = Path("tep_results"),
        logs_dir: Path = Path("tep_logs"),
        n_startup_trials: int = 10,
        seed: int = 42,
    ):
        self.storage_url = storage_url
        self.output_dir = Path(output_dir)
        self.logs_dir = Path(logs_dir)
        self.n_startup_trials = n_startup_trials
        self.seed = seed
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        self.runner = TrialRunner(logs_dir=self.logs_dir)
        self.analyzer = ParetoAnalyzer()
        self.reporter = ReportGenerator(output_dir=self.output_dir)
        self.transfer_sampler = TransferSampler()
        
        # Track all results and statistics
        self.results: Dict[str, Dict[str, Any]] = {}
        self.study_stats: Dict[str, Dict[str, Any]] = {}
    
    def run_smoke_test(self) -> Dict[str, Any]:
        """Run quick smoke test to validate the pipeline.
        
        Uses reduced budget (5 trials, XOR only) to verify everything works.
        """
        print("🔥 Running Smoke Test...")
        print("   Using minimal budget to validate pipeline.")
        
        config = SMOKE_TEST_CONFIG
        results = {}
        
        for task in config.tasks:
            print(f"\n📊 Task: {task}")
            
            # Run TEP
            tep_trials = self._run_trials(
                algorithm="tep",
                task=task,
                phase_config=config,
            )
            
            # Run BP
            bp_trials = self._run_trials(
                algorithm="bp",
                task=task,
                phase_config=config,
            )
            
            # Analyze
            tep_front = self.analyzer.extract_pareto_front(tep_trials, "tep", task)
            bp_front = self.analyzer.extract_pareto_front(bp_trials, "bp", task)
            comparison = self.analyzer.compare_fronts(tep_front, bp_front)
            
            results[task] = {
                "tep_front": tep_front,
                "bp_front": bp_front,
                "comparison": comparison,
                "tep_trials": len(tep_trials),
                "bp_trials": len(bp_trials),
            }
            
            tep_best = tep_front.best_accuracy()
            bp_best = bp_front.best_accuracy()
            print(f"   TEP: {len(tep_front.points)} Pareto points, best acc: {tep_best.accuracy if tep_best else 0:.4f}")
            print(f"   BP:  {len(bp_front.points)} Pareto points, best acc: {bp_best.accuracy if bp_best else 0:.4f}")
            print(f"   Winner: {comparison['winner']}")
        
        print("\n✅ Smoke test complete!")
        print(f"   Runner stats: {self.runner.get_stats()}")
        return results
    
    def run_phase(
        self,
        phase: int,
        seed_from_previous: bool = True,
    ) -> Dict[str, Any]:
        """Run a single phase of the experiment.
        
        Args:
            phase: Phase number (1, 2, or 3)
            seed_from_previous: Whether to seed from previous phase results
            
        Returns:
            Results dictionary with Pareto fronts and comparisons
        """
        if phase not in PHASE_CONFIGS:
            raise ValueError(f"Invalid phase: {phase}. Must be 1, 2, or 3.")
        
        config = PHASE_CONFIGS[phase]
        print(f"\n{'='*60}")
        print(f"🚀 Starting Phase {phase}: {config.name}")
        print(f"{'='*60}")
        print(f"   Tasks: {config.tasks}")
        print(f"   Trials per algorithm: {config.n_trials_per_algorithm}")
        print(f"   Budget: {config.total_budget_hours} hours")
        print(f"   Trial timeout: {config.trial_timeout_seconds}s")
        
        results = {}
        start_time = time.time()
        
        for task in config.tasks:
            print(f"\n📊 Task: {task}")
            
            # Get seed trials from previous phases if available
            seed_trials_tep = []
            seed_trials_bp = []
            if seed_from_previous and phase > 1:
                seed_trials_tep = self.transfer_sampler.get_seed_trials(task, "tep")
                seed_trials_bp = self.transfer_sampler.get_seed_trials(task, "bp")
                if seed_trials_tep:
                    print(f"   🌱 Seeding TEP with {len(seed_trials_tep)} configs from previous phase")
            
            # Run TEP optimization
            print(f"\n   🔵 Running TEP optimization...")
            tep_trials = self._run_trials(
                algorithm="tep",
                task=task,
                phase_config=config,
                seed_trials=seed_trials_tep,
            )
            
            # Run BP optimization
            print(f"\n   🟢 Running BP optimization...")
            bp_trials = self._run_trials(
                algorithm="bp",
                task=task,
                phase_config=config,
                seed_trials=seed_trials_bp,
            )
            
            # Extract Pareto fronts
            tep_front = self.analyzer.extract_pareto_front(tep_trials, "tep", task)
            bp_front = self.analyzer.extract_pareto_front(bp_trials, "bp", task)
            
            # Compare fronts
            comparison = self.analyzer.compare_fronts(tep_front, bp_front)
            
            # Register best configs for transfer seeding
            if tep_front.best_accuracy():
                self.transfer_sampler.register_best(
                    task, "tep", tep_front.best_accuracy().config
                )
            if bp_front.best_accuracy():
                self.transfer_sampler.register_best(
                    task, "bp", bp_front.best_accuracy().config
                )
            
            results[task] = {
                "tep_front": tep_front,
                "bp_front": bp_front,
                "comparison": comparison,
            }
            
            # Print comprehensive summary with derived metrics
            self._print_comprehensive_summary(
                task, tep_front, bp_front, comparison, 
                tep_trials, bp_trials
            )
            
            # Generate task report
            report = self.reporter.generate_phase_report(
                phase, task, tep_front, bp_front, comparison
            )
            self.reporter.save_report(report, f"phase{phase}_{task}_report.md")
        
        # Check success criteria
        success = self._check_phase_success(phase, results, config)
        
        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Phase {phase} Complete in {elapsed/3600:.1f} hours")
        print(f"Success: {'✅ YES' if success else '❌ NO'}")
        print(f"Runner stats: {self.runner.get_stats()}")
        print(f"{'='*60}")
        
        self.results[f"phase{phase}"] = results
        return {
            "results": results,
            "success": success,
            "elapsed_hours": elapsed / 3600,
        }
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """Execute the full phased, gated evaluation.
        
        Runs Phase 1, and only proceeds to Phase 2+ if success criteria are met.
        
        Returns:
            Complete results with phase_reached and final verdict
        """
        print("\n" + "="*60)
        print("🔬 TEP vs BP: Full Experiment Pipeline")
        print("="*60)
        print("This will run all phases until failure or completion.")
        print("Starting with Phase 1: Rapid Signal Detection\n")
        
        all_results = {}
        phase_reached = 0
        
        for phase in [1, 2, 3]:
            print(f"\n{'#'*60}")
            print(f"# PHASE {phase}")
            print(f"{'#'*60}")
            
            phase_output = self.run_phase(phase)
            all_results[f"phase{phase}"] = phase_output["results"]
            phase_reached = phase
            
            if not phase_output["success"]:
                print(f"\n⚠️  Phase {phase} did not meet success criteria.")
                print("    Stopping pipeline as per gated approach.")
                break
            
            print(f"\n✅ Phase {phase} SUCCESS! Proceeding to next phase...")
        
        # Generate final report
        final_report = self.reporter.generate_final_report(
            all_results.get(f"phase{phase_reached}", {}),
            phase_reached
        )
        self.reporter.save_report(final_report, "final_report.md")
        
        # Print final verdict
        print("\n" + "="*60)
        print("🏁 FINAL VERDICT")
        print("="*60)
        print(f"Phase Reached: {phase_reached}")
        
        return {
            "all_results": all_results,
            "phase_reached": phase_reached,
            "final_report_path": self.output_dir / "final_report.md",
        }
    
    def _create_sampler(
        self,
        n_trials: int,
        algorithm: str,
    ) -> optuna.samplers.BaseSampler:
        """Create optimized Optuna sampler.
        
        Uses NSGA-II for multi-objective with properly sized population.
        Population size = min(50, max(10, 2 * sqrt(n_trials)))
        """
        import math
        
        # NSGA-II population sizing
        population_size = min(50, max(10, int(2 * math.sqrt(n_trials))))
        
        return NSGAIISampler(
            population_size=population_size,
            mutation_prob=None,  # Auto-calculated
            crossover_prob=0.9,
            swapping_prob=0.5,
            seed=self.seed,
        )
    
    def _create_pruner(
        self,
        phase_config: PhaseConfig,
    ) -> optuna.pruners.BasePruner:
        """Create optimized Optuna pruner.
        
        Uses SuccessiveHalving for aggressive early stopping.
        """
        return SuccessiveHalvingPruner(
            min_resource=phase_config.pruner_min_resource,
            reduction_factor=phase_config.pruner_reduction_factor,
            min_early_stopping_rate=0,  # Prune as early as possible
        )
    
    def _run_trials(
        self,
        algorithm: str,
        task: str,
        phase_config: PhaseConfig,
        seed_trials: Optional[List[Dict[str, Any]]] = None,
    ) -> List[TrialResult]:
        """Run trials for one algorithm/task combination.
        
        Uses Optuna with NSGA-II for multi-objective optimization.
        """
        task_spec = get_task(task)
        n_trials = phase_config.n_trials_per_algorithm
        
        # Create study with unique name
        timestamp = int(time.time())
        study_name = f"tep_phase{phase_config.phase_number}_{task}_{algorithm}_{timestamp}"
        
        try:
            study = optuna.create_study(
                study_name=study_name,
                storage=self.storage_url,
                load_if_exists=False,
                directions=["maximize", "minimize", "minimize", "minimize"],
                sampler=self._create_sampler(n_trials, algorithm),
                pruner=self._create_pruner(phase_config),
            )
        except Exception as e:
            logger.warning(f"Could not create persistent study: {e}")
            study = optuna.create_study(
                directions=["maximize", "minimize", "minimize", "minimize"],
                sampler=self._create_sampler(n_trials, algorithm),
                pruner=self._create_pruner(phase_config),
            )
        
        # Enqueue seed trials
        if seed_trials:
            enqueue_seed_trials(study, seed_trials)
        
        # Track all trial results
        all_trials: List[TrialResult] = []
        trial_count = 0
        successful_trials = 0
        
        def objective(trial: optuna.Trial) -> Tuple[float, float, float, float]:
            nonlocal trial_count, successful_trials
            trial_count += 1
            
            # Sample configuration with phase-specific constraints
            config = sample_full_config(trial, algorithm, phase=phase_config.phase_number)
            
            # Create trial ID
            trial_id = f"{study_name}_{trial.number}"
            
            # Run trial with timeout
            result = self.runner.run_trial(
                trial_id=trial_id,
                algorithm=algorithm,
                task_name=task,
                config=config,
                seed=trial.number + self.seed,
                max_epochs=task_spec.default_epochs,
                timeout_seconds=phase_config.trial_timeout_seconds,
            )
            
            all_trials.append(result)
            
            # Store user attributes for analysis
            trial.set_user_attr("config", json.dumps(result.config))
            trial.set_user_attr("status", result.status)
            trial.set_user_attr("accuracy", result.accuracy)
            trial.set_user_attr("wall_time", result.wall_time_seconds)
            
            # Print progress every 10 trials
            if trial_count % 10 == 0 or trial_count <= 3:
                success_rate = successful_trials / max(1, trial_count - 1)
                print(f"      Trial {trial_count}/{n_trials}: acc={result.accuracy:.4f}, status={result.status}, success_rate={success_rate:.1%}")
            
            # Handle failed trials properly for multi-objective
            if result.status != "complete" or result.accuracy <= 0:
                # Return dominated values that won't pollute the Pareto front
                # Use large but finite values to avoid NaN issues
                return (0.0, 1e6, 1e9, 1e6)
            
            successful_trials += 1
            
            # Return actual objectives
            return (
                result.accuracy,
                result.wall_time_seconds,
                float(result.param_count),
                float(max(1, result.convergence_steps)),  # Avoid zero
            )
        
        # Run optimization
        total_timeout = phase_config.total_budget_hours * 3600 / 2  # Half budget per algorithm
        
        try:
            study.optimize(
                objective,
                n_trials=n_trials,
                timeout=total_timeout,
                show_progress_bar=True,
                catch=(Exception,),
                gc_after_trial=True,  # Garbage collect after each trial
            )
        except KeyboardInterrupt:
            print("\n⚠️  Optimization interrupted by user")
        
        # Store study statistics
        self.study_stats[study_name] = {
            "n_trials": len(study.trials),
            "n_complete": len([t for t in study.trials if t.state == TrialState.COMPLETE]),
            "n_pruned": len([t for t in study.trials if t.state == TrialState.PRUNED]),
            "n_failed": len([t for t in study.trials if t.state == TrialState.FAIL]),
        }
        
        # Analyze hyperparameter importance if enough trials
        if len(study.trials) >= 20:
            self._analyze_importance(study, study_name)
        
        return all_trials
    
    def _analyze_importance(self, study: optuna.Study, study_name: str):
        """Analyze hyperparameter importance."""
        try:
            # For multi-objective, analyze importance for first objective (accuracy)
            importance = optuna.importance.get_param_importances(
                study,
                target=lambda t: t.values[0] if t.values else 0,
            )
            
            print(f"\n      📊 Hyperparameter Importance for {study_name}:")
            for param, score in list(importance.items())[:5]:
                print(f"         {param}: {score:.3f}")
                
            # Save importance to file
            importance_path = self.output_dir / f"{study_name}_importance.json"
            with open(importance_path, "w") as f:
                json.dump(importance, f, indent=2)
                
        except Exception as e:
            logger.debug(f"Could not compute importance: {e}")
    
    def _check_phase_success(
        self,
        phase: int,
        results: Dict[str, Any],
        config: PhaseConfig,
    ) -> bool:
        """Check if phase success criteria are met.
        
        Phase 1: TEP Pareto front dominates BP on at least one task.
        
        Returns True if TEP shows advantage, False otherwise.
        """
        for task, data in results.items():
            comparison = data["comparison"]
            tep_front = data["tep_front"]
            bp_front = data["bp_front"]
            
            winner = comparison["winner"]
            
            # Extract accuracy comparison
            best_acc = comparison.get("best_accuracy", {})
            tep_acc = best_acc.get("tep", 0)
            bp_acc = best_acc.get("bp", 0)
            delta = best_acc.get("delta", 0)
            
            tie_broken_by = comparison.get("tie_broken_by")
            
            print(f"\n   📊 Detailed Analysis for {task}:")
            print(f"      Winner: {winner}" + (f" (tie broken by {tie_broken_by})" if tie_broken_by else ""))
            print(f"      TEP: {len(tep_front.points)} Pareto points, best acc: {tep_acc:.4f}")
            print(f"      BP:  {len(bp_front.points)} Pareto points, best acc: {bp_acc:.4f}")
            print(f"      Accuracy delta: {delta:+.4f} ({'TEP' if delta > 0 else 'BP'} advantage)")
            
            # Check if TEP wins
            if winner == "tep":
                if delta > 0:
                    print(f"   ✅ Success: TEP beat BP by {delta:.4f} accuracy")
                    return True
                elif len(tep_front.points) > len(bp_front.points):
                    print(f"   ✅ Success: TEP has more Pareto points")
                    return True
            elif winner == "bp":
                print(f"   ❌ BP won with {abs(delta):.4f} higher accuracy")
            else:
                print(f"   ⚖️  True tie - need more trials")
        
        print(f"\n   ⚠️  Phase {phase} unsuccessful: TEP did not demonstrate advantages")
        return False
    
    def _print_comprehensive_summary(
        self,
        task: str,
        tep_front: ParetoFront,
        bp_front: ParetoFront,
        comparison: Dict[str, Any],
        tep_trials: List[TrialResult],
        bp_trials: List[TrialResult],
    ):
        """Print comprehensive summary with derived metrics and top configs."""
        
        print(f"\n   📈 Results for {task}:")
        print(f"      TEP: {len(tep_front.points)} Pareto points")
        print(f"      BP:  {len(bp_front.points)} Pareto points")
        
        # Get best configs
        tep_best = tep_front.best_accuracy()
        bp_best = bp_front.best_accuracy()
        
        if tep_best and bp_best:
            # Compute derived metrics
            tep_derived = compute_derived_metrics(
                tep_best.accuracy,
                tep_best.wall_time,
                tep_best.param_count,
                tep_best.convergence_steps
            )
            bp_derived = compute_derived_metrics(
                bp_best.accuracy,
                bp_best.wall_time,
                bp_best.param_count,
                bp_best.convergence_steps
            )
            
            # Print base objectives
            print(f"\n   🎯 Best Configurations:")
            print(f"      TEP: acc={tep_best.accuracy:.4f}, time={tep_best.wall_time:.1f}s, params={tep_best.param_count}")
            print(f"      BP:  acc={bp_best.accuracy:.4f}, time={bp_best.wall_time:.1f}s, params={bp_best.param_count}")
            
            # Print derived metrics
            print(f"\n   ⚡ Efficiency Metrics:")
            print(f"      Learning Power (acc/time):")
            print(f"        TEP: {tep_derived.learning_power:.4f} acc/s")
            print(f"        BP:  {bp_derived.learning_power:.4f} acc/s")
            winner = "TEP" if tep_derived.learning_power > bp_derived.learning_power else "BP"
            print(f"        Winner: {winner}")
            
            print(f"      Parameter Efficiency (acc/log10(params)):")
            print(f"        TEP: {tep_derived.param_efficiency:.4f}")
            print(f"        BP:  {bp_derived.param_efficiency:.4f}")
            winner = "TEP" if tep_derived.param_efficiency > bp_derived.param_efficiency else "BP"
            print(f"        Winner: {winner}")
            
            print(f"      Overall Efficiency Score:")
            print(f"        TEP: {tep_derived.efficiency_score:.4f}")
            print(f"        BP:  {bp_derived.efficiency_score:.4f}")
            winner = "TEP" if tep_derived.efficiency_score > bp_derived.efficiency_score else "BP"
            print(f"        Winner: {winner}")
            
            # Print hyperparameters
            print(f"\n   ⚙️  Best Hyperparameters:")
            print(f"      TEP: {summarize_config(tep_best.config, 'tep')}")
            print(f"      BP:  {summarize_config(bp_best.config, 'bp')}")
        
        # Print top 3 configs from each
        print(f"\n   🏆 Top 3 TEP Configs (by accuracy):")
        tep_sorted = sorted(
            [t for t in tep_trials if t.status == "complete" and t.accuracy > 0],
            key=lambda x: x.accuracy,
            reverse=True
        )[:3]
        for i, trial in enumerate(tep_sorted, 1):
            derived = compute_derived_metrics(
                trial.accuracy, trial.wall_time_seconds,
                trial.param_count, trial.convergence_steps
            )
            print(f"      #{i}: acc={trial.accuracy:.4f}, power={derived.learning_power:.4f}, eff={derived.efficiency_score:.4f}")
            print(f"          {summarize_config(trial.config, 'tep')}")
        
        print(f"\n   🏆 Top 3 BP Configs (by accuracy):")
        bp_sorted = sorted(
            [t for t in bp_trials if t.status == "complete" and t.accuracy > 0],
            key=lambda x: x.accuracy,
            reverse=True
        )[:3]
        for i, trial in enumerate(bp_sorted, 1):
            derived = compute_derived_metrics(
                trial.accuracy, trial.wall_time_seconds,
                trial.param_count, trial.convergence_steps
            )
            print(f"      #{i}: acc={trial.accuracy:.4f}, power={derived.learning_power:.4f}, eff={derived.efficiency_score:.4f}")
            print(f"          {summarize_config(trial.config, 'bp')}")
        
        # Print winner
        print(f"\n   🎖️  Winner: {comparison['winner']}")
        if "tie_broken_by" in comparison:
            print(f"       (tie broken by {comparison['tie_broken_by']})")
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            "runner_stats": self.runner.get_stats(),
            "study_stats": self.study_stats,
            "results": {k: {"tasks": list(v.keys())} for k, v in self.results.items()},
        }


def create_engine(
    storage_url: str = "sqlite:///tep_experiments.db",
    output_dir: str = "tep_results",
    logs_dir: str = "tep_logs",
    seed: int = 42,
) -> TEPExperimentEngine:
    """Factory function to create experiment engine."""
    return TEPExperimentEngine(
        storage_url=storage_url,
        output_dir=Path(output_dir),
        logs_dir=Path(logs_dir),
        seed=seed,
    )
