import argparse
import random
import sys
import numpy as np
import yaml
from pathlib import Path
from typing import List, Dict

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# Import relevant classes from our local package
from .core import HyperOptTrial
from .search_spaces import SearchSpace, EqPropSearchSpace, BaselineSearchSpace
from .storage import HyperOptDB
from .evaluator import CostAwareEvaluator, TimeBudgetEvaluator
from .sampling import Sampler
from .analysis import ParetoAnalyzer, TrialMatcher, StatisticalAnalyzer

class HyperOptEngine:
    """Main hyperparameter optimization engine."""
    
    def __init__(self, config_path: str = "validation_config.yaml"):
        self.config = self._load_config(config_path)
        self.db = HyperOptDB(self.config.get("output", {}).get(
            "hyperopt_db", "data/hyperopt_results.json"))
        
        self.eqprop_space = self._create_eqprop_space()
        self.baseline_space = self._create_baseline_space()
        
        logs_dir = Path(self.config.get("output", {}).get(
            "logs_dir", "logs/hyperopt"))
        self.evaluator = CostAwareEvaluator(logs_dir)
        
        self.matcher = TrialMatcher(
            strategy=self.config.get("hyperopt", {}).get(
                "matching", {}).get("strategy", "time_matched"),
            tolerance=self.config.get("hyperopt", {}).get(
                "matching", {}).get("tolerance", 0.1)
        )
        
        self.analyzer = StatisticalAnalyzer()
        self.sampler = Sampler()

    def _load_config(self, path: str) -> dict:
        """Load configuration from YAML."""
        if Path(path).exists():
            with open(path) as f:
                return yaml.safe_load(f)
        return {}
    
    def _create_eqprop_space(self) -> EqPropSearchSpace:
        """Create EqProp search space from config."""
        cfg = self.config.get("hyperopt", {}).get("eqprop_search_space", {})
        return EqPropSearchSpace(
            beta=cfg.get("beta", [0.05, 0.1, 0.15, 0.2, 0.22, 0.25, 0.3, 0.5]),
            damping=cfg.get("damping", [0.7, 0.8, 0.9, 0.95]),
            max_iters=cfg.get("max_iters", [10, 20, 50, 100]),
            tol=cfg.get("tol", [1e-4, 1e-5, 1e-6]),
            attention_type=cfg.get("attention_type", ["linear"]),
            symmetric=cfg.get("symmetric", [False, True]),
            update_mode=cfg.get("update_mode", ["mse_proxy", "vector_field"]),
            d_model=cfg.get("d_model", [64, 128, 256]),
            lr=cfg.get("lr", [5e-4, 1e-3, 2e-3]),
        )
    
    def _create_baseline_space(self) -> BaselineSearchSpace:
        """Create baseline search space from config."""
        cfg = self.config.get("hyperopt", {}).get("baseline_search_space", {})
        return BaselineSearchSpace(
            lr=cfg.get("lr", [1e-4, 5e-4, 1e-3, 2e-3, 5e-3]),
            optimizer=cfg.get("optimizer", ["adam", "adamw"]),
            d_model=cfg.get("d_model", [64, 128, 256]),
            weight_decay=cfg.get("weight_decay", [0, 1e-4, 1e-3]),
            scheduler=cfg.get("scheduler", ["none", "cosine"]),
        )

    def run(self, task: str = "mnist", n_trials: int = 50,
            strategy: str = "random", seeds: List[int] = None,
            epochs: int = 5, headless: bool = False,
            time_budget: float = None):
        """Run hyperparameter optimization."""
        
        # Override evaluator if time budget is set
        if time_budget is not None:
            self.evaluator = TimeBudgetEvaluator(self.evaluator.logs_dir, time_budget)
            
        print("\n" + "=" * 70)
        print("  TorEqProp Competitive Hyperparameter Optimization")
        print("=" * 70)
        print(f"  Task: {task}")
        print(f"  Strategy: {strategy}")
        print(f"  Trials per algorithm: {n_trials}")
        print(f"  Epochs per trial: {epochs if time_budget is None else 'dynamic (time budget)'}")
        print(f"  EqProp search space: {self.eqprop_space.size()} configs")
        print(f"  Baseline search space: {self.baseline_space.size()} configs")
        print("=" * 70)
        
        if seeds is None:
            seeds = [0, 1, 2]
        
        rng = random.Random(42)
        
        # Generate trial configurations
        eqprop_configs = self.sampler.sample_configs(self.eqprop_space, n_trials, strategy)
        baseline_configs = self.sampler.sample_configs(self.baseline_space, n_trials, strategy)
        
        # Run EqProp trials
        print("\n" + "-" * 70)
        print("🔋 Phase 1: EqProp Hyperparameter Search")
        print("-" * 70)
        
        self._run_phase(eqprop_configs, "eqprop", task, seeds, epochs, headless, "eq")
        
        # Run baseline trials
        print("\n" + "-" * 70)
        print("⚡ Phase 2: Baseline Hyperparameter Search")
        print("-" * 70)
        
        self._run_phase(baseline_configs, "bp", task, seeds, epochs, headless, "bp")
        
        # Analysis
        self._print_analysis(task)
    
    def _run_phase(self, configs, algorithm, task, seeds, epochs, headless, prefix):
        """Run a phase of trials for a specific algorithm."""
        
        # Create iterator with progress bar if available
        if HAS_TQDM:
            trial_iterator = tqdm(
                list(enumerate(configs)),
                desc=f"{'EqProp' if algorithm=='eqprop' else 'Baseline'} Trials",
                total=len(configs) * len(seeds),
                unit="trial"
            )
        else:
            trial_iterator = enumerate(configs)
        
        for i, cfg in trial_iterator:
            for seed in seeds:
                trial_id = f"{prefix}_{task}_{i}_s{seed}"
                
                # Skip if already complete
                existing = self.db.get_trial(trial_id)
                if existing and existing.status == "complete":
                    if HAS_TQDM:
                        trial_iterator.set_postfix_str(f"⏭️  {trial_id} (cached)")
                    else:
                        print(f"  ⏭️  {trial_id} already complete, skipping")
                    continue
                
                trial = HyperOptTrial(
                    trial_id=trial_id,
                    algorithm=algorithm,
                    config=cfg,
                    task=task,
                    seed=seed,
                )
                
                if not headless and not HAS_TQDM:
                    self._print_trial_info(i, len(configs), seed, trial_id, cfg)
                
                def callback(line):
                    if not headless and not HAS_TQDM:
                        print(f"   {line.strip()}", flush=True)
                
                trial = self.evaluator.evaluate(trial, epochs=epochs, callback=callback)
                self.db.add_trial(trial)
                
                status = "✅" if trial.status == "complete" else "❌"
                perf_str = f"{trial.performance:.4f}"
                time_str = f"{trial.cost.wall_time_seconds:.1f}s"
                
                if HAS_TQDM:
                    trial_iterator.set_postfix_str(f"{status} {perf_str} @ {time_str}")
                elif not headless:
                    print(f"   {status} Performance: {perf_str}, Time: {time_str}")

    def _print_trial_info(self, i, total, seed, trial_id, cfg):
        print(f"\n📊 Trial {i+1}/{total} seed {seed}: {trial_id}")
        # Simplified print, can elaborate based on config contents if needed
        print(f"   Config: {cfg}")
    
    def _print_analysis(self, task: str):
        """Print analysis of completed trials."""
        print("\n" + "=" * 70)
        print("  HYPEROPT ANALYSIS")
        print("=" * 70)
        
        eqprop_trials = self.db.get_trials(algorithm="eqprop", task=task, status="complete")
        baseline_trials = self.db.get_trials(algorithm="bp", task=task, status="complete")
        
        if not eqprop_trials or not baseline_trials:
            print("  ❌ Insufficient trials for analysis")
            return
        
        # Best configurations
        best_eq = max(eqprop_trials, key=lambda t: t.performance)
        best_bl = max(baseline_trials, key=lambda t: t.performance)
        
        print(f"\n📊 Best Configurations:")
        print(f"\n  🔋 EqProp Best: {best_eq.performance:.4f}")
        configs_str = ", ".join([f"{k}={v}" for k, v in best_eq.config.items() if k in ['beta', 'damping', 'max_iters', 'd_model']])
        print(f"     Config: {configs_str}")
        print(f"     Time: {best_eq.cost.wall_time_seconds:.1f}s")
        
        print(f"\n  ⚡ Baseline Best: {best_bl.performance:.4f}")
        configs_str = ", ".join([f"{k}={v}" for k, v in best_bl.config.items() if k in ['lr', 'optimizer', 'd_model']])
        print(f"     Config: {configs_str}")
        print(f"     Time: {best_bl.cost.wall_time_seconds:.1f}s")
        
        # Statistical comparison
        eq_perfs = [t.performance for t in eqprop_trials]
        bl_perfs = [t.performance for t in baseline_trials]
        
        result = self.analyzer.compare(eq_perfs, bl_perfs, "EqProp", "Baseline")
        
        print(f"\n📈 Statistical Comparison (all trials):")
        print(f"   EqProp: {result.algo1_mean:.4f} ± {result.algo1_std:.4f} (n={result.algo1_n})")
        print(f"   Baseline: {result.algo2_mean:.4f} ± {result.algo2_std:.4f} (n={result.algo2_n})")
        print(f"   Difference: {result.improvement_pct:+.2f}%")
        print(f"   p-value: {result.p_value:.4f}")
        print(f"   Cohen's d: {result.cohens_d:.2f}")
        print(f"   Significant: {'Yes' if result.is_significant else 'No'}")
        
        # Matched comparison
        pairs = self.matcher.match(eqprop_trials, baseline_trials)
        
        if pairs:
            print(f"\n⚖️  Matched Comparisons ({len(pairs)} pairs, {self.matcher.strategy}):")
            eq_wins = sum(1 for p in pairs if p.performance_diff() > 0)
            bl_wins = sum(1 for p in pairs if p.performance_diff() < 0)
            ties = len(pairs) - eq_wins - bl_wins
            
            print(f"   EqProp wins: {eq_wins}/{len(pairs)}")
            print(f"   Baseline wins: {bl_wins}/{len(pairs)}")
            print(f"   Ties: {ties}/{len(pairs)}")
            
            avg_diff = np.mean([p.performance_diff() for p in pairs])
            avg_time_ratio = np.mean([p.time_ratio() for p in pairs])
            
            print(f"   Avg performance diff: {avg_diff:+.4f}")
            print(f"   Avg time ratio (EqProp/Baseline): {avg_time_ratio:.2f}x")
        
        # Pareto frontier
        all_trials = eqprop_trials + baseline_trials
        frontier = ParetoAnalyzer.pareto_frontier(all_trials, ["performance", "time"])
        
        print(f"\n🎯 Pareto Frontier (performance vs time):")
        eq_on_frontier = sum(1 for t in frontier if t.algorithm == "eqprop")
        bl_on_frontier = sum(1 for t in frontier if t.algorithm == "bp")
        print(f"   Total on frontier: {len(frontier)}")
        print(f"   EqProp: {eq_on_frontier}, Baseline: {bl_on_frontier}")
        
        for t in sorted(frontier, key=lambda x: x.performance, reverse=True)[:5]:
            marker = "🔋" if t.algorithm == "eqprop" else "⚡"
            print(f"   {marker} {t.performance:.4f} @ {t.cost.wall_time_seconds:.1f}s")
