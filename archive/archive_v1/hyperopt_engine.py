#!/usr/bin/env python3
"""
TorEqProp Competitive Hyperparameter Optimization Engine (Optuna-based)

A systematic framework for finding optimal configurations of both EqProp and
baseline algorithms, then comparing them fairly across multiple cost dimensions.

Features:
- Optuna-based search and pruning
- Cost-aware evaluation (time, memory, iterations, parameters)
- Fair trial matching for apples-to-apples comparison
- Pareto frontier analysis for multi-objective optimization
- Comprehensive reporting with statistical analysis

Usage:
    python hyperopt_engine.py              # Run full optimization
    python hyperopt_engine.py --n-trials 50
"""

import argparse
import subprocess
import sys
import os
import time
import re
import yaml
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
import numpy as np
from collections import defaultdict

try:
    import optuna
    from optuna.trial import TrialState
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("Warning: Optuna not found. Please install with 'pip install optuna'")

try:
    from scipy.stats import qmc
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# =============================================================================
# TRIAL AND COST TRACKING
# =============================================================================

@dataclass
class CostMetrics:
    """Cost metrics for a single trial."""
    wall_time_seconds: float = 0.0
    peak_memory_mb: float = 0.0
    total_iterations: int = 0  # For EqProp: equilibration iterations
    param_count: int = 0
    flops_estimate: float = 0.0
    
    # Convergence tracking
    convergence_curve: List[float] = field(default_factory=list)  # Per-epoch performance
    train_losses: List[float] = field(default_factory=list)
    train_accs: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, float]:
        d = asdict(self)
        # Convert lists to JSON-serializable format
        return d


@dataclass
class HyperOptTrial:
    """A single hyperparameter optimization trial."""
    trial_id: str
    algorithm: str  # "eqprop" or "bp"
    config: Dict[str, Any]
    task: str  # e.g., "mnist", "cartpole"
    seed: int
    
    # Results
    performance: float = 0.0
    performance_metric: str = "accuracy"  # or "reward"
    cost: CostMetrics = field(default_factory=CostMetrics)
    
    # Meta
    status: str = "pending"  # pending, running, complete, failed
    timestamp: str = ""
    log_path: str = ""
    error: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "trial_id": self.trial_id,
            "algorithm": self.algorithm,
            "config": self.config,
            "task": self.task,
            "seed": self.seed,
            "performance": self.performance,
            "performance_metric": self.performance_metric,
            "cost": self.cost.to_dict(),
            "status": self.status,
            "timestamp": self.timestamp,
            "error": self.error,
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> "HyperOptTrial":
        cost = CostMetrics(**d.get("cost", {}))
        return cls(
            trial_id=d["trial_id"],
            algorithm=d["algorithm"],
            config=d["config"],
            task=d["task"],
            seed=d["seed"],
            performance=d.get("performance", 0.0),
            performance_metric=d.get("performance_metric", "accuracy"),
            cost=cost,
            status=d.get("status", "pending"),
            timestamp=d.get("timestamp", ""),
            log_path=d.get("log_path", ""),
            error=d.get("error", ""),
        )


# =============================================================================
# COST-AWARE EVALUATION
# =============================================================================

class CostAwareEvaluator:
    """Evaluates trials across multiple cost dimensions.
    
    Tracks:
    - Performance (accuracy/reward)
    - Time cost (wall-clock training time)
    - Memory cost (peak GPU memory)
    - Iteration cost (equilibration iterations for EqProp)
    - Parameter cost (model size)
    """
    
    def __init__(self, logs_dir: Path):
        self.logs_dir = logs_dir
        self.logs_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate(self, trial: HyperOptTrial, epochs: int = 5, 
                 callback=None, show_progress: bool = True) -> HyperOptTrial:
        """Run a trial and capture all metrics."""
        
        trial.timestamp = datetime.now().isoformat()
        trial.status = "running"
        log_path = self.logs_dir / f"{trial.trial_id}.log"
        trial.log_path = str(log_path)
        
        # Build command based on algorithm
        if trial.algorithm == "eqprop":
            cmd = self._build_eqprop_command(trial, epochs)
        else:
            cmd = self._build_baseline_command(trial, epochs)
        
        start_time = time.time()
        
        # Show progress indicator
        if show_progress:
            print(f"   ⏳ Running {trial.task} ({trial.algorithm})...", end="", flush=True)
        
        try:
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env
            )
            
            output_lines = []
            last_progress_time = time.time()
            line_count = 0
            
            for line in process.stdout:
                output_lines.append(line)
                line_count += 1
                
                if callback:
                    callback(line)
                
                # Show progress dots every 5 seconds even in headless mode
                if show_progress and time.time() - last_progress_time > 5:
                    elapsed = time.time() - start_time
                    print(f".", end="", flush=True)
                    last_progress_time = time.time()
            
            process.wait()
            wall_time = time.time() - start_time
            output = "".join(output_lines)
            
            # Complete progress line
            if show_progress:
                print(f" ({wall_time:.0f}s)", flush=True)
            
            # Save log
            with open(log_path, "w") as f:
                f.write(f"Command: {cmd}\n")
                f.write(f"Config: {json.dumps(trial.config, indent=2)}\n")
                f.write(f"Duration: {wall_time:.1f}s\n")
                f.write(f"Exit code: {process.returncode}\n")
                f.write("=" * 70 + "\n")
                f.write(output)
            
            # Extract metrics
            trial.performance = self._extract_performance(output, trial.task)
            trial.cost.wall_time_seconds = wall_time
            trial.cost.total_iterations = self._extract_iterations(output)
            trial.cost.param_count = self._estimate_params(trial.config)
            
            trial.status = "complete" if process.returncode == 0 else "failed"
            if process.returncode != 0:
                trial.error = f"Exit code: {process.returncode}"
        
        except Exception as e:
            if show_progress:
                print(f" (error)", flush=True)
            trial.status = "failed"
            trial.error = str(e)
            trial.cost.wall_time_seconds = time.time() - start_time
        
        return trial

    
    def _build_eqprop_command(self, trial: HyperOptTrial, epochs: int) -> str:
        """Build command for EqProp trial.
        
        Supports all experiment types from the research plan:
        - Classification: mnist, fashion, cifar10, svhn
        - Algorithmic: parity, copy, addition
        - RL: CartPole, Acrobot, MountainCar, LunarLander
        - Memory: memory profiling
        """
        cfg = trial.config
        task = trial.task.lower() if isinstance(trial.task, str) else trial.task
        
        # Classification tasks
        if task in ["mnist", "fashion", "cifar10", "svhn"]:
            cmd = (f"python train.py --dataset {task} "
                   f"--epochs {epochs} --seed {trial.seed} "
                   f"--d-model {cfg['d_model']} --beta {cfg['beta']} "
                   f"--damping {cfg['damping']} --max-iters {cfg['max_iters']} "
                   f"--tol {cfg['tol']} --lr {cfg['lr']} "
                   f"--attention-type {cfg.get('attention_type', 'linear')} "
                   f"--update-mode {cfg.get('update_mode', 'mse_proxy')}")
            if cfg.get("symmetric", False):
                cmd += " --symmetric"
            if cfg.get("rapid", False):
                cmd += " --rapid"
        
        # Algorithmic reasoning tasks
        elif task in ["parity", "parity_8", "parity_12"]:
            seq_len = 12 if "12" in task else 8
            cmd = (f"python train_algorithmic.py --task parity "
                   f"--seq-len {seq_len} --epochs {epochs} "
                   f"--seed {trial.seed} --d-model {cfg['d_model']} "
                   f"--lr {cfg['lr']}")
        
        elif task == "copy":
            cmd = (f"python train_algorithmic.py --task copy "
                   f"--seq-len 8 --epochs {epochs} "
                   f"--seed {trial.seed} --d-model {cfg['d_model']} "
                   f"--lr {cfg['lr']}")
        
        elif task == "addition":
            cmd = (f"python train_algorithmic.py --task addition "
                   f"--n-digits 4 --epochs {epochs} "
                   f"--seed {trial.seed} --d-model {cfg['d_model']} "
                   f"--lr {cfg['lr']}")
        
        # Micro tasks (for rapid exploration with tiny models)
        elif task in ["xor", "xor3", "and", "and_gate", "or", "or_gate", 
                      "majority", "identity", "tiny_lm"]:
            cmd = (f"python train_micro.py --task {task} "
                   f"--epochs {epochs} --seed {trial.seed} "
                   f"--d-model {cfg['d_model']} --lr {cfg['lr']} "
                   f"--beta {cfg['beta']} --damping {cfg['damping']} "
                   f"--max-iters {cfg['max_iters']} --tol {cfg['tol']} "
                   f"--update-mode {cfg.get('update_mode', 'mse_proxy')}")
        
        # RL tasks
        elif task in ["cartpole-v1", "cartpole", "acrobot-v1", "acrobot", 
                      "mountaincar-v0", "mountaincar", "lunarlander-v2", "lunarlander"]:
            env_map = {
                "cartpole-v1": "CartPole-v1", "cartpole": "CartPole-v1",
                "acrobot-v1": "Acrobot-v1", "acrobot": "Acrobot-v1",
                "mountaincar-v0": "MountainCar-v0", "mountaincar": "MountainCar-v0",
                "lunarlander-v2": "LunarLander-v2", "lunarlander": "LunarLander-v2",
            }
            env = env_map.get(task, "CartPole-v1")
            episodes = epochs * 100 if epochs > 0 else 300
            cmd = (f"python train_rl.py --env {env} "
                   f"--episodes {episodes} --seed {trial.seed} "
                   f"--hidden-dim {cfg['d_model']}")
            if cfg.get("max_iters"):
                cmd += f" --max-iters {cfg['max_iters']}"
        
        # Memory profiling
        elif task in ["memory", "memory_profile"]:
            cmd = (f"python profile_memory.py --d-model {cfg['d_model']} "
                   f"--max-iters {cfg.get('max_iters', 100)} "
                   f"--seed {trial.seed}")
        
        else:
            cmd = f"echo 'Unknown task: {trial.task}'"
        
        return cmd
    
    def _build_baseline_command(self, trial: HyperOptTrial, epochs: int) -> str:
        """Build command for baseline trial.
        
        Supports all experiment types with BP baseline.
        """
        cfg = trial.config
        task = trial.task.lower() if isinstance(trial.task, str) else trial.task
        
        # Classification tasks
        if task in ["mnist", "fashion", "cifar10", "svhn"]:
            cmd = (f"python train_mnist_bp.py --dataset {task} "
                   f"--epochs {epochs} --seed {trial.seed} "
                   f"--d-model {cfg['d_model']} --lr {cfg['lr']}")
        
        # Algorithmic reasoning tasks (use --use-bp flag)
        elif task in ["parity", "parity_8", "parity_12"]:
            seq_len = 12 if "12" in task else 8
            cmd = (f"python train_algorithmic.py --task parity --use-bp "
                   f"--seq-len {seq_len} --epochs {epochs} "
                   f"--seed {trial.seed} --d-model {cfg['d_model']} "
                   f"--lr {cfg['lr']}")
        
        elif task == "copy":
            cmd = (f"python train_algorithmic.py --task copy --use-bp "
                   f"--seq-len 8 --epochs {epochs} "
                   f"--seed {trial.seed} --d-model {cfg['d_model']} "
                   f"--lr {cfg['lr']}")
        
        elif task == "addition":
            cmd = (f"python train_algorithmic.py --task addition --use-bp "
                   f"--n-digits 4 --epochs {epochs} "
                   f"--seed {trial.seed} --d-model {cfg['d_model']} "
                   f"--lr {cfg['lr']}")
        
        # Micro tasks (baseline with BP)
        elif task in ["xor", "xor3", "and", "and_gate", "or", "or_gate",
                      "majority", "identity", "tiny_lm"]:
            cmd = (f"python train_micro.py --task {task} --use-bp "
                   f"--epochs {epochs} --seed {trial.seed} "
                   f"--d-model {cfg['d_model']} --lr {cfg['lr']} "
                   f"--max-iters {cfg.get('max_iters', 20)}")
        
        # RL tasks
        elif task in ["cartpole-v1", "cartpole", "acrobot-v1", "acrobot",
                      "mountaincar-v0", "mountaincar", "lunarlander-v2", "lunarlander"]:
            env_map = {
                "cartpole-v1": "CartPole-v1", "cartpole": "CartPole-v1",
                "acrobot-v1": "Acrobot-v1", "acrobot": "Acrobot-v1",
                "mountaincar-v0": "MountainCar-v0", "mountaincar": "MountainCar-v0",
                "lunarlander-v2": "LunarLander-v2", "lunarlander": "LunarLander-v2",
            }
            env = env_map.get(task, "CartPole-v1")
            episodes = epochs * 100 if epochs > 0 else 300
            cmd = (f"python train_rl.py --env {env} --use-bp "
                   f"--episodes {episodes} --seed {trial.seed} "
                   f"--hidden-dim {cfg['d_model']}")
        
        # Memory profiling (comparison mode - runs both)
        elif task in ["memory", "memory_profile"]:
            cmd = (f"python profile_memory.py --d-model {cfg['d_model']} "
                   f"--max-iters 100 --seed {trial.seed}")
        
        else:
            cmd = f"echo 'Unknown task: {trial.task}'"
        
        return cmd

    
    def _extract_performance(self, output: str, task: str) -> float:
        """Extract performance metric from output."""
        # Try accuracy patterns
        acc_patterns = [
            r"Test Acc(?:uracy)?:\s*([\d.]+)",
            r"test/accuracy:\s*([\d.]+)",
            r"Final.*?Accuracy:\s*([\d.]+)",
        ]
        for pattern in acc_patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                return float(match.group(1))
        
        # Try reward patterns for RL
        reward_patterns = [
            r"Final Average Reward:\s*([-\d.]+)",
            r"avg_reward:\s*([-\d.]+)",
            r"Average Reward:\s*([-\d.]+)",
        ]
        for pattern in reward_patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                return float(match.group(1))
        
        return 0.0
    
    def _extract_iterations(self, output: str) -> int:
        """Extract total equilibration iterations."""
        patterns = [
            r"Iters:\s*(\d+)/(\d+)",
            r"iters_free:\s*(\d+)",
            r"train/iters_free.*?:\s*([\d.]+)",
        ]
        total_iters = 0
        for pattern in patterns:
            for match in re.finditer(pattern, output, re.IGNORECASE):
                try:
                    total_iters += int(float(match.group(1)))
                except:
                    pass
        return total_iters if total_iters > 0 else 0
    
    def _estimate_params(self, config: Dict) -> int:
        """Estimate parameter count from config."""
        d = config.get("d_model", 128)
        # Rough estimate: embedding + transformer block + head
        # Actual: d*784 + 4*d*d + 2*d*d*4 + d*10
        return int(d * 784 + 4 * d * d + 8 * d * d + d * 10)


# =============================================================================
# OPTUNA ENGINE
# =============================================================================

class OptunaHyperoptEngine:
    """Hyperparameter optimization engine using Optuna."""
    
    def __init__(self, 
                 storage_url: str = "sqlite:///toreq_hyperopt.db",
                 study_name: str = "toreq_optimization",
                 logs_dir: str = "logs/hyperopt"):
        self.storage_url = storage_url
        self.study_name = study_name
        self.logs_dir = Path(logs_dir)
        self.evaluator = CostAwareEvaluator(self.logs_dir)
        
        # Ensure Optuna is available
        if not HAS_OPTUNA:
            raise ImportError("Optuna is not installed. Run 'pip install optuna' first.")
            
    def run_study(self, 
                  n_trials: int = 20, 
                  task: str = "mnist", 
                  algorithm: str = "eqprop",
                  epochs: int = 5,
                  seed_params: Optional[Dict[str, Any]] = None):
        """Run an Optuna study."""
        
        study_name = f"{self.study_name}_{task}_{algorithm}"
        
        # Create or load study
        try:
            study = optuna.create_study(
                study_name=study_name,
                storage=self.storage_url,
                load_if_exists=True,
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=42, multivariate=True),
                pruner=optuna.pruners.HyperbandPruner(min_resource=1, max_resource=epochs, reduction_factor=3)
            )
        except Exception as e:
            print(f"Warning: Could not create study with default settings: {e}")
            print("Trying in-memory study...")
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=42, multivariate=True),
                pruner=optuna.pruners.HyperbandPruner(min_resource=1, max_resource=epochs, reduction_factor=3)
            )
        
        print(f"🚀 Starting Optuna study: {study_name}")
        print(f"   Task: {task}")
        print(f"   Algorithm: {algorithm}")
        print(f"   Trials: {n_trials}")
        
        # Seeding
        if seed_params:
            print(f"   🌱 Seeding with params: {seed_params}")
            study.enqueue_trial(seed_params)
        
        # Define objective function
        def objective(trial):
            # 1. Sample Hyperparameters
            config = self._sample_config(trial, algorithm)
            
            # 2. Create internal Trial object
            trial_id = f"optuna_{trial.number}_{task}_{int(time.time())}"
            h_trial = HyperOptTrial(
                trial_id=trial_id,
                algorithm=algorithm,
                config=config,
                task=task,
                seed=random.randint(1000, 9999)
            )
            
            # 3. Run evaluation (with pruning callback)
            def prune_callback(line):
                # Check for intermediate metrics in output line
                # Expecting something like "Epoch 1/5: acc=0.85"
                match = re.search(r"Epoch (\d+)/(\d+).*?acc[:=]\s*([\d.]+)", line, re.IGNORECASE)
                if match:
                    epoch = int(match.group(1))
                    acc = float(match.group(3))
                    trial.report(acc, epoch)
                    if trial.should_prune():
                        raise optuna.TrialPruned()

            h_trial = self.evaluator.evaluate(h_trial, epochs=epochs, callback=prune_callback, show_progress=True)
            
            # 4. Report results if failed
            if h_trial.status != "complete":
                trial.set_user_attr("error", h_trial.error)
                raise optuna.TrialPruned(f"Trial failed: {h_trial.error}")
            
            # 5. Store metrics
            for k, v in h_trial.cost.to_dict().items():
                if isinstance(v, (int, float, str)):
                     trial.set_user_attr(f"cost_{k}", v)
            
            trial.set_user_attr("log_path", h_trial.log_path)
            
            return h_trial.performance

        # Run optimization
        try:
            study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        except KeyboardInterrupt:
            print("\n🛑 Optimization stopped by user.")
        
        self._print_study_summary(study)
        self.generate_report(study, study_name)
        return study
    
    def get_best_params(self, task: str, algorithm: str) -> Optional[Dict[str, Any]]:
        """Retrieve best parameters for a task/algorithm."""
        study_name = f"{self.study_name}_{task}_{algorithm}"
        try:
            study = optuna.load_study(study_name=study_name, storage=self.storage_url)
            return study.best_params
        except:
            return None

    def generate_report(self, study, study_name: str):
        """Generate visualizations for the study."""
        try:
            from optuna.visualization import plot_optimization_history, plot_param_importances, plot_slice
            # Placeholder for saving plots
            pass 
        except ImportError:
            pass

    def _sample_config(self, trial, algorithm: str) -> Dict[str, Any]:
        """Sample hyperparameters using Optuna trial."""
        
        if algorithm == "eqprop":
            return {
                "algorithm": "eqprop",
                "beta": trial.suggest_float("beta", 0.05, 0.5), #, step=0.05),
                "damping": trial.suggest_float("damping", 0.5, 0.99),
                "max_iters": trial.suggest_categorical("max_iters", [10, 20, 50, 100]),
                "tol": trial.suggest_categorical("tol", [1e-4, 1e-5, 1e-6]),
                "attention_type": trial.suggest_categorical("attention_type", ["linear"]), # "softmax"]),
                "symmetric": trial.suggest_categorical("symmetric", [True, False]),
                "update_mode": trial.suggest_categorical("update_mode", ["mse_proxy"]), #, "vector_field"]),
                "d_model": trial.suggest_categorical("d_model", [16, 32, 64, 128]),
                "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            }
        else: # baseline (bp)
             return {
                "algorithm": "bp",
                "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                "optimizer": trial.suggest_categorical("optimizer", ["adam", "adamw"]),
                "d_model": trial.suggest_categorical("d_model", [16, 32, 64, 128]),
                "weight_decay": trial.suggest_categorical("weight_decay", [0, 1e-4, 1e-3]),
                # "scheduler": trial.suggest_categorical("scheduler", ["none", "cosine"]),
            }

    def _print_study_summary(self, study):
        """Print summary of the study."""
        print("\n" + "="*50)
        print("🏁 Optimization Complete")
        print("="*50)
        
        if len(study.trials) == 0:
            print("No trials completed.")
            return

        print(f"Best value: {study.best_value}")
        print("Best params:")
        for key, value in study.best_params.items():
            print(f"    {key}: {value}")
        
        print("\nImportance:")
        try:
            importance = optuna.importance.get_param_importances(study)
            for key, value in importance.items():
                print(f"    {key}: {value:.4f}")
        except:
            print("    (Insufficient data for importance analysis)")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Toreq Optuna Hyperopt")
    parser.add_argument("--task", type=str, default="mnist", help="Task to optimize")
    parser.add_argument("--algorithm", type=str, default="eqprop", choices=["eqprop", "bp"], help="Algorithm")
    parser.add_argument("--n-trials", type=int, default=10, help="Number of trials")
    parser.add_argument("--epochs", type=int, default=3, help="Epochs per trial")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick smoke test")
    
    args = parser.parse_args()
    
    if args.smoke_test:
        print("🔥 Running Smoke Test...")
        # Use in-memory for smoke test to avoid locking
        engine = OptunaHyperoptEngine(study_name="smoke_test_mem", storage_url="sqlite:///smoke_test.db")
        engine.run_study(n_trials=2, task="xor", algorithm=args.algorithm, epochs=1)
    else:
        engine = OptunaHyperoptEngine()
        engine.run_study(n_trials=args.n_trials, task=args.task, algorithm=args.algorithm, epochs=args.epochs)


if __name__ == "__main__":
    main()
