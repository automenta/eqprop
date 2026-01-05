import time
import os
import subprocess
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from .core import HyperOptTrial, CostMetrics

# Try to import tqdm
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

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
            print(f"   ⏳ Running", end="", flush=True)
        
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
                   f"--attention-type {cfg['attention_type']} "
                   f"--update-mode {cfg['update_mode']}")
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
                   f"--update-mode {cfg['update_mode']}")
        
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
        return total_iters if total_iters > 0 else 1  # Default 1 for BP
    
    def _estimate_params(self, config: Dict) -> int:
        """Estimate parameter count from config."""
        d = config.get("d_model", 128)
        # Rough estimate: embedding + transformer block + head
        # Actual: d*784 + 4*d*d + 2*d*d*4 + d*10
        return int(d * 784 + 4 * d * d + 8 * d * d + d * 10)


class TimeBudgetEvaluator(CostAwareEvaluator):
    """Evaluator that matches trials by time budget for fair comparison.
    
    Instead of fixing epochs, this gives both algorithms the same wall-clock
    time budget. This ensures fair comparison when algorithms have different
    per-epoch costs (e.g., EqProp vs BP).
    """
    
    def __init__(self, logs_dir: Path, time_budget_seconds: float = 60.0):
        super().__init__(logs_dir)
        self.time_budget = time_budget_seconds
    
    def evaluate(self, trial: HyperOptTrial, epochs: int = None,
                 callback=None, show_progress: bool = True) -> HyperOptTrial:
        """Run trial within time budget instead of fixed epochs.
        
        Dynamically estimates epochs based on time per epoch and budget.
        """
        if epochs is None:
            # Estimate epochs from time budget
            # Start with 1 epoch to measure time, then extrapolate
            epochs = self._estimate_epochs_for_budget(trial)
        
        return super().evaluate(trial, epochs, callback, show_progress)
    
    def _estimate_epochs_for_budget(self, trial: HyperOptTrial) -> int:
        """Estimate how many epochs fit in time budget."""
        # Heuristics based on task and model size
        d_model = trial.config.get("d_model", 128)
        task = trial.task.lower()
        
        # Rough estimates (seconds per epoch)
        if task in ["xor", "xor3", "and", "or", "majority", "identity"]:
            time_per_epoch = 1.0 * (d_model / 16)  # ~1s for d=16
        elif task == "tiny_lm":
            time_per_epoch = 2.0 * (d_model / 16)
        elif task == "mnist":
            time_per_epoch = 10.0 * (d_model / 64)
        elif task in ["parity", "copy", "addition"]:
            time_per_epoch = 5.0 * (d_model / 64)
        elif task in ["cartpole", "acrobot", "mountaincar", "lunarlander"]:
            time_per_epoch = 3.0 * (d_model / 64)  # Converted from episodes
        else:
            time_per_epoch = 10.0
        
        # Additional factor for algorithm
        if trial.algorithm == "eqprop":
            time_per_epoch *= 0.5  # EqProp typically faster
        
        epochs = max(1, int(self.time_budget / time_per_epoch))
        return epochs
