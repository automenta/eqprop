"""
Time-aware experiment runner for the research engine.

Executes experiments with strict time budget enforcement and automatic
configuration reduction when experiments exceed limits.
"""

import subprocess
import sys
import time
import signal
import os
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass

from .config import ResearchConfig, FAST_PRESETS, DEFAULT_CONFIG
from .collector import Trial, CostMetrics, ResultCollector


@dataclass
class RunnerStats:
    """Statistics for the runner."""
    experiments_run: int = 0
    experiments_success: int = 0
    experiments_failed: int = 0
    experiments_timeout: int = 0
    total_time: float = 0.0
    configs_reduced: int = 0


class TimeAwareRunner:
    """Runs experiments with time budget enforcement."""
    
    def __init__(
        self,
        collector: Optional[ResultCollector] = None,
        config: ResearchConfig = DEFAULT_CONFIG,
        on_time_warning: Optional[Callable[[str], None]] = None,
    ):
        self.collector = collector or ResultCollector(config.output_dir)
        self.config = config
        self.on_time_warning = on_time_warning
        self.stats = RunnerStats()
        
        # Mapping of tasks to training scripts
        self.task_scripts = {
            "xor": "train_micro.py",
            "xor3": "train_micro.py",
            "and": "train_micro.py",
            "parity": "train_algorithmic.py",
            "mnist": "train.py",
            "fashion": "train.py",
            "cartpole": "train_rl.py",
            "cifar10": "train.py",
        }
    
    def run(
        self,
        task: str,
        algorithm: str,
        config: Dict[str, Any],
        seed: int = 42,
        tier: str = "micro",
        max_time: Optional[float] = None,
    ) -> Trial:
        """
        Run an experiment with time budget enforcement.
        
        Args:
            task: Task name (xor, mnist, etc.)
            algorithm: "eqprop" or "bp"
            config: Configuration dict
            seed: Random seed
            tier: Validation tier
            max_time: Max time in seconds (uses config default if None)
            
        Returns:
            Completed Trial object
        """
        max_time = max_time or self.config.max_experiment_time
        
        trial_id = f"{algorithm}_{task}_{seed}_{int(time.time() * 1000)}"
        
        trial = Trial(
            trial_id=trial_id,
            algorithm=algorithm,
            task=task,
            config=config,
            seed=seed,
            tier=tier,
            started_at=datetime.now().isoformat(),
        )
        
        start_time = time.time()
        
        try:
            # Build command
            cmd = self._build_command(task, algorithm, config, seed)
            
            # Execute with timeout
            result = self._execute_with_timeout(cmd, max_time)
            
            trial.cost.wall_time_seconds = time.time() - start_time
            
            if result["success"]:
                trial.status = "complete"
                trial.performance = result.get("performance", 0.0)
                trial.cost.param_count = result.get("params", 0)
                trial.cost.epochs_completed = config.get("epochs", 0)
                self.stats.experiments_success += 1
            else:
                trial.status = "failed"
                trial.error = result.get("error", "Unknown error")
                self.stats.experiments_failed += 1
                
        except TimeoutError:
            trial.status = "timeout"
            trial.error = f"Exceeded {max_time}s time budget"
            trial.cost.wall_time_seconds = time.time() - start_time
            self.stats.experiments_timeout += 1
            
            # Warn about time consumption
            self._warn_time_exceeded(task, trial.cost.wall_time_seconds, max_time)
            
            # Auto-reduce config if enabled
            if self.config.auto_reduce_on_timeout:
                reduced_config = self.config.get_reduced_config(config)
                self.stats.configs_reduced += 1
                # Note: don't store full config to avoid circular reference
                trial.config["_was_reduced"] = True
                trial.config.update(reduced_config)
                
        except Exception as e:
            trial.status = "failed"
            trial.error = str(e)
            trial.cost.wall_time_seconds = time.time() - start_time
            self.stats.experiments_failed += 1
        
        trial.completed_at = datetime.now().isoformat()
        self.stats.experiments_run += 1
        self.stats.total_time += trial.cost.wall_time_seconds
        
        # Save to collector
        if self.collector:
            self.collector.save_trial(trial)
        
        return trial
    
    def _build_command(
        self,
        task: str,
        algorithm: str,
        config: Dict[str, Any],
        seed: int,
    ) -> list:
        """Build the command to run the experiment."""
        script = self.task_scripts.get(task, "train_micro.py")
        
        cmd = [sys.executable, "-u", script]
        
        # Task selection
        if script == "train_micro.py":
            cmd.extend(["--task", task])
        elif script == "train_algorithmic.py":
            cmd.extend(["--task", task])
        elif script == "train_rl.py":
            cmd.extend(["--env", "CartPole-v1"])
        
        # Algorithm selection
        if algorithm == "bp":
            cmd.append("--use-bp")
        
        # Common config
        cmd.extend(["--seed", str(seed)])
        
        # Add config parameters
        if "epochs" in config:
            cmd.extend(["--epochs", str(config["epochs"])])
        if "d_model" in config:
            cmd.extend(["--d-model", str(config["d_model"])])
        if "lr" in config:
            cmd.extend(["--lr", str(config["lr"])])
        if "batch_size" in config:
            cmd.extend(["--batch-size", str(config["batch_size"])])
        if "max_iters" in config:
            cmd.extend(["--max-iters", str(config["max_iters"])])
        if "beta" in config and algorithm == "eqprop":
            cmd.extend(["--beta", str(config["beta"])])
        if "damping" in config:
            cmd.extend(["--damping", str(config["damping"])])
        # TODO: Add --attention-type support to training scripts
        # if "attention_type" in config and algorithm == "eqprop":
        #     cmd.extend(["--attention-type", str(config["attention_type"])])
        if "episodes" in config:
            cmd.extend(["--episodes", str(config["episodes"])])
        
        return cmd
    
    def _execute_with_timeout(
        self,
        cmd: list,
        timeout: float,
    ) -> Dict[str, Any]:
        """Execute command with timeout, return results."""
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=Path(__file__).parent.parent,  # toreq root
            )
            
            stdout, _ = process.communicate(timeout=timeout)
            
            if process.returncode == 0:
                performance = self._extract_performance(stdout)
                params = self._extract_params(stdout)
                return {
                    "success": True,
                    "performance": performance,
                    "params": params,
                    "output": stdout,
                }
            else:
                return {
                    "success": False,
                    "error": f"Exit code {process.returncode}: {stdout[-500:] if stdout else 'No output'}",
                    "output": stdout,
                }
                
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise TimeoutError(f"Command timed out after {timeout}s")
    
    def _extract_performance(self, output: str) -> float:
        """Extract performance metric from output."""
        patterns = [
            # Accuracy patterns
            r"Test Accuracy:\s*([\d.]+)%",
            r"test_acc[:\s]*([\d.]+)",
            r"Final Accuracy:\s*([\d.]+)",
            r"Accuracy:\s*([\d.]+)",
            # RL patterns
            r"Average Reward:\s*([\d.]+)",
            r"Avg Reward:\s*([\d.]+)",
            r"Mean Reward:\s*([\d.]+)",
            # Generic patterns
            r"Performance:\s*([\d.]+)",
            r"Score:\s*([\d.]+)",
            # Epoch-wise (take last)
            r"Epoch.*acc[:\s]*([\d.]+)",
        ]
        
        best_perf = 0.0
        for pattern in patterns:
            matches = re.findall(pattern, output, re.IGNORECASE)
            if matches:
                for match in matches:
                    try:
                        val = float(match)
                        # Convert percentage to decimal if > 1
                        if val > 1 and val <= 100:
                            val = val / 100
                        best_perf = max(best_perf, val)
                    except ValueError:
                        continue
        
        return best_perf
    
    def _extract_params(self, output: str) -> int:
        """Extract parameter count from output."""
        patterns = [
            r"Parameters:\s*([\d,]+)",
            r"Params:\s*([\d,]+)",
            r"Total params:\s*([\d,]+)",
            r"(\d+)\s*parameters",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                try:
                    return int(match.group(1).replace(",", ""))
                except ValueError:
                    continue
        
        return 0
    
    def _warn_time_exceeded(self, task: str, actual: float, limit: float):
        """Warn about time budget exceedance."""
        msg = f"⚠️ {task} took {actual:.1f}s (limit: {limit:.0f}s) - consider reducing epochs/d_model"
        
        if self.on_time_warning:
            self.on_time_warning(msg)
        else:
            print(msg)
    
    def run_batch(
        self,
        experiments: list,
        stop_on_failure: bool = False,
    ) -> list:
        """
        Run a batch of experiments.
        
        Args:
            experiments: List of (task, algorithm, config, seed) tuples
            stop_on_failure: Stop on first failure
            
        Returns:
            List of Trial objects
        """
        trials = []
        
        for task, algorithm, config, seed in experiments:
            trial = self.run(task, algorithm, config, seed)
            trials.append(trial)
            
            if stop_on_failure and trial.status != "complete":
                break
        
        return trials
    
    def sample_and_run(
        self,
        task: str,
        algorithm: str,
        tier: str = "micro",
    ) -> Trial:
        """Sample a random config and run experiment."""
        preset = FAST_PRESETS.get(task, FAST_PRESETS["xor"])
        config = preset.to_dict()
        
        # Add algorithm-specific params
        if algorithm == "eqprop":
            config["beta"] = random.choice(self.config.eqprop_beta_values)
            config["damping"] = random.choice(self.config.eqprop_damping_values)
        
        config["d_model"] = random.choice(self.config.d_model_values)
        config["lr"] = random.choice(self.config.lr_values)
        
        seed = random.randint(0, 9999)
        
        return self.run(task, algorithm, config, seed, tier)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get runner statistics."""
        return {
            "experiments_run": self.stats.experiments_run,
            "experiments_success": self.stats.experiments_success,
            "experiments_failed": self.stats.experiments_failed,
            "experiments_timeout": self.stats.experiments_timeout,
            "success_rate": self.stats.experiments_success / max(1, self.stats.experiments_run),
            "total_time": self.stats.total_time,
            "avg_time": self.stats.total_time / max(1, self.stats.experiments_run),
            "configs_reduced": self.stats.configs_reduced,
        }
