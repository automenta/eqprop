"""
Unified trial runner for TEP experiments.

Handles both TEP and BP with identical treatment:
- Hard timeout enforcement
- Consistent metric extraction
- Convergence curve tracking
- Failure rate monitoring
- Robust error handling with retries
"""

import time
import signal
import sys
import traceback
import numpy as np
from typing import Dict, Any, Optional, Callable, Tuple, List
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn

from .config import TrialResult, PhaseConfig
from .tasks import TaskSpec, get_task
from .objectives import (
    TrainingMetrics,
    train_and_evaluate_tep,
    train_and_evaluate_bp,
    count_parameters,
)


class TrialTimeoutError(Exception):
    """Raised when trial exceeds time limit."""
    pass


class TrialRunner:
    """Runs individual trials with timeout and metric collection.
    
    Ensures identical treatment for TEP and BP algorithms.
    Features:
    - Timeout enforcement with graceful handling
    - Automatic retry on transient failures
    - GPU memory management (cleanup between trials)
    - Detailed logging and statistics
    """
    
    def __init__(
        self,
        logs_dir: Path = Path("tep_logs"),
        device: str = "auto",
        max_retries: int = 2,
        cleanup_gpu: bool = True,
    ):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # GPU Performance optimizations
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True  # Auto-tune convolutions
            torch.set_float32_matmul_precision('high')  # Use TF32 on Ampere+
        
        self.max_retries = max_retries
        self.cleanup_gpu = cleanup_gpu
        
        # Statistics
        self.trials_run = 0
        self.trials_success = 0
        self.trials_failed = 0
        self.trials_timeout = 0
        self.trials_pruned = 0
        self.trials_retried = 0
        
        # Error tracking for debugging
        self.recent_errors: List[str] = []
    
    def run_trial(
        self,
        trial_id: str,
        algorithm: str,
        task_name: str,
        config: Dict[str, Any],
        seed: int = 42,
        max_epochs: int = 30,
        timeout_seconds: float = 90.0,
        report_callback: Optional[Callable[[int, float], bool]] = None,
    ) -> TrialResult:
        """Run a single trial with timeout enforcement.
        
        Args:
            trial_id: Unique identifier for this trial
            algorithm: "tep" or "bp"
            task_name: Name of the task (e.g., "xor", "digits_8x8")
            config: Hyperparameter configuration
            seed: Random seed
            max_epochs: Maximum training epochs
            timeout_seconds: Hard timeout for the trial
            report_callback: Optional callback(epoch, accuracy) -> continue
                            Returns False to prune the trial
        
        Returns:
            TrialResult with all metrics and status
        """
        self.trials_run += 1
        
        # Retry loop for transient failures
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                result = self._run_trial_impl(
                    trial_id=trial_id,
                    algorithm=algorithm,
                    task_name=task_name,
                    config=config,
                    seed=seed + attempt,  # Different seed per retry
                    max_epochs=max_epochs,
                    timeout_seconds=timeout_seconds,
                    report_callback=report_callback,
                )
                
                # Check if result is valid
                if result.status == "complete" and result.accuracy > 0:
                    self.trials_success += 1
                    return result
                elif result.status == "complete":
                    # Zero accuracy - might be numerical issue, try once more
                    if attempt < self.max_retries:
                        self.trials_retried += 1
                        continue
                    self.trials_success += 1
                    return result
                elif result.status == "timeout":
                    self.trials_timeout += 1
                    return result
                else:
                    # Failed - retry if attempts remain
                    last_error = result.error
                    if attempt < self.max_retries:
                        self.trials_retried += 1
                        continue
                    self.trials_failed += 1
                    return result
                    
            except Exception as e:
                last_error = f"{type(e).__name__}: {str(e)}"
                if attempt < self.max_retries:
                    self.trials_retried += 1
                    continue
                
                self.trials_failed += 1
                self._track_error(last_error)
                return TrialResult(
                    trial_id=trial_id,
                    algorithm=algorithm,
                    task=task_name,
                    seed=seed,
                    config=config,
                    status="failed",
                    error=last_error,
                )
            finally:
                # GPU cleanup between trials
                if self.cleanup_gpu and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Should not reach here
        self.trials_failed += 1
        return TrialResult(
            trial_id=trial_id,
            algorithm=algorithm,
            task=task_name,
            seed=seed,
            config=config,
            status="failed",
            error=last_error or "Unknown error after retries",
        )
    
    def _run_trial_impl(
        self,
        trial_id: str,
        algorithm: str,
        task_name: str,
        config: Dict[str, Any],
        seed: int,
        max_epochs: int,
        timeout_seconds: float,
        report_callback: Optional[Callable],
    ) -> TrialResult:
        """Internal implementation of trial execution."""
        
        # Set seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        start_time = time.time()
        
        try:
            # Get task specification
            task_spec = get_task(task_name)
            
            # Handle RL tasks differently
            if task_spec.task_type == "rl":
                result = self._run_rl_trial(
                    trial_id, algorithm, task_spec, config, seed,
                    max_epochs, timeout_seconds, report_callback
                )
            else:
                result = self._run_classification_trial(
                    trial_id, algorithm, task_spec, config, seed,
                    max_epochs, timeout_seconds, report_callback
                )
            
            return result
            
        except TrialTimeoutError as e:
            return TrialResult(
                trial_id=trial_id,
                algorithm=algorithm,
                task=task_name,
                seed=seed,
                config=config,
                wall_time_seconds=time.time() - start_time,
                status="timeout",
                error=str(e),
            )
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            self._track_error(error_msg)
            return TrialResult(
                trial_id=trial_id,
                algorithm=algorithm,
                task=task_name,
                seed=seed,
                config=config,
                wall_time_seconds=time.time() - start_time,
                status="failed",
                error=error_msg,
            )
    
    def _run_classification_trial(
        self,
        trial_id: str,
        algorithm: str,
        task_spec: TaskSpec,
        config: Dict[str, Any],
        seed: int,
        max_epochs: int,
        timeout_seconds: float,
        report_callback: Optional[Callable],
    ) -> TrialResult:
        """Run a classification trial (XOR, digits, MNIST)."""
        
        # Get data loaders
        batch_size = config.get("batch_size", task_spec.default_batch_size)
        
        # Robust loader creation with error handling
        try:
            train_loader = task_spec.get_train_loader(batch_size=batch_size, seed=seed)
            test_loader = task_spec.get_test_loader(batch_size=batch_size, seed=seed)
        except Exception as e:
            raise RuntimeError(f"Failed to create data loaders: {e}")
        
        if train_loader is None or test_loader is None:
            raise RuntimeError(f"Task {task_spec.name} returned None loaders")
        
        # Create model
        model, embedding, output_head, solver = self._create_model(
            task_spec, config, algorithm
        )
        
        # Create pruning callback that checks timeout
        start_time = time.time()
        
        def timeout_aware_callback(epoch: int, accuracy: float) -> bool:
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                raise TrialTimeoutError(f"Trial exceeded {timeout_seconds}s timeout at epoch {epoch}")
            
            # Forward to user callback
            if report_callback:
                return report_callback(epoch, accuracy)
            return True
        
        # Train and evaluate
        if algorithm == "tep":
            from src.trainer import EqPropTrainer
            
            beta = config.get("beta", 0.1)
            lr = config.get("lr", 1e-3)
            
            # Validate beta range
            if not (0.001 <= beta <= 1.0):
                beta = max(0.001, min(1.0, beta))
            
            trainer = EqPropTrainer(
                model, solver, output_head,
                beta=beta,
                lr=lr,
                update_mode="mse_proxy",
            )
            trainer.optimizer.add_param_group({'params': embedding.parameters()})
            
            metrics = train_and_evaluate_tep(
                model, embedding, output_head, solver, trainer,
                train_loader, test_loader, config,
                max_epochs=max_epochs,
                report_callback=timeout_aware_callback,
            )
        else:  # BP
            metrics = train_and_evaluate_bp(
                model, embedding, output_head, solver,
                train_loader, test_loader, config,
                max_epochs=max_epochs,
                report_callback=timeout_aware_callback,
            )
        
        return TrialResult(
            trial_id=trial_id,
            algorithm=algorithm,
            task=task_spec.name,
            seed=seed,
            config=config,
            accuracy=metrics.final_accuracy,
            wall_time_seconds=metrics.wall_time_seconds,
            param_count=metrics.param_count,
            convergence_steps=metrics.convergence_steps,
            convergence_curve=metrics.convergence_curve,
            final_loss=metrics.final_loss,
            status="complete",
        )
    
    def _run_rl_trial(
        self,
        trial_id: str,
        algorithm: str,
        task_spec: TaskSpec,
        config: Dict[str, Any],
        seed: int,
        max_epochs: int,
        timeout_seconds: float,
        report_callback: Optional[Callable],
    ) -> TrialResult:
        """Run an RL trial (CartPole, Acrobot).
        
        For now, delegates to existing train_rl.py via subprocess.
        TODO: Integrate directly for better metric collection.
        """
        import subprocess
        
        start_time = time.time()
        hidden_dim = config.get("hidden_units", 64)
        
        # Build command
        env_name = {
            "cartpole_v1": "CartPole-v1",
            "acrobot_v1": "Acrobot-v1",
        }.get(task_spec.name, "CartPole-v1")
        
        cmd = [
            sys.executable, "train_rl.py",
            "--env", env_name,
            "--episodes", str(max_epochs),
            "--seed", str(seed),
            "--hidden-dim", str(hidden_dim),
        ]
        
        if algorithm == "bp":
            cmd.append("--use-bp")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                cwd=str(Path(__file__).parent.parent),
            )
            
            # Extract metrics from output
            import re
            output = result.stdout
            
            reward_match = re.search(r"Final Average Reward:\s*([-\d.]+)", output)
            reward = float(reward_match.group(1)) if reward_match else 0.0
            
            wall_time = time.time() - start_time
            
            return TrialResult(
                trial_id=trial_id,
                algorithm=algorithm,
                task=task_spec.name,
                seed=seed,
                config=config,
                accuracy=reward,  # Use reward as "accuracy" for RL
                wall_time_seconds=wall_time,
                param_count=self._estimate_rl_params(hidden_dim, task_spec),
                status="complete" if result.returncode == 0 else "failed",
                error=result.stderr if result.returncode != 0 else "",
            )
            
        except subprocess.TimeoutExpired:
            raise TrialTimeoutError(f"RL trial exceeded {timeout_seconds}s timeout")
    
    def _create_model(
        self,
        task_spec: TaskSpec,
        config: Dict[str, Any],
        algorithm: str,
    ) -> Tuple[nn.Module, nn.Module, nn.Module, Any]:
        """Create model components from configuration.
        
        Returns: (model, embedding, output_head, solver)
        """
        from src.models import LoopedTransformerBlock
        from src.solver import EquilibriumSolver
        
        hidden_units = config.get("hidden_units", 64)
        n_layers = config.get("n_hidden_layers", 1)
        activation = config.get("activation", "tanh")
        
        # Validate hidden_units to avoid numerical issues
        hidden_units = max(4, min(512, hidden_units))
        
        n_heads = config.get("_valid_n_heads", 1)  # From constraint application
        d_ff = hidden_units * 4  # Standard transformer ratio
        
        # TEP-specific params (with defaults for BP)
        attention_type = config.get("attention_type", "linear")
        symmetric = config.get("symmetric", False)
        
        # Embedding layer with proper initialization
        embedding = nn.Linear(task_spec.input_dim, hidden_units).to(self.device)
        nn.init.xavier_uniform_(embedding.weight)
        nn.init.zeros_(embedding.bias)
        
        # Model - use sampled parameters!
        model = LoopedTransformerBlock(
            d_model=hidden_units,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=0.0,
            attention_type=attention_type,
            symmetric=symmetric,
        ).to(self.device)
        
        # Output head with proper initialization
        output_head = nn.Linear(hidden_units, task_spec.output_dim).to(self.device)
        nn.init.xavier_uniform_(output_head.weight)
        nn.init.zeros_(output_head.bias)
        
        # Solver with validated parameters
        if algorithm == "tep":
            eq_iters = config.get("eq_iters", 20)
            gamma = config.get("gamma", 0.9)
            tolerance = config.get("tolerance", 1e-4)
            # Clamp to valid ranges
            eq_iters = max(5, min(100, eq_iters))
            gamma = max(0.5, min(0.99, gamma))
            tolerance = max(1e-6, min(1e-2, tolerance))
        else:
            eq_iters = 20  # Fixed for BP
            gamma = 0.9
            tolerance = 1e-4
        
        solver = EquilibriumSolver(
            max_iters=eq_iters,
            tol=tolerance,
            damping=gamma,
        )
        
        return model, embedding, output_head, solver
    
    def _estimate_rl_params(self, hidden_dim: int, task_spec: TaskSpec) -> int:
        """Estimate parameter count for RL model."""
        # Policy network: input -> hidden -> hidden -> output
        params = task_spec.input_dim * hidden_dim + hidden_dim
        params += hidden_dim * hidden_dim + hidden_dim
        params += hidden_dim * task_spec.output_dim + task_spec.output_dim
        return params
    
    def _track_error(self, error: str):
        """Track recent errors for debugging."""
        self.recent_errors.append(error[:500])  # Truncate long errors
        if len(self.recent_errors) > 10:
            self.recent_errors.pop(0)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get runner statistics."""
        return {
            "trials_run": self.trials_run,
            "trials_success": self.trials_success,
            "trials_failed": self.trials_failed,
            "trials_timeout": self.trials_timeout,
            "trials_pruned": self.trials_pruned,
            "trials_retried": self.trials_retried,
            "success_rate": self.trials_success / max(1, self.trials_run),
            "recent_errors": self.recent_errors[-3:] if self.recent_errors else [],
        }
