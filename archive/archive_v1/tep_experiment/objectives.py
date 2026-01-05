"""
Multi-objective evaluation for TEP experiments.

Computes all 4 objectives:
1. accuracy: Final validation accuracy (maximize)
2. wall_time: Total training wall-time in seconds (minimize)
3. param_count: Number of trainable parameters (minimize)
4. convergence_steps: Steps to reach 90% of best accuracy (minimize)

Enhanced Features:
- NaN/Inf handling for numerical stability
- Gradient clipping to prevent explosions
- Early stopping on divergence
- Comprehensive metric validation
"""

from typing import Dict, Any, List, Tuple, Optional
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field

from .config import TrialResult


# =============================================================================
# PARAMETER COUNTING
# =============================================================================

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_parameters_from_config(config: Dict[str, Any], input_dim: int, output_dim: int) -> int:
    """Estimate parameter count from configuration.
    
    Used when we don't have the actual model (e.g., for pruned trials).
    """
    n_layers = config.get("n_hidden_layers", 1)
    hidden = config.get("hidden_units", 64)
    
    # Embedding: input -> hidden
    params = input_dim * hidden + hidden  # weights + bias
    
    # Hidden layers (transformer blocks)
    # Rough estimate: 4 * hidden^2 per layer (Q, K, V, out projections)
    # Plus FFN: 2 * hidden * (4*hidden)
    params_per_layer = 4 * hidden * hidden + 2 * hidden * 4 * hidden
    params += n_layers * params_per_layer
    
    # Output head
    params += hidden * output_dim + output_dim
    
    return int(params)


# =============================================================================
# NUMERICAL STABILITY
# =============================================================================

def is_valid_number(x: float) -> bool:
    """Check if a number is valid (not NaN or Inf)."""
    return not (math.isnan(x) or math.isinf(x))


def safe_accuracy(correct: int, total: int) -> float:
    """Compute accuracy safely, handling edge cases."""
    if total == 0:
        return 0.0
    acc = correct / total
    return acc if is_valid_number(acc) else 0.0


def check_model_health(model: nn.Module) -> Tuple[bool, str]:
    """Check if model weights are healthy (no NaN/Inf).
    
    Returns:
        (is_healthy, message)
    """
    for name, param in model.named_parameters():
        if param.data is None:
            return False, f"Parameter {name} is None"
        if torch.isnan(param.data).any():
            return False, f"Parameter {name} contains NaN"
        if torch.isinf(param.data).any():
            return False, f"Parameter {name} contains Inf"
    return True, "Model healthy"


def clip_gradients(model: nn.Module, max_norm: float = 1.0) -> float:
    """Clip gradients to prevent explosion.
    
    Returns:
        Total gradient norm before clipping
    """
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


# =============================================================================
# CONVERGENCE SPEED
# =============================================================================

def compute_convergence_steps(
    convergence_curve: List[float],
    threshold_fraction: float = 0.9,
) -> int:
    """Compute steps to reach threshold fraction of best performance.
    
    Args:
        convergence_curve: List of performance values per step/epoch
        threshold_fraction: Fraction of best value to reach (default 0.9)
        
    Returns:
        Number of steps to reach threshold (or len(curve) if never reached)
    """
    if not convergence_curve:
        return 0
    
    # Filter out invalid values
    valid_curve = [v for v in convergence_curve if is_valid_number(v)]
    if not valid_curve:
        return 0
    
    best_value = max(valid_curve)
    if best_value <= 0:
        return len(convergence_curve)
    
    threshold = threshold_fraction * best_value
    
    for step, value in enumerate(convergence_curve):
        if is_valid_number(value) and value >= threshold:
            return step + 1  # 1-indexed
    
    return len(convergence_curve)


# =============================================================================
# TRAINING LOOP WITH METRICS COLLECTION
# =============================================================================

@dataclass
class TrainingMetrics:
    """Metrics collected during training."""
    final_accuracy: float = 0.0
    wall_time_seconds: float = 0.0
    param_count: int = 0
    convergence_steps: int = 0
    convergence_curve: List[float] = field(default_factory=list)
    final_loss: float = 0.0
    epochs_completed: int = 0
    early_stopped: bool = False
    diverged: bool = False
    
    def is_valid(self) -> bool:
        """Check if metrics are valid."""
        return (
            is_valid_number(self.final_accuracy) and
            is_valid_number(self.wall_time_seconds) and
            is_valid_number(self.final_loss) and
            self.final_accuracy >= 0
        )


def train_and_evaluate_tep(
    model: nn.Module,
    embedding: nn.Module,
    output_head: nn.Module,
    solver,
    trainer,
    train_loader,
    test_loader,
    config: Dict[str, Any],
    max_epochs: int = 30,
    report_callback=None,
    grad_clip: float = 1.0,
    divergence_threshold: float = 100.0,
) -> TrainingMetrics:
    """Train a TEP model and collect all metrics.
    
    Args:
        model: Equilibrium model
        embedding: Input embedding layer
        output_head: Classification head
        solver: EquilibriumSolver
        trainer: EqPropTrainer
        train_loader: Training data loader
        test_loader: Test data loader
        config: Hyperparameter configuration
        max_epochs: Maximum training epochs
        report_callback: Optional callback(epoch, accuracy) for pruning
        grad_clip: Maximum gradient norm
        divergence_threshold: Loss threshold for early stopping
        
    Returns:
        TrainingMetrics with all 4 objectives
    """
    device = next(model.parameters()).device
    
    # Count parameters
    param_count = (
        count_parameters(model) + 
        count_parameters(embedding) + 
        count_parameters(output_head)
    )
    
    # Check initial model health
    healthy, msg = check_model_health(model)
    if not healthy:
        return TrainingMetrics(
            param_count=param_count,
            diverged=True,
        )
    
    # Training loop
    convergence_curve = []
    start_time = time.time()
    epochs_completed = 0
    diverged = False
    
    for epoch in range(max_epochs):
        # Training
        model.train()
        embedding.train()
        output_head.train()
        
        total_loss = 0.0
        n_batches = 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            x_emb = embedding(data).unsqueeze(0)
            
            try:
                metrics = trainer.train_step(x_emb, target)
                batch_loss = metrics.get("loss", 0.0)
                
                # Check for divergence
                if not is_valid_number(batch_loss) or batch_loss > divergence_threshold:
                    diverged = True
                    break
                
                total_loss += batch_loss
                n_batches += 1
                
            except RuntimeError as e:
                # Handle CUDA OOM or other errors
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                diverged = True
                break
        
        if diverged:
            break
        
        # Compute average loss
        avg_loss = total_loss / max(1, n_batches)
        
        # Check model health after training epoch
        healthy, msg = check_model_health(model)
        if not healthy:
            diverged = True
            break
        
        # Evaluation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                x_emb = embedding(data).unsqueeze(0)
                h0 = torch.zeros_like(x_emb)
                
                try:
                    h_fixed, _ = solver.solve(model, h0, x_emb)
                    y_pred = output_head(h_fixed.mean(dim=0))
                    correct += (y_pred.argmax(-1) == target).sum().item()
                    total += target.size(0)
                except Exception:
                    # Skip batch on error
                    continue
        
        test_acc = safe_accuracy(correct, total)
        convergence_curve.append(test_acc)
        epochs_completed = epoch + 1
        
        # Report for pruning
        if report_callback:
            try:
                should_continue = report_callback(epoch, test_acc)
                if not should_continue:
                    break
            except Exception:
                break  # Treat callback errors as stop signals
    
    wall_time = time.time() - start_time
    
    # Compute final metrics
    valid_accs = [a for a in convergence_curve if is_valid_number(a)]
    final_accuracy = max(valid_accs) if valid_accs else 0.0
    convergence_steps = compute_convergence_steps(convergence_curve)
    
    return TrainingMetrics(
        final_accuracy=final_accuracy,
        wall_time_seconds=wall_time,
        param_count=param_count,
        convergence_steps=convergence_steps,
        convergence_curve=convergence_curve,
        final_loss=avg_loss if not diverged else float('inf'),
        epochs_completed=epochs_completed,
        early_stopped=epochs_completed < max_epochs and not diverged,
        diverged=diverged,
    )


def train_and_evaluate_bp(
    model: nn.Module,
    embedding: nn.Module,
    output_head: nn.Module,
    solver,
    train_loader,
    test_loader,
    config: Dict[str, Any],
    max_epochs: int = 30,
    report_callback=None,
    grad_clip: float = 1.0,
    divergence_threshold: float = 100.0,
) -> TrainingMetrics:
    """Train a BP model and collect all metrics.
    
    Uses standard backpropagation for comparison.
    Enhanced with gradient clipping and divergence detection.
    """
    device = next(model.parameters()).device
    
    # Count parameters
    param_count = (
        count_parameters(model) + 
        count_parameters(embedding) + 
        count_parameters(output_head)
    )
    
    # Check initial model health
    healthy, msg = check_model_health(model)
    if not healthy:
        return TrainingMetrics(param_count=param_count, diverged=True)
    
    # Setup optimizer
    all_params = (
        list(model.parameters()) + 
        list(embedding.parameters()) + 
        list(output_head.parameters())
    )
    
    optimizer_name = config.get("optimizer", "adam")
    lr = config.get("lr", 1e-3)
    weight_decay = config.get("weight_decay", 0.0)
    
    # Clamp learning rate
    lr = max(1e-6, min(0.1, lr))
    
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(all_params, lr=lr)
    
    # Training loop
    convergence_curve = []
    start_time = time.time()
    epochs_completed = 0
    diverged = False
    avg_loss = 0.0
    
    for epoch in range(max_epochs):
        # Training
        model.train()
        embedding.train()
        output_head.train()
        
        total_loss = 0.0
        n_batches = 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            try:
                x_emb = embedding(data).unsqueeze(0)
                h0 = torch.zeros_like(x_emb)
                h_fixed, _ = solver.solve(model, h0, x_emb)
                
                y_pred = output_head(h_fixed.mean(dim=0))
                loss = F.cross_entropy(y_pred, target)
                
                # Check for invalid loss
                if not is_valid_number(loss.item()) or loss.item() > divergence_threshold:
                    diverged = True
                    break
                
                loss.backward()
                
                # Gradient clipping
                clip_gradients(model, grad_clip)
                clip_gradients(embedding, grad_clip)
                clip_gradients(output_head, grad_clip)
                
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                diverged = True
                break
        
        if diverged:
            break
        
        avg_loss = total_loss / max(1, n_batches)
        
        # Check model health
        healthy, msg = check_model_health(model)
        if not healthy:
            diverged = True
            break
        
        # Evaluation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                
                try:
                    x_emb = embedding(data).unsqueeze(0)
                    h0 = torch.zeros_like(x_emb)
                    h_fixed, _ = solver.solve(model, h0, x_emb)
                    y_pred = output_head(h_fixed.mean(dim=0))
                    correct += (y_pred.argmax(-1) == target).sum().item()
                    total += target.size(0)
                except Exception:
                    continue
        
        test_acc = safe_accuracy(correct, total)
        convergence_curve.append(test_acc)
        epochs_completed = epoch + 1
        
        # Report for pruning
        if report_callback:
            try:
                should_continue = report_callback(epoch, test_acc)
                if not should_continue:
                    break
            except Exception:
                break
    
    wall_time = time.time() - start_time
    
    valid_accs = [a for a in convergence_curve if is_valid_number(a)]
    final_accuracy = max(valid_accs) if valid_accs else 0.0
    convergence_steps = compute_convergence_steps(convergence_curve)
    
    return TrainingMetrics(
        final_accuracy=final_accuracy,
        wall_time_seconds=wall_time,
        param_count=param_count,
        convergence_steps=convergence_steps,
        convergence_curve=convergence_curve,
        final_loss=avg_loss if not diverged else float('inf'),
        epochs_completed=epochs_completed,
        early_stopped=epochs_completed < max_epochs and not diverged,
        diverged=diverged,
    )


# =============================================================================
# OBJECTIVE FUNCTION FOR OPTUNA
# =============================================================================

def compute_objectives(metrics: TrainingMetrics) -> Tuple[float, float, float, float]:
    """Convert TrainingMetrics to Optuna objectives.
    
    Returns: (accuracy, wall_time, param_count, convergence_steps)
    
    Optuna handles maximization (accuracy) vs minimization (others)
    via study directions.
    
    Handles invalid metrics by returning dominated values.
    """
    if not metrics.is_valid() or metrics.diverged:
        # Return dominated values for invalid trials
        return (0.0, 1e6, 1e9, 1e6)
    
    return (
        metrics.final_accuracy,
        metrics.wall_time_seconds,
        float(metrics.param_count),
        float(max(1, metrics.convergence_steps)),
    )


def metrics_to_trial_result(
    trial_id: str,
    algorithm: str,
    task: str,
    seed: int,
    config: Dict[str, Any],
    metrics: TrainingMetrics,
    status: str = "complete",
    error: str = "",
) -> TrialResult:
    """Convert TrainingMetrics to TrialResult."""
    # Determine status based on metrics
    if metrics.diverged:
        status = "failed"
        error = "Training diverged"
    elif not metrics.is_valid():
        status = "failed"
        error = "Invalid metrics"
    
    return TrialResult(
        trial_id=trial_id,
        algorithm=algorithm,
        task=task,
        seed=seed,
        config=config,
        accuracy=metrics.final_accuracy,
        wall_time_seconds=metrics.wall_time_seconds,
        param_count=metrics.param_count,
        convergence_steps=metrics.convergence_steps,
        convergence_curve=metrics.convergence_curve,
        final_loss=metrics.final_loss if is_valid_number(metrics.final_loss) else 0.0,
        status=status,
        error=error,
    )
