"""
Derived metrics and efficiency analysis for TEP experiments.

Computes combination metrics that reveal trade-offs:
- Learning power (accuracy / time)
- Parameter efficiency (accuracy / params)
- Convergence efficiency
- Overall efficiency score
"""

from typing import Dict, Any, List, Tuple
from dataclasses import dataclass


@dataclass
class DerivedMetrics:
    """Derived metrics for analysis."""
    # Efficiency metrics
    learning_power: float = 0.0  # accuracy / wall_time (higher = better)
    param_efficiency: float = 0.0  # accuracy / log10(params) (higher = better)
    convergence_efficiency: float = 0.0  # accuracy / convergence_steps
    
    # Combined score (geometric mean of normalized metrics)
    efficiency_score: float = 0.0
    
    # Time-normalized metrics
    accuracy_per_second: float = 0.0
    params_per_accuracy: float = 0.0  # Lower is better
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "learning_power": self.learning_power,
            "param_efficiency": self.param_efficiency,
            "convergence_efficiency": self.convergence_efficiency,
            "efficiency_score": self.efficiency_score,
            "accuracy_per_second": self.accuracy_per_second,
            "params_per_accuracy": self.params_per_accuracy,
        }


def compute_derived_metrics(
    accuracy: float,
    wall_time: float,
    param_count: int,
    convergence_steps: int,
) -> DerivedMetrics:
    """Compute all derived metrics from base objectives.
    
    Args:
        accuracy: Final test accuracy (0-1)
        wall_time: Training time in seconds
        param_count: Number of parameters
        convergence_steps: Steps to 90% of best
        
    Returns:
        DerivedMetrics with computed values
    """
    import math
    
    # Avoid division by zero
    wall_time = max(0.1, wall_time)
    param_count = max(1, param_count)
    convergence_steps = max(1, convergence_steps)
    accuracy = max(0.0, min(1.0, accuracy))
    
    # Learning power: how much accuracy per second
    learning_power = accuracy / wall_time
    
    # Parameter efficiency: accuracy per log-parameter
    # Use log scale since params vary by orders of magnitude
    param_efficiency = accuracy / math.log10(param_count + 1)
    
    # Convergence efficiency: accuracy achieved per convergence step
    convergence_efficiency = accuracy / convergence_steps
    
    # Simple time-normalized metric
    accuracy_per_second = accuracy / wall_time
    
    # Inverse metric (lower is better)
    params_per_accuracy = param_count / max(0.01, accuracy)
    
    # Combined efficiency score (geometric mean of normalized metrics)
    # Normalize each metric to 0-1 range roughly
    norm_learning = min(1.0, learning_power / 0.1)  # 0.1 acc/sec is very good
    norm_param = min(1.0, param_efficiency)  # Already roughly 0-1
    norm_conv = min(1.0, convergence_efficiency / 0.05)  # 0.05 is good
    
    efficiency_score = (norm_learning * norm_param * norm_conv) ** (1/3)
    
    return DerivedMetrics(
        learning_power=learning_power,
        param_efficiency=param_efficiency,
        convergence_efficiency=convergence_efficiency,
        efficiency_score=efficiency_score,
        accuracy_per_second=accuracy_per_second,
        params_per_accuracy=params_per_accuracy,
    )


def format_metric(value: float, name: str) -> str:
    """Format a derived metric for display."""
    if name == "params_per_accuracy":
        return f"{value:.0f} params/acc"
    elif "efficiency" in name or "power" in name:
        return f"{value:.4f}"
    elif "per_second" in name:
        return f"{value:.4f} acc/s"
    else:
        return f"{value:.4f}"


def compare_derived_metrics(
    tep_metrics: DerivedMetrics,
    bp_metrics: DerivedMetrics,
) -> Dict[str, str]:
    """Compare derived metrics between TEP and BP.
    
    Returns dict with winner for each metric.
    """
    return {
        "learning_power": "TEP" if tep_metrics.learning_power > bp_metrics.learning_power else "BP",
        "param_efficiency": "TEP" if tep_metrics.param_efficiency > bp_metrics.param_efficiency else "BP",
        "convergence_efficiency": "TEP" if tep_metrics.convergence_efficiency > bp_metrics.convergence_efficiency else "BP",
        "efficiency_score": "TEP" if tep_metrics.efficiency_score > bp_metrics.efficiency_score else "BP",
    }


def summarize_config(config: Dict[str, Any], algorithm: str) -> str:
    """Create human-readable summary of config.
    
    Shows most important hyperparameters.
    """
    parts = []
    
    # Shared params
    parts.append(f"layers={config.get('n_hidden_layers', 1)}")
    parts.append(f"hidden={config.get('hidden_units', '?')}")
    parts.append(f"act={config.get('activation', '?')}")
    parts.append(f"lr={config.get('lr', '?'):.4f}")
    parts.append(f"bs={config.get('batch_size', '?')}")
    
    # Algorithm-specific
    if algorithm == "tep":
        parts.append(f"β={config.get('beta', '?'):.3f}")
        parts.append(f"γ={config.get('gamma', '?'):.3f}")
        parts.append(f"eq_iters={config.get('eq_iters', '?')}")
        parts.append(f"tol={config.get('tolerance', '?'):.0e}")
        if "attention_type" in config:
            parts.append(f"attn={config.get('attention_type', '?')}")
        if "symmetric" in config:
            parts.append(f"sym={config.get('symmetric', '?')}")
    else:  # BP
        parts.append(f"opt={config.get('optimizer', '?')}")
        wd = config.get('weight_decay', 0)
        if wd > 0:
            parts.append(f"wd={wd:.0e}")
    
    return ", ".join(parts)
