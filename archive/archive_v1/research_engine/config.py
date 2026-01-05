"""
Configuration management for the research engine.

Defines experiment presets, validation tiers, and global settings.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path


@dataclass
class ExperimentPreset:
    """Ultra-compact configuration for fast exploration."""
    epochs: int = 2
    d_model: int = 8
    max_iters: int = 5
    batch_size: int = 32
    lr: float = 1e-3
    subset: Optional[int] = None  # For datasets like MNIST, use subset
    episodes: Optional[int] = None  # For RL tasks
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "epochs": self.epochs,
            "d_model": self.d_model,
            "max_iters": self.max_iters,
            "batch_size": self.batch_size,
            "lr": self.lr,
        }
        if self.subset is not None:
            result["subset"] = self.subset
        if self.episodes is not None:
            result["episodes"] = self.episodes
        return result


# Fast presets targeting ~15 seconds per experiment
FAST_PRESETS: Dict[str, ExperimentPreset] = {
    "xor": ExperimentPreset(epochs=2, d_model=8, max_iters=3, batch_size=16),
    "xor3": ExperimentPreset(epochs=2, d_model=8, max_iters=3, batch_size=16),
    "and": ExperimentPreset(epochs=2, d_model=8, max_iters=3, batch_size=16),
    "parity": ExperimentPreset(epochs=3, d_model=16, max_iters=5, batch_size=32),
    "mnist": ExperimentPreset(epochs=1, d_model=32, max_iters=8, batch_size=128, subset=2000),
    "fashion": ExperimentPreset(epochs=1, d_model=32, max_iters=8, batch_size=128, subset=2000),
    "cartpole": ExperimentPreset(epochs=1, d_model=16, max_iters=3, episodes=3),
}


@dataclass
class ValidationTier:
    """Definition of a validation tier for progressive validation."""
    name: str
    tasks: List[Tuple[str, int]]  # (task_name, epochs)
    promotion_threshold: float     # Minimum score to promote to next tier
    max_time_per_trial: float      # Maximum seconds per trial
    d_model: int = 16              # Default model size for this tier
    
    def get_config_for_task(self, task: str) -> Dict[str, Any]:
        """Get configuration for a specific task in this tier."""
        for t, epochs in self.tasks:
            if t == task:
                preset = FAST_PRESETS.get(task, ExperimentPreset())
                config = preset.to_dict()
                config["epochs"] = epochs
                config["d_model"] = self.d_model
                return config
        return {}


# Progressive validation tiers
TIERS: Dict[str, ValidationTier] = {
    "micro": ValidationTier(
        name="micro",
        #tasks=[("xor", 3), ("xor3", 3), ("and", 3)],
        tasks=[("mnist", 3)],
        promotion_threshold=0.6,
        max_time_per_trial=20.0,
        d_model=8,
    ),
    "small": ValidationTier(
        name="small",
        tasks=[("parity", 5), ("mnist", 3), ("cartpole", 10)],
        promotion_threshold=0.7,
        max_time_per_trial=30.0,
        d_model=32,
    ),
    "medium": ValidationTier(
        name="medium",
        tasks=[("mnist", 10), ("fashion", 10), ("cartpole", 25)],
        promotion_threshold=0.8,
        max_time_per_trial=120.0,
        d_model=64,
    ),
    "large": ValidationTier(
        name="large",
        tasks=[("mnist", 50), ("fashion", 50), ("cifar10", 20)],
        promotion_threshold=0.9,
        max_time_per_trial=600.0,
        d_model=128,
    ),
}

TIER_ORDER = ["micro", "small", "medium", "large"]


@dataclass
class ResearchConfig:
    """Global configuration for research engine."""
    
    # Output directories
    output_dir: Path = field(default_factory=lambda: Path("research_output"))
    
    # Time budgets
    max_experiment_time: float = 15.0  # seconds for fast experiments (targeting 0.25 min)
    total_budget_minutes: float = 60.0  # total research time
    target_time_per_experiment: float = 0.5  # minutes (15 seconds)
    
    # Dashboard settings
    dashboard_refresh_rate: float = 2.0  # seconds between updates
    show_all_parameters: bool = True     # Full transparency
    
    # Experiment settings
    default_seeds: List[int] = field(default_factory=lambda: [42, 123, 456])
    min_seeds_for_stats: int = 3
    
    # Search space (compact for speed)
    eqprop_beta_values: List[float] = field(default_factory=lambda: [0.15, 0.2, 0.22, 0.25, 0.3, 0.4])
    eqprop_damping_values: List[float] = field(default_factory=lambda: [0.8, 0.9])
    eqprop_attention_types: List[str] = field(default_factory=lambda: ["linear", "softmax"])
    eqprop_max_iters: List[int] = field(default_factory=lambda: [3, 5, 8, 10, 20])
    d_model_values: List[int] = field(default_factory=lambda: [4, 8, 16, 32])
    lr_values: List[float] = field(default_factory=lambda: [1e-3, 2e-3])
    
    # Auto-reduction settings (when experiments exceed time budget)
    auto_reduce_on_timeout: bool = True
    reduction_factors: Dict[str, float] = field(default_factory=lambda: {
        "epochs": 0.5,
        "d_model": 0.5,
        "max_iters": 0.5,
    })
    
    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_reduced_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Reduce config when experiment exceeds time budget."""
        reduced = config.copy()
        for key, factor in self.reduction_factors.items():
            if key in reduced:
                if isinstance(reduced[key], int):
                    reduced[key] = max(1, int(reduced[key] * factor))
                elif isinstance(reduced[key], float):
                    reduced[key] = reduced[key] * factor
        return reduced


# Default configuration
DEFAULT_CONFIG = ResearchConfig()
