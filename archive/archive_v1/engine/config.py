"""Discovery configuration for TorEqProp experiments."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path
import yaml


@dataclass
class HypothesisConfig:
    """Configuration for a testable hypothesis."""
    name: str
    description: str
    test_method: str
    success_criteria: str
    priority: int = 1  # 1=highest
    

@dataclass
class DiscoveryConfig:
    """Configuration for a discovery campaign.
    
    Guides research towards publishable results with scientific rigor.
    """
    
    # Campaign identity
    name: str = "TorEqProp Discovery"
    description: str = "Systematic exploration of EqProp advantages"
    
    # Patience budget (time limits)
    max_total_time_hours: float = 24.0
    max_trial_time_seconds: float = 300.0
    
    # Search space
    d_model_options: List[int] = field(default_factory=lambda: [8, 16, 32, 64, 128])
    beta_options: List[float] = field(default_factory=lambda: [0.18, 0.20, 0.22, 0.24, 0.26])
    damping_options: List[float] = field(default_factory=lambda: [0.7, 0.8, 0.9])
    lr_options: List[float] = field(default_factory=lambda: [1e-3, 2e-3, 5e-3])
    
    # Tasks to explore (ordered by speed)
    tasks: List[str] = field(default_factory=lambda: [
        # Micro tasks (seconds)
        "xor", "and", "xor3", "majority", "identity", "tiny_lm",
        # Small tasks (minutes)
        "parity", "copy",
        # Medium tasks (minutes)
        "mnist", "fashion",
        # RL tasks (varies)
        "cartpole", "acrobot",
    ])
    
    # Hypotheses to test
    hypotheses: List[Dict[str, str]] = field(default_factory=lambda: [
        {
            "name": "speed_advantage",
            "description": "EqProp achieves 90% of BP accuracy at 2x+ speed",
            "test_method": "matched_time_comparison",
            "success_criteria": ">2x speedup at >90% relative accuracy",
        },
        {
            "name": "rl_superiority", 
            "description": "EqProp outperforms BP on RL tasks",
            "test_method": "direct_comparison",
            "success_criteria": "EqProp > BP on ≥2 environments",
        },
        {
            "name": "optimal_beta",
            "description": "β=0.22 is optimal for transformer training",
            "test_method": "beta_sweep",
            "success_criteria": "Clear peak with p<0.05",
        },
        {
            "name": "beta_annealing_fails",
            "description": "β-annealing causes training collapse",
            "test_method": "fixed_vs_annealing",
            "success_criteria": "Annealing collapses, fixed stable",
        },
    ])
    
    # Experimental rigor
    min_seeds: int = 3
    significance_level: float = 0.05
    min_trials_per_config: int = 3
    
    # Output
    output_dir: str = "results"
    log_dir: str = "logs"
    
    @classmethod
    def from_yaml(cls, path: str) -> "DiscoveryConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def to_yaml(self, path: str):
        """Save configuration to YAML file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)
    
    def get_micro_tasks(self) -> List[str]:
        """Get only micro tasks for rapid exploration."""
        return ["xor", "and", "xor3", "majority", "identity", "tiny_lm"]
    
    def get_classification_tasks(self) -> List[str]:
        """Get classification tasks."""
        return ["mnist", "fashion", "cifar10"]
    
    def get_rl_tasks(self) -> List[str]:
        """Get RL tasks."""
        return ["cartpole", "acrobot", "lunarlander"]


# Default rapid exploration config
RAPID_CONFIG = DiscoveryConfig(
    name="Rapid Exploration",
    max_total_time_hours=1.0,
    max_trial_time_seconds=60.0,
    d_model_options=[8, 16, 32],
    tasks=["xor", "xor3", "tiny_lm", "parity"],
    min_seeds=2,
)

# Default full campaign config
FULL_CONFIG = DiscoveryConfig(
    name="Full Discovery Campaign",
    max_total_time_hours=24.0,
    min_seeds=5,
)
