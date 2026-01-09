"""
Central configuration for EqProp Trainer.
"""

from dataclasses import dataclass

@dataclass
class TrainerConfig:
    """Global configuration for the trainer."""
    epochs: int = 3  # Centralized default for all experiments (baseline & trials)
    quick_mode: bool = True
    max_epoch_time: float = 20.0 # Strict pruning threshold (seconds per epoch)
    task: str = 'shakespeare'

# Global instance
GLOBAL_CONFIG = TrainerConfig()
