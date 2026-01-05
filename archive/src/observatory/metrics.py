"""
Observatory Metrics: Track and compute visualization metrics.

Metrics:
- Settling Time (T_relax): Steps until velocity < threshold
- Nudge Depth (D_nudge): How many layers show visible gradient signal
- Energy Consumption: Estimated based on lazy neuron skips
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class ObservatoryMetrics:
    """Track metrics for observatory visualization."""
    
    # Settling metrics
    settling_times: List[float] = field(default_factory=list)
    velocity_threshold: float = 1e-4
    
    # Nudge depth tracking
    nudge_depths: List[int] = field(default_factory=list)
    nudge_visibility_threshold: float = 0.01
    
    # Energy tracking (for lazy engine)
    total_updates: int = 0
    skipped_updates: int = 0
    
    def compute_settling_time(self, velocity_history: List[torch.Tensor]) -> int:
        """Compute steps until velocity drops below threshold.
        
        Args:
            velocity_history: List of velocity tensors at each step
            
        Returns:
            Number of steps to settle (or len if never settled)
        """
        for step, velocity in enumerate(velocity_history):
            mean_velocity = velocity.abs().mean().item()
            if mean_velocity < self.velocity_threshold:
                self.settling_times.append(step + 1)
                return step + 1
        
        self.settling_times.append(len(velocity_history))
        return len(velocity_history)
    
    def compute_nudge_depth(self, 
                           layer_nudges: Dict[str, torch.Tensor],
                           layer_order: List[str]) -> int:
        """Compute how many layers show visible nudge signal.
        
        Counts from the output layer backward until nudge magnitude
        drops below visibility threshold.
        
        Args:
            layer_nudges: Dict mapping layer name to nudge magnitude tensor
            layer_order: Ordered list of layer names (output first or last)
            
        Returns:
            Number of layers with visible nudge (D_nudge)
        """
        depth = 0
        # Reverse to count from output toward input
        for name in reversed(layer_order):
            if name not in layer_nudges:
                break
            nudge = layer_nudges[name]
            mean_nudge = nudge.abs().mean().item()
            if mean_nudge < self.nudge_visibility_threshold:
                break
            depth += 1
        
        self.nudge_depths.append(depth)
        return depth
    
    def record_update(self, was_skipped: bool = False):
        """Record a neuron update for energy tracking."""
        self.total_updates += 1
        if was_skipped:
            self.skipped_updates += 1
    
    @property
    def skip_ratio(self) -> float:
        """Fraction of updates skipped (lazy efficiency)."""
        if self.total_updates == 0:
            return 0.0
        return self.skipped_updates / self.total_updates
    
    @property
    def flop_savings_percent(self) -> float:
        """Estimated FLOP savings from lazy updates."""
        return self.skip_ratio * 100.0
    
    @property
    def mean_settling_time(self) -> float:
        """Average settling time across all measurements."""
        if not self.settling_times:
            return 0.0
        return sum(self.settling_times) / len(self.settling_times)
    
    @property
    def mean_nudge_depth(self) -> float:
        """Average nudge depth across all measurements."""
        if not self.nudge_depths:
            return 0.0
        return sum(self.nudge_depths) / len(self.nudge_depths)
    
    def reset(self):
        """Reset all metrics."""
        self.settling_times = []
        self.nudge_depths = []
        self.total_updates = 0
        self.skipped_updates = 0
    
    def summary(self) -> Dict[str, float]:
        """Return summary dict of all metrics."""
        return {
            'mean_settling_time': self.mean_settling_time,
            'mean_nudge_depth': self.mean_nudge_depth,
            'flop_savings_percent': self.flop_savings_percent,
            'total_updates': self.total_updates,
            'skipped_updates': self.skipped_updates,
        }
    
    def __repr__(self) -> str:
        return (f"ObservatoryMetrics(T_relax={self.mean_settling_time:.1f}, "
                f"D_nudge={self.mean_nudge_depth:.1f}, "
                f"FLOP_savings={self.flop_savings_percent:.1f}%)")
