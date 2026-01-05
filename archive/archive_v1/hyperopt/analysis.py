from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np
from collections import defaultdict
from .core import HyperOptTrial

# Import from existing statistics module
# Assuming statistics.py is in the root or accessible path
try:
    from statistics import StatisticalAnalyzer, ComparisonResult
except ImportError:
    # Fallback or need to adjust path if used from package
    import sys
    sys.path.append("..")
    from statistics import StatisticalAnalyzer, ComparisonResult

@dataclass
class MatchedPair:
    """A matched pair of EqProp and baseline trials."""
    eqprop_trial: HyperOptTrial
    baseline_trial: HyperOptTrial
    match_quality: float  # 0-1, how well matched
    match_criteria: str  # What was matched on
    
    def performance_diff(self) -> float:
        """EqProp performance - Baseline performance."""
        return self.eqprop_trial.performance - self.baseline_trial.performance
    
    def time_ratio(self) -> float:
        """Time ratio: EqProp / Baseline."""
        if self.baseline_trial.cost.wall_time_seconds > 0:
            return (self.eqprop_trial.cost.wall_time_seconds / 
                    self.baseline_trial.cost.wall_time_seconds)
        return 1.0


class TrialMatcher:
    """Match EqProp trials to baseline trials for fair comparison.
    
    Matching strategies:
    - time_matched: Similar training time
    - param_matched: Same parameter count
    - size_matched: Same model size (d_model)
    """
    
    def __init__(self, strategy: str = "time_matched", tolerance: float = 0.1):
        self.strategy = strategy
        self.tolerance = tolerance
    
    def match(self, eqprop_trials: List[HyperOptTrial],
              baseline_trials: List[HyperOptTrial]) -> List[MatchedPair]:
        """Find matched pairs between EqProp and baseline trials."""
        
        if self.strategy == "time_matched":
            return self._match_by_time(eqprop_trials, baseline_trials)
        elif self.strategy == "param_matched":
            return self._match_by_params(eqprop_trials, baseline_trials)
        elif self.strategy == "size_matched":
            return self._match_by_size(eqprop_trials, baseline_trials)
        else:
            raise ValueError(f"Unknown matching strategy: {self.strategy}")
    
    def _match_by_time(self, eqprop: List[HyperOptTrial],
                       baseline: List[HyperOptTrial]) -> List[MatchedPair]:
        """Match trials with similar training time."""
        pairs = []
        used_baseline = set()
        
        for eq in eqprop:
            if eq.status != "complete":
                continue
            
            best_match = None
            best_diff = float("inf")
            
            for bl in baseline:
                if bl.trial_id in used_baseline or bl.status != "complete":
                    continue
                
                time_diff = abs(eq.cost.wall_time_seconds - bl.cost.wall_time_seconds)
                relative_diff = time_diff / max(eq.cost.wall_time_seconds, 1)
                
                if relative_diff < self.tolerance and relative_diff < best_diff:
                    best_match = bl
                    best_diff = relative_diff
            
            if best_match:
                quality = 1.0 - best_diff
                pairs.append(MatchedPair(eq, best_match, quality, "time_matched"))
                used_baseline.add(best_match.trial_id)
        
        return pairs
    
    def _match_by_params(self, eqprop: List[HyperOptTrial],
                         baseline: List[HyperOptTrial]) -> List[MatchedPair]:
        """Match trials with similar parameter count."""
        pairs = []
        used_baseline = set()
        
        for eq in eqprop:
            if eq.status != "complete":
                continue
            
            best_match = None
            best_diff = float("inf")
            
            for bl in baseline:
                if bl.trial_id in used_baseline or bl.status != "complete":
                    continue
                
                param_diff = abs(eq.cost.param_count - bl.cost.param_count)
                relative_diff = param_diff / max(eq.cost.param_count, 1)
                
                if relative_diff < self.tolerance and relative_diff < best_diff:
                    best_match = bl
                    best_diff = relative_diff
            
            if best_match:
                quality = 1.0 - best_diff
                pairs.append(MatchedPair(eq, best_match, quality, "param_matched"))
                used_baseline.add(best_match.trial_id)
        
        return pairs
    
    def _match_by_size(self, eqprop: List[HyperOptTrial],
                       baseline: List[HyperOptTrial]) -> List[MatchedPair]:
        """Match trials with same model size (d_model)."""
        pairs = []
        
        # Group by d_model
        eq_by_size = defaultdict(list)
        bl_by_size = defaultdict(list)
        
        for eq in eqprop:
            if eq.status == "complete":
                d = eq.config.get("d_model", 128)
                eq_by_size[d].append(eq)
        
        for bl in baseline:
            if bl.status == "complete":
                d = bl.config.get("d_model", 128)
                bl_by_size[d].append(bl)
        
        # Match best from each size bucket
        for d_model in eq_by_size:
            if d_model not in bl_by_size:
                continue
            
            # Sort by performance
            eq_sorted = sorted(eq_by_size[d_model], 
                              key=lambda t: t.performance, reverse=True)
            bl_sorted = sorted(bl_by_size[d_model],
                              key=lambda t: t.performance, reverse=True)
            
            # Match best with best, second with second, etc.
            for eq, bl in zip(eq_sorted, bl_sorted):
                pairs.append(MatchedPair(eq, bl, 1.0, f"size_matched_d{d_model}"))
        
        return pairs


class ParetoAnalyzer:
    """Find Pareto-optimal configurations across multiple objectives."""
    
    @staticmethod
    def is_dominated(trial1: HyperOptTrial, trial2: HyperOptTrial,
                     objectives: List[str]) -> bool:
        """Check if trial1 is dominated by trial2.
        
        trial1 is dominated if trial2 is better or equal in all objectives
        and strictly better in at least one.
        """
        better_in_one = False
        
        for obj in objectives:
            val1 = ParetoAnalyzer._get_objective_value(trial1, obj)
            val2 = ParetoAnalyzer._get_objective_value(trial2, obj)
            
            # Higher is better for performance, lower is better for costs
            if obj == "performance":
                if val2 < val1:
                    return False  # trial2 worse in this objective
                if val2 > val1:
                    better_in_one = True
            else:  # cost objectives: lower is better
                if val2 > val1:
                    return False
                if val2 < val1:
                    better_in_one = True
        
        return better_in_one
    
    @staticmethod
    def _get_objective_value(trial: HyperOptTrial, obj: str) -> float:
        if obj == "performance":
            return trial.performance
        elif obj == "time":
            return trial.cost.wall_time_seconds
        elif obj == "memory":
            return trial.cost.peak_memory_mb
        elif obj == "params":
            return trial.cost.param_count
        elif obj == "iterations":
            return trial.cost.total_iterations
        else:
            return 0.0
    
    @staticmethod
    def pareto_frontier(trials: List[HyperOptTrial],
                       objectives: List[str] = None) -> List[HyperOptTrial]:
        """Find Pareto-optimal trials.
        
        Default objectives: maximize performance, minimize time.
        """
        if objectives is None:
            objectives = ["performance", "time"]
        
        # Filter to complete trials
        complete = [t for t in trials if t.status == "complete"]
        
        frontier = []
        for candidate in complete:
            dominated = False
            for other in complete:
                if other.trial_id == candidate.trial_id:
                    continue
                if ParetoAnalyzer.is_dominated(candidate, other, objectives):
                    dominated = True
                    break
            
            if not dominated:
                frontier.append(candidate)
        
        return frontier
