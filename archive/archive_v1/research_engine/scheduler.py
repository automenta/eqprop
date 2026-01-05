"""
Progressive experiment scheduler for the research engine.

Implements intelligent experiment selection with progressive validation:
- Start with cheap micro-experiments
- Promote promising configurations to higher tiers
- Prioritize unexplored parameter space regions
"""

import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional, Set
from collections import defaultdict

from .config import (
    ResearchConfig, TIERS, TIER_ORDER, ValidationTier, 
    FAST_PRESETS, DEFAULT_CONFIG
)
from .collector import ResultCollector, Trial


@dataclass
class ExperimentPriority:
    """Priority score for an experiment."""
    task: str
    algorithm: str
    config: Dict[str, Any]
    tier: str
    priority_score: float
    reason: str = ""


class ProgressiveScheduler:
    """
    Schedules experiments with progressive validation.
    
    Strategy:
    1. Exhaust micro-tier first (cheap validation)
    2. Promote promising configs to next tier
    3. Explore underrepresented regions of parameter space
    4. Balance exploration vs exploitation
    """
    
    def __init__(
        self,
        collector: Optional[ResultCollector] = None,
        config: ResearchConfig = DEFAULT_CONFIG,
    ):
        self.collector = collector or ResultCollector(config.output_dir)
        self.config = config
        
        # Track what we've run
        self._completed: Set[str] = set()
        self._promoted: Dict[str, str] = {}  # config_hash -> tier promoted to
        
        # Load existing trials
        self._load_existing()
    
    def _load_existing(self):
        """Load existing trials from collector."""
        trials = self.collector.get_trials(status="complete", limit=10000)
        for trial in trials:
            config_hash = self._hash_config(trial.config)
            self._completed.add(f"{trial.task}_{trial.algorithm}_{config_hash}")
    
    def _hash_config(self, config: Dict) -> str:
        """Create a hash for a config (for deduplication)."""
        # Use key params for hashing
        key_params = ["d_model", "lr", "beta", "damping", "epochs"]
        parts = []
        for k in sorted(key_params):
            if k in config:
                parts.append(f"{k}={config[k]}")
        return "_".join(parts) if parts else "default"
    
    def get_next_experiment(self) -> Tuple[str, str, Dict[str, Any], str]:
        """
        Get the next experiment to run.
        
        Returns:
            (task, algorithm, config, tier)
        """
        # 1. Check for pending micro-tier experiments
        micro_exp = self._get_micro_experiment()
        if micro_exp:
            return micro_exp
        
        # 2. Check for promotable configs
        promoted_exp = self._get_promoted_experiment()
        if promoted_exp:
            return promoted_exp
        
        # 3. Continue exploration with random sampling
        return self._get_exploration_experiment()
    
    def _get_micro_experiment(self) -> Optional[Tuple[str, str, Dict, str]]:
        """Get an unexplored micro-tier experiment."""
        tier = TIERS["micro"]
        
        for task, epochs in tier.tasks:
            for algo in ["eqprop", "bp"]:
                # Generate unexplored configs
                for _ in range(5):  # Try a few random configs
                    config = self._sample_config(task, algo, tier)
                    config_hash = self._hash_config(config)
                    key = f"{task}_{algo}_{config_hash}"
                    
                    if key not in self._completed:
                        return (task, algo, config, "micro")
        
        return None
    
    def _get_promoted_experiment(self) -> Optional[Tuple[str, str, Dict, str]]:
        """Get experiment for a configuration that should be promoted."""
        # Find configs that performed well in lower tiers
        trials = self.collector.get_trials(status="complete")
        
        # Group by config
        config_results: Dict[str, List[Trial]] = defaultdict(list)
        for trial in trials:
            config_hash = self._hash_config(trial.config)
            config_results[config_hash].append(trial)
        
        # Find promotable configs
        for config_hash, config_trials in config_results.items():
            best = max(config_trials, key=lambda t: t.performance)
            current_tier = best.tier
            
            # Check if should promote
            if current_tier not in TIER_ORDER:
                continue
            
            tier_idx = TIER_ORDER.index(current_tier)
            if tier_idx >= len(TIER_ORDER) - 1:
                continue  # Already at max tier
            
            tier_info = TIERS[current_tier]
            if best.performance >= tier_info.promotion_threshold:
                next_tier_name = TIER_ORDER[tier_idx + 1]
                
                # Check if already promoted
                if config_hash in self._promoted:
                    if self._promoted[config_hash] == next_tier_name:
                        continue
                
                # Generate experiment at higher tier
                next_tier = TIERS[next_tier_name]
                for task, epochs in next_tier.tasks:
                    config = best.config.copy()
                    config["epochs"] = epochs
                    config["d_model"] = next_tier.d_model
                    
                    key = f"{task}_{best.algorithm}_{self._hash_config(config)}"
                    if key not in self._completed:
                        self._promoted[config_hash] = next_tier_name
                        return (task, best.algorithm, config, next_tier_name)
        
        return None
    
    def _get_exploration_experiment(self) -> Tuple[str, str, Dict, str]:
        """Get a random exploration experiment."""
        # Pick a random tier (weighted towards lower tiers)
        tier_weights = [0.5, 0.3, 0.15, 0.05]
        tier_name = random.choices(TIER_ORDER, weights=tier_weights[:len(TIER_ORDER)])[0]
        tier = TIERS[tier_name]
        
        # Pick random task and algorithm
        task, epochs = random.choice(tier.tasks)
        algo = random.choice(["eqprop", "bp"])
        
        # Generate config
        config = self._sample_config(task, algo, tier)
        config["epochs"] = epochs
        
        return (task, algo, config, tier_name)
    
    def _sample_config(
        self,
        task: str,
        algorithm: str,
        tier: ValidationTier,
    ) -> Dict[str, Any]:
        """Sample a configuration for a task."""
        # Start with fast preset
        preset = FAST_PRESETS.get(task, FAST_PRESETS["xor"])
        config = preset.to_dict()
        
        # Adjust for tier
        config["d_model"] = tier.d_model
        
        # Sample hyperparameters
        config["lr"] = random.choice(self.config.lr_values)
        
        if algorithm == "eqprop":
            config["beta"] = random.choice(self.config.eqprop_beta_values)
            config["damping"] = random.choice(self.config.eqprop_damping_values)
            config["attention_type"] = random.choice(self.config.eqprop_attention_types)
            config["max_iters"] = random.choice(self.config.eqprop_max_iters)
        
        return config
    
    def mark_completed(self, task: str, algorithm: str, config: Dict):
        """Mark an experiment as completed."""
        config_hash = self._hash_config(config)
        self._completed.add(f"{task}_{algorithm}_{config_hash}")
    
    def get_progress(self) -> Dict[str, Any]:
        """Get progress across all tiers."""
        trials = self.collector.get_trials(status="complete")
        
        progress = {}
        for tier_name in TIER_ORDER:
            tier = TIERS[tier_name]
            tier_trials = [t for t in trials if t.tier == tier_name]
            
            total_expected = len(tier.tasks) * 2 * 3  # tasks * algos * min_seeds
            completed = len(tier_trials)
            
            progress[tier_name] = {
                "completed": completed,
                "expected": total_expected,
                "percent": min(100, int(completed / max(1, total_expected) * 100)),
                "tasks": [t for t, _ in tier.tasks],
            }
        
        return progress
    
    def get_promotable_configs(self) -> List[Dict[str, Any]]:
        """Get configs that are ready for promotion."""
        trials = self.collector.get_trials(status="complete")
        
        # Group by config
        config_results: Dict[str, List[Trial]] = defaultdict(list)
        for trial in trials:
            config_hash = self._hash_config(trial.config)
            config_results[config_hash].append(trial)
        
        promotable = []
        for config_hash, config_trials in config_results.items():
            best = max(config_trials, key=lambda t: t.performance)
            current_tier = best.tier
            
            if current_tier not in TIER_ORDER:
                continue
            
            tier_idx = TIER_ORDER.index(current_tier)
            if tier_idx >= len(TIER_ORDER) - 1:
                continue
            
            tier_info = TIERS[current_tier]
            if best.performance >= tier_info.promotion_threshold:
                promotable.append({
                    "config": best.config,
                    "algorithm": best.algorithm,
                    "current_tier": current_tier,
                    "next_tier": TIER_ORDER[tier_idx + 1],
                    "performance": best.performance,
                    "threshold": tier_info.promotion_threshold,
                })
        
        return promotable
    
    def get_unexplored_regions(self) -> List[Dict[str, Any]]:
        """Identify underexplored regions of parameter space."""
        trials = self.collector.get_trials(status="complete")
        
        # Count experiments per parameter value
        param_counts: Dict[str, Dict[Any, int]] = defaultdict(lambda: defaultdict(int))
        
        for trial in trials:
            for key, value in trial.config.items():
                if key in ["d_model", "lr", "beta", "damping", "epochs"]:
                    param_counts[key][value] += 1
        
        # Find underexplored values
        underexplored = []
        
        for param, values in param_counts.items():
            if not values:
                continue
            
            avg_count = sum(values.values()) / len(values)
            for value, count in values.items():
                if count < avg_count * 0.5:  # Less than half the average
                    underexplored.append({
                        "parameter": param,
                        "value": value,
                        "count": count,
                        "avg_count": avg_count,
                    })
        
        return sorted(underexplored, key=lambda x: x["count"])
