import random
from typing import List, Dict, Any, Optional
from .search_spaces import SearchSpace

try:
    from scipy.stats import qmc
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

class Sampler:
    """Handles hyperparameter sampling strategies."""
    
    def __init__(self, rng: random.Random = None):
        self.rng = rng if rng else random.Random()

    def sample_configs(self, space: SearchSpace, n: int, 
                        strategy: str) -> List[Dict]:
        """Sample configurations from search space.
        
        Strategies:
        - grid: Full grid (or random subset if too large)
        - random: Pure random sampling
        - sobol: Quasi-random Sobol sequence (better coverage, requires scipy)
        - lhs: Latin Hypercube Sampling (even coverage, requires scipy)
        """
        if strategy == "grid":
            all_configs = space.grid()
            if len(all_configs) <= n:
                return all_configs
            return self.rng.sample(all_configs, n)
        
        elif strategy == "sobol":
            if not HAS_SCIPY:
                print("⚠️  scipy not available, falling back to random sampling")
                return [space.sample(self.rng) for _ in range(n)]
            return self._sobol_sample(space, n)
        
        elif strategy == "lhs":
            if not HAS_SCIPY:
                print("⚠️  scipy not available, falling back to random sampling")
                return [space.sample(self.rng) for _ in range(n)]
            return self._lhs_sample(space, n)
        
        elif strategy == "random":
            return [space.sample(self.rng) for _ in range(n)]
        
        else:
            # Default to random
            return [space.sample(self.rng) for _ in range(n)]

    def _sobol_sample(self, space: SearchSpace, n: int) -> List[Dict]:
        """Sample using Sobol quasi-random sequence for better coverage."""
        all_grid = space.grid()
        if not all_grid or len(all_grid) == 0:
            return [space.sample() for _ in range(n)]
        
        sample_config = all_grid[0]
        param_info = {}
        for key in sample_config.keys():
            if key == "algorithm":
                continue
            values = sorted(set(cfg[key] for cfg in all_grid if key in cfg))
            param_info[key] = values
        
        if not param_info:
            return [space.sample() for _ in range(n)]
        
        param_names = list(param_info.keys())
        # Sobol sequence requires d <= 21201
        if len(param_names) > 0:
            sampler = qmc.Sobol(d=len(param_names), scramble=True, seed=42)
            samples = sampler.random(n)
        else:
            return [space.sample() for _ in range(n)]
        
        configs = []
        for sample in samples:
            config = {"algorithm": sample_config["algorithm"]}
            for i, param_name in enumerate(param_names):
                values = param_info[param_name]
                idx = int(sample[i] * len(values))
                idx = min(idx, len(values) - 1)
                config[param_name] = values[idx]
            configs.append(config)
        
        return configs
    
    def _lhs_sample(self, space: SearchSpace, n: int) -> List[Dict]:
        """Latin Hypercube Sampling for better coverage."""
        if not HAS_SCIPY:
            return [space.sample() for _ in range(n)]
        
        all_grid = space.grid()
        if not all_grid:
            return [space.sample() for _ in range(n)]
        
        sample_config = all_grid[0]
        param_info = {}
        for key in sample_config.keys():
            if key == "algorithm":
                continue
            values = sorted(set(cfg[key] for cfg in all_grid if key in cfg))
            param_info[key] = values
        
        if not param_info:
            return [space.sample() for _ in range(n)]
        
        param_names = list(param_info.keys())
        sampler = qmc.LatinHypercube(d=len(param_names), seed=42)
        samples = sampler.random(n)
        
        configs = []
        for sample in samples:
            config = {"algorithm": sample_config["algorithm"]}
            for i, param_name in enumerate(param_names):
                values = param_info[param_name]
                idx = int(sample[i] * len(values))
                idx = min(idx, len(values) - 1)
                config[param_name] = values[idx]
            configs.append(config)
        
        return configs
