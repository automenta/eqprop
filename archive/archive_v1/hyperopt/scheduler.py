from typing import List, Dict, Any, Callable
import math
from .core import HyperOptTrial
from .evaluator import CostAwareEvaluator

class MultiFidelityScheduler:
    """Orchestrates multi-fidelity optimization (e.g., Successive Halving)."""
    
    def __init__(self, evaluator: CostAwareEvaluator, db, reduction_factor: int = 3):
        self.evaluator = evaluator
        self.db = db
        self.eta = reduction_factor
    
    def run_successive_halving(self, 
                               configs: List[Dict], 
                               algorithm: str,
                               task: str,
                               seeds: List[int],
                               min_epochs: int = 1,
                               max_epochs: int = 10,
                               callback: Callable = None):
        """Run Successive Halving Algorithm (SHA).
        
        1. Start with all configs running for min_epochs.
        2. Keep top 1/eta fraction of configs.
        3. Run survivors for eta * current_epochs.
        4. Repeat until max_epochs reached.
        """
        n_configs = len(configs)
        
        # Calculate number of rungs (stages)
        # max_epochs = min_epochs * (eta ^ s_max)
        s_max = int(math.log(max_epochs / min_epochs, self.eta))
        
        # Adjust max_epochs to exactly match the geometric verification series if needed, 
        # or just use it as a cap. We'll use the rungs to define epochs.
        
        current_configs = configs
        current_epochs = min_epochs
        
        print(f"\n🚀 Starting Successive Halving ({len(configs)} configs, {s_max+1} rungs)")
        
        for rung in range(s_max + 1):
            n_survivors = int(n_configs / (self.eta ** rung))
            if n_survivors < 1:
                n_survivors = 1
            
            print(f"\n🪜 Rung {rung}: Running {len(current_configs)} configs for {current_epochs} epochs")
            
            rung_results = []
            
            for i, cfg in enumerate(current_configs):
                # We average over seeds for decision making
                seed_perfs = []
                
                for seed in seeds:
                    # Construct ID that includes rung info to avoid collision or allow continuation?
                    # Ideally we resume the SAME trial ID implies adding more epochs.
                    # But our System stores 'complete' status.
                    # Implementation detail: We update the existing trial if it exists, or create new?
                    # Simpler for now: Separate trial IDs per rung or checking epoch count.
                    # Let's use suffix _r{rung} for distinct tracking in DB
                    
                    trial_id = f"{algorithm}_{task}_cfg{i}_s{seed}_r{rung}"
                    
                    trial = HyperOptTrial(
                        trial_id=trial_id,
                        algorithm=algorithm,
                        config=cfg,
                        task=task,
                        seed=seed
                    )
                    
                    if callback:
                        callback(f"Rung {rung} Trial {trial_id}")
                    
                    # Evaluate
                    trial = self.evaluator.evaluate(trial, epochs=current_epochs, show_progress=False)
                    self.db.add_trial(trial)
                    seed_perfs.append(trial.performance)
                
                avg_perf = sum(seed_perfs) / len(seed_perfs)
                rung_results.append((cfg, avg_perf))
            
            # Sort and select top candidates
            rung_results.sort(key=lambda x: x[1], reverse=True) # Higher is better
            
            best_perf = rung_results[0][1]
            print(f"   Best in Rung {rung}: {best_perf:.4f}")
            
            if rung < s_max:
                n_next = int(len(current_configs) / self.eta)
                if n_next < 1: n_next = 1
                
                current_configs = [x[0] for x in rung_results[:n_next]]
                current_epochs *= self.eta
                # Cap at max_epochs
                if current_epochs > max_epochs:
                    current_epochs = max_epochs
                
                print(f"   Promoting top {n_next} configs to {current_epochs} epochs")
            else:
                print("🏁 Optimization Complete")
                
        return rung_results[0][0] # Return best config
