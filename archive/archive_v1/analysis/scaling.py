import numpy as np
from typing import List, Dict, Callable
import json
from pathlib import Path

# Optional matplotlib import
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Handle both package and standalone imports
try:
    from ..hyperopt.core import HyperOptTrial
    from ..hyperopt.engine import HyperOptEngine
except ImportError:
    try:
        from hyperopt.core import HyperOptTrial
        from hyperopt.engine import HyperOptEngine
    except ImportError:
        from hyperopt_engine import HyperOptTrial, HyperOptEngine

class ScalingAnalyzer:
    """Automates the study of Neural Scaling Laws for EqProp.
    
    References:
    - Kaplan et al. (2020) "Scaling Laws for Neural Language Models"
    - Hoffmann et al. (2022) "Training Compute-Optimal Large Language Models"
    
    Tracks:
    - Test Loss L
    - Compute C (FLOPS/Time)
    - Parameters N
    - Dataset Size D
    
    Goal: Fit L(N) = (N_c / N)^alpha_N
    """
    
    def __init__(self, engine: HyperOptEngine, results_dir: str = "scaling_results"):
        self.engine = engine
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def run_scaling_sweep(self, task: str = "tiny_lm", 
                          sizes: List[int] = [16, 32, 64, 128],
                          epochs_per_grad: int = 1000):
        """Run a sweep over model sizes."""
        
        results = []
        
        print(f"📈 Starting Scaling Sweep for {task}...")
        
        for d_model in sizes:
            print(f"   Testing size d_model={d_model}...")
            
            # Use fixed optimal hyperparameters (found via hyperopt potentially)
            # For scaling laws, we usually keep hyperparams fixed or scale them known ways
            config = {
                "d_model": d_model,
                "beta": 0.1,
                "damping": 0.5,
                "max_iters": 20,
                "lr": 0.001
            }
            
            # Create a custom direct trial runner or use engine's infra
            # Here we wrap engine for simplicity
            trial = HyperOptTrial(
                trial_id=f"scaling_d{d_model}",
                algorithm="eqprop",
                config=config,
                task=task,
                seed=42
            )
            
            # Run
            trial = self.engine.evaluator.evaluate(trial, epochs=5) # Reduced for demo
            
            # Store data
            data_point = {
                "d_model": d_model,
                "params": trial.cost.param_count,
                "loss": 1.0 - trial.performance, # Assuming performance is acc, we want loss
                "time": trial.cost.wall_time_seconds,
                "compute": trial.cost.total_iterations * trial.cost.param_count # Proxy for FLOPS
            }
            results.append(data_point)
            
            # Quick log
            print(f"     -> Loss: {data_point['loss']:.4f}, Params: {data_point['params']}")
        
        # Save results
        with open(self.results_dir / f"scaling_{task}.json", "w") as f:
            json.dump(results, f, indent=2)
            
        return results

    def plot_scaling_laws(self, results: List[Dict]):
        """Generate component N vs Loss plot."""
        params = [r["params"] for r in results]
        losses = [r["loss"] for r in results]
        
        try:
            # Log-Log Plot
            plt.figure(figsize=(8, 6))
            plt.loglog(params, losses, 'o-', linewidth=2, markersize=8)
            plt.xlabel("Parameters (N)")
            plt.ylabel("Test Loss (L)")
            plt.title("EqProp Scaling Law")
            plt.grid(True, which="both", ls="-")
            
            # Fit line
            log_n = np.log(params)
            log_l = np.log(losses)
            slope, intercept = np.polyfit(log_n, log_l, 1)
            
            plt.text(params[0], losses[0], f"alpha = {-slope:.2f}", fontsize=12)
            
            path = self.results_dir / "scaling_plot.png"
            plt.savefig(path)
            print(f"📉 Scaling plot saved to {path}")
            
        except ImportError:
            print("matplotlib not installed, skipping plot.")

