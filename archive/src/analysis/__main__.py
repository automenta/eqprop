#!/usr/bin/env python3
"""CLI for running analytical model validation.

Usage:
    python -m src.analysis --task xor --model LoopedMLP
    python -m src.analysis --task all --models "LoopedMLP,ToroidalMLP,ModernEqProp"
    python -m src.analysis --task identity --model LoopedMLP --verbose
"""

import argparse
import sys
import torch

# Add parent to path for src imports
sys.path.insert(0, '.')

from src.models import LoopedMLP, ToroidalMLP, ModernEqProp, GatedMLP
from src.analysis import (
    IterationAnalyzer, AnalysisReport,
    get_all_tasks, IdentityTask, LinearTask, XORTask, MemorizationTask, AttractorTask
)


MODEL_REGISTRY = {
    "LoopedMLP": lambda dim: LoopedMLP(dim, 64, 10, symmetric=True),
    "LoopedMLP_nonsym": lambda dim: LoopedMLP(dim, 64, 10, symmetric=False),
    "ToroidalMLP": lambda dim: ToroidalMLP(dim, 64, 10),
    "ModernEqProp": lambda dim: ModernEqProp(dim, 64, 10),
    "GatedMLP": lambda dim: GatedMLP(dim, 64, 10)
}

TASK_REGISTRY = {
    "identity": lambda: IdentityTask(dim=20),
    "linear": lambda: LinearTask(input_dim=20, output_dim=10),
    "xor": lambda: XORTask(n_bits=4),
    "memorization": lambda: MemorizationTask(n_patterns=10, input_dim=20, output_dim=10),
    "attractor": lambda: AttractorTask(n_attractors=3, input_dim=20)
}


def main():
    parser = argparse.ArgumentParser(description="Analytical Model Validator")
    parser.add_argument("--task", type=str, default="xor",
                       help="Task name or 'all' for all tasks")
    parser.add_argument("--model", type=str, default="LoopedMLP",
                       help="Model name")
    parser.add_argument("--models", type=str, default=None,
                       help="Comma-separated list of models for comparison")
    parser.add_argument("--n-samples", type=int, default=100,
                       help="Number of samples")
    parser.add_argument("--max-steps", type=int, default=50,
                       help="Maximum equilibrium steps")
    parser.add_argument("--verbose", action="store_true",
                       help="Print detailed output")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device (cpu/cuda)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Analytical Model Validator")
    print("=" * 60)
    
    # Get task(s)
    if args.task == "all":
        tasks = {name: factory() for name, factory in TASK_REGISTRY.items()}
    else:
        if args.task not in TASK_REGISTRY:
            print(f"Unknown task: {args.task}")
            print(f"Available: {list(TASK_REGISTRY.keys())}")
            return 1
        tasks = {args.task: TASK_REGISTRY[args.task]()}
    
    # Get model(s)
    if args.models:
        model_names = [m.strip() for m in args.models.split(",")]
    else:
        model_names = [args.model]
    
    # Create comparison table
    results = []
    
    for task_name, task in tasks.items():
        print(f"\n## Task: {task_name}")
        print(f"   {task.description}")
        print("-" * 50)
        
        for model_name in model_names:
            if model_name not in MODEL_REGISTRY:
                print(f"  Unknown model: {model_name}")
                continue
            
            # Create model with matching input dim
            model = MODEL_REGISTRY[model_name](task.input_dim)
            
            # Run analysis
            analyzer = IterationAnalyzer(
                model, device=args.device, max_steps=args.max_steps
            )
            
            try:
                report = analyzer.analyze_task(task, args.n_samples, validate_theory=True)
                
                # Print summary
                conv_icon = "✓" if report.trajectory.converged else "✗"
                theory_icon = "✓" if (report.theoretical and 
                                       report.theoretical.energy_descent_valid and
                                       report.theoretical.contraction_valid) else "✗"
                
                lipschitz = f"{report.theoretical.lipschitz_constant:.3f}" if report.theoretical else "N/A"
                conv_steps = report.trajectory.convergence_step or "N/C"
                
                print(f"  {model_name:20s} | Conv: {conv_icon} ({conv_steps:>3}) | "
                      f"Energy↓: {report.aggregate_metrics.energy_monotonic} | "
                      f"L={lipschitz:>5} | "
                      f"Acc: {report.task_accuracy:.2%}")
                
                if args.verbose:
                    print(report.to_markdown())
                
                results.append(report.to_dict())
                
            except Exception as e:
                print(f"  {model_name:20s} | ERROR: {e}")
    
    # Export results
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
    
    print("\n" + "=" * 60)
    print("Analysis Complete")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
