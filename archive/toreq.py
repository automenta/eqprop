#!/usr/bin/env python3
"""
TorEqProp Research CLI

Main entry point for all research operations:
- Smoke tests
- Comparison campaigns
- Claim validation
- Paper generation

Usage:
    python toreq.py --smoke-test           # Quick verification
    python toreq.py --campaign             # Full comparison
    python toreq.py --validate-claims      # Validate all publication claims
    python toreq.py --generate-paper NAME  # Generate paper from results
"""

import argparse
import sys
import subprocess
from pathlib import Path

import torch
from src.models import LoopedMLP, ToroidalMLP
from src.training import EqPropTrainer, get_mnist_loaders
from hyperopt import run_study


def smoke_test():
    print("Running Smoke Test...")
    model = LoopedMLP(784, 256, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    trainer = EqPropTrainer(model, optimizer)
    
    # Fake data
    x = torch.randn(4, 784)
    y = torch.tensor([0, 1, 2, 3])
    
    try:
        metrics = trainer.step(x, y)
        print(f"Smoke Test Passed! Metrics: {metrics}")
    except Exception as e:
        print(f"Smoke Test Failed: {e}")
        sys.exit(1)

def campaign(time_budget, epochs, dataset_size, task_name):
    print(f"Starting Comparison Campaign ({task_name}, {epochs} epochs, Budget: {time_budget}s, Data: {dataset_size})...")
    
    results = {}
    
    # 1. Backprop Baseline
    print("\n--- optimizing BackpropMLP ---")
    bp_stats = run_study("bp_study", "BackpropMLP", n_trials=50, time_budget=time_budget, epochs=epochs, dataset_size=dataset_size, task_name=task_name)
    results["BackpropMLP"] = bp_stats
    
    # 2. LoopedMLP (EqProp Baseline)
    print("\n--- optimizing LoopedMLP ---")
    looped_stats = run_study("looped_study", "LoopedMLP", n_trials=50, time_budget=time_budget, epochs=epochs, dataset_size=dataset_size, task_name=task_name)
    results["LoopedMLP"] = looped_stats
    
    # 3. ToroidalMLP (TEP)
    print("\n--- optimizing ToroidalMLP ---")
    toroidal_stats = run_study("toroidal_study", "ToroidalMLP", n_trials=50, time_budget=time_budget, epochs=epochs, dataset_size=dataset_size, task_name=task_name)
    results["ToroidalMLP"] = toroidal_stats

    # 4. ModernEqProp (Archived)
    print("\n--- optimizing ModernEqProp ---")
    modern_stats = run_study("modern_study", "ModernEqProp", n_trials=50, time_budget=time_budget, epochs=epochs, dataset_size=dataset_size, task_name=task_name)
    results["ModernEqProp"] = modern_stats
    
    print("\n\n=== CAMPAIGN RESULTS ===")
    print(f"{'Model':<15} | {'Best Acc':<10} | {'Time/Trial':<10} | {'Params':<8}")
    print("-" * 55)
    for model, (score, time_avg, params) in results.items():
        print(f"{model:<15} | {score:.4f}     | {time_avg:.2f}s      | {params:<8}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TorEqProp Research CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python toreq.py --smoke-test             Quick verification
  python toreq.py --campaign               Full model comparison
  python toreq.py --validate-claims        Validate publication claims
  python toreq.py --generate-paper spectral_normalization
  
For more info, see RESEARCH_STATUS.md
"""
    )
    
    # Core commands
    parser.add_argument("--smoke-test", action="store_true", 
                        help="Run quick verification")
    parser.add_argument("--campaign", action="store_true", 
                        help="Run full comparison campaign")
    
    # Campaign options
    parser.add_argument("--time-budget", type=int, default=60, 
                        help="Time in seconds per model for campaign")
    parser.add_argument("--epochs", type=int, default=3, 
                        help="Epochs per trial")
    parser.add_argument("--dataset-size", type=int, default=1000, 
                        help="Training set size (max 60000)")
    parser.add_argument("--task", type=str, default="mnist", 
                        help="Task: mnist, digits, cartpole, acrobot, tiny-lm")
    
    # Publication commands
    parser.add_argument("--validate-claims", action="store_true",
                        help="Validate all publication claims")
    parser.add_argument("--generate-paper", type=str, metavar="NAME",
                        help="Generate paper: spectral_normalization, beta_stability")
    parser.add_argument("--list-papers", action="store_true",
                        help="List available paper templates")
    
    args = parser.parse_args()
    
    if args.smoke_test:
        smoke_test()
    elif args.campaign:
        campaign(args.time_budget, args.epochs, args.dataset_size, args.task)
    elif args.validate_claims:
        # Run the claims validator
        script = Path(__file__).parent / "scripts" / "generate_paper.py"
        if script.exists():
            result = subprocess.run([sys.executable, str(script), "--validate-claims"])
            sys.exit(result.returncode)
        else:
            print("Error: scripts/generate_paper.py not found")
            print("Run: python scripts/generate_paper.py --validate-claims")
            sys.exit(1)
    elif args.generate_paper:
        # Generate a paper
        script = Path(__file__).parent / "scripts" / "generate_paper.py"
        if script.exists():
            result = subprocess.run([sys.executable, str(script), "--paper", args.generate_paper])
            sys.exit(result.returncode)
        else:
            print("Error: scripts/generate_paper.py not found")
            sys.exit(1)
    elif args.list_papers:
        # List available papers
        papers_dir = Path(__file__).parent / "papers"
        print("Available paper templates:")
        if papers_dir.exists():
            for path in papers_dir.glob("*_paper.md"):
                if "generated" not in path.name and "template" not in path.name:
                    name = path.stem.replace("_paper", "")
                    print(f"  - {name}")
        else:
            print("  (none found - check papers/ directory)")
    else:
        parser.print_help()
        print("\n" + "="*60)
        print("Quick Start:")
        print("  1. python toreq.py --smoke-test")
        print("  2. python toreq.py --validate-claims")
        print("  3. python toreq.py --generate-paper spectral_normalization")
        print("="*60)
