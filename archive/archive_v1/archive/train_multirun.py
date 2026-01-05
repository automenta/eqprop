"""Multi-seed validation script for TorEqProp robustness evaluation."""

import torch
import subprocess
import json
from pathlib import Path
import numpy as np


def run_training(seed: int, config_args: dict) -> dict:
    """Run single training run with given seed.
    
    Args:
        seed: Random seed
        config_args: Dictionary of config arguments
        
    Returns:
        Dictionary with results
    """
    # Build command
    cmd = ["python", "train.py", "--seed", str(seed)]
    
    # Add config arguments
    for key, value in config_args.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key.replace('_', '-')}")
        else:
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])
    
    print(f"\n{'='*70}")
    print(f"Running seed {seed}")
    print(f"{'='*70}")
    
    # Run training
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Parse results from stdout
    lines = result.stdout.split('\n')
    test_accs = []
    for line in lines:
        if "Test Acc:" in line:
            # Extract test accuracy
            parts = line.split("Test Acc:")
            if len(parts) > 1:
                acc_str = parts[1].strip().split()[0]
                test_accs.append(float(acc_str))
    
    final_acc = test_accs[-1] if test_accs else 0.0
    
    # Parse best accuracy from final summary
    for line in lines:
        if "Best test accuracy:" in line:
            final_acc = float(line.split(":")[-1].strip())
            break
    
    return {
        "seed": seed,
        "final_test_acc": final_acc,
        "all_test_accs": test_accs,
        "stdout": result.stdout,
        "stderr": result.stderr
    }


def main():
    """Run multi-seed validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-seed validation for TorEqProp")
    
    # Seeds
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5],
                        help="Seeds to run")
    
    # Model config
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    
    # Training config
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--beta-anneal", action="store_true")
    parser.add_argument("--damping", type=float, default=0.8)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--epochs", type=int, default=10)
    
    # Other
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--output", type=str, default="multirun_results.json",
                        help="Output file for results")
    
    args = parser.parse_args()
    
    # Prepare config
    config_args = {
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "d_ff": args.d_ff,
        "dropout": args.dropout,
        "beta": args.beta,
        "beta_anneal": args.beta_anneal,
        "damping": args.damping,
        "lr": args.lr,
        "epochs": args.epochs,
        "compile": args.compile
    }
    
    # Run all seeds
    results = []
    for seed in args.seeds:
        result = run_training(seed, config_args)
        results.append(result)
        
        # Save log
        log_path = Path(f"seed_{seed}.log")
        with open(log_path, "w") as f:
            f.write(result["stdout"])
        
        print(f"\nSeed {seed}: Final accuracy = {result['final_test_acc']:.4f}")
    
    # Compute statistics
    final_accs = [r["final_test_acc"] for r in results]
    mean_acc = np.mean(final_accs)
    std_acc = np.std(final_accs)
    min_acc = np.min(final_accs)
    max_acc = np.max(final_accs)
    
    # Summary
    summary = {
        "config": config_args,
        "seeds": args.seeds,
        "results": results,
        "statistics": {
            "mean_test_acc": float(mean_acc),
            "std_test_acc": float(std_acc),
            "min_test_acc": float(min_acc),
            "max_test_acc": float(max_acc),
            "all_final_accs": final_accs
        }
    }
    
    # Save results
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n" + "="*70)
    print("Multi-Seed Validation Results")
    print("="*70)
    print(f"Seeds: {args.seeds}")
    print(f"Mean Test Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"Min: {min_acc:.4f}, Max: {max_acc:.4f}")
    print(f"\nIndividual results:")
    for seed, acc in zip(args.seeds, final_accs):
        print(f"  Seed {seed}: {acc:.4f}")
    print(f"\nResults saved to: {args.output}")
    print("="*70)
    
    # Check if target achieved
    if mean_acc >= 0.95:
        print("\n🎉 SUCCESS: Mean accuracy ≥ 95% achieved!")
    elif mean_acc >= 0.94:
        print("\n✅ GOOD: Mean accuracy ≥ 94%, close to target")
    else:
        print(f"\n⚠️  WARNING: Mean accuracy {mean_acc:.2%} below target")


if __name__ == "__main__":
    main()
