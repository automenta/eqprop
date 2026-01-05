"""β Stability Boundary Characterization Experiment.

This script systematically sweeps across β values to:
1. Identify the precise stability threshold
2. Find the optimal β for maximum accuracy
3. Document the theory-practice gap (β→0 theoretical vs β≥0.23 practical)
"""

import torch
import json
import argparse
from pathlib import Path
import subprocess
import time
from datetime import datetime


def run_training(beta: float, epochs: int = 15, output_dir: Path = Path("logs/beta_sweep")):
    """Run a single training run with specified β."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / f"beta_{beta:.2f}.log"
    
    cmd = [
        "python", "train.py",
        "--d-model", "256",
        "--n-heads", "8",
        "--d-ff", "1024",
        "--beta", str(beta),
        "--damping", "0.8",
        "--lr", "0.002",
        "--epochs", str(epochs),
        "--dropout", "0.1",
        "--compile"
    ]
    
    print(f"\n{'='*70}")
    print(f"Training with β={beta:.2f}")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    with open(log_file, "w") as f:
        try:
            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                check=False
            )
            elapsed = time.time() - start_time
            
            # Parse results from log
            results = parse_log(log_file)
            results["beta"] = beta
            results["elapsed_time"] = elapsed
            results["exit_code"] = result.returncode
            
            return results
            
        except Exception as e:
            print(f"Error running β={beta:.2f}: {e}")
            return {
                "beta": beta,
                "collapsed": True,
                "error": str(e)
            }


def parse_log(log_file: Path) -> dict:
    """Parse training log to extract key metrics."""
    with open(log_file) as f:
        lines = f.readlines()
    
    # Extract epoch-by-epoch accuracy
    train_accs = []
    test_accs = []
    collapsed = False
    collapse_epoch = None
    
    for i, line in enumerate(lines):
        # Look for epoch summaries: "  Train Loss: X.XX, Train Acc: 0.XXXX"
        if "Train Loss:" in line and "Train Acc:" in line:
            try:
                # Parse train accuracy from this line
                train_part = line.split("Train Acc:")[1].strip()
                train_acc = float(train_part)
                train_accs.append(train_acc)
                
            except (IndexError, ValueError):
                continue
        
        # Look for test accuracy: "  Test Loss: X.XX, Test Acc: 0.XXXX"
        if "Test Loss:" in line and "Test Acc:" in line:
            try:
                # Parse test accuracy from this line
                test_part = line.split("Test Acc:")[1].strip()
                test_acc = float(test_part)
                test_accs.append(test_acc)
                
                # Detect catastrophic collapse (after we have a test acc)
                if len(train_accs) > 0 and train_accs[-1] < 0.20 and len(train_accs) > 2:
                    collapsed = True
                    collapse_epoch = len(train_accs) - 1
                    
            except (IndexError, ValueError):
                continue
    
    # Determine stability
    stable = not collapsed and len(test_accs) > 0
    
    return {
        "train_accs": train_accs,
        "test_accs": test_accs,
        "final_train_acc": train_accs[-1] if train_accs else 0.0,
        "final_test_acc": test_accs[-1] if test_accs else 0.0,
        "peak_test_acc": max(test_accs) if test_accs else 0.0,
        "collapsed": collapsed,
        "collapse_epoch": collapse_epoch,
        "stable": stable,
        "num_epochs": len(test_accs)
    }


def main():
    parser = argparse.ArgumentParser(description="β Stability Sweep")
    parser.add_argument("--beta-min", type=float, default=0.20, help="Minimum β")
    parser.add_argument("--beta-max", type=float, default=0.26, help="Maximum β")
    parser.add_argument("--beta-step", type=float, default=0.01, help="β step size")
    parser.add_argument("--epochs", type=int, default=15, help="Epochs per run")
    parser.add_argument("--output-dir", type=str, default="logs/beta_sweep", help="Output directory")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate β values
    beta_values = []
    beta = args.beta_min
    while beta <= args.beta_max + 1e-9:  # Small epsilon for float comparison
        beta_values.append(round(beta, 2))
        beta += args.beta_step
    
    print(f"\n{'='*70}")
    print(f"β STABILITY BOUNDARY CHARACTERIZATION")
    print(f"{'='*70}")
    print(f"β range: [{args.beta_min:.2f}, {args.beta_max:.2f}]")
    print(f"β step: {args.beta_step:.2f}")
    print(f"β values: {beta_values}")
    print(f"Epochs per run: {args.epochs}")
    print(f"Total runs: {len(beta_values)}")
    print(f"Estimated time: {len(beta_values) * 15 * 66 / 3600:.1f} hours")
    print(f"Output: {output_dir}")
    print(f"{'='*70}\n")
    
    # Run experiments
    all_results = []
    start_time = time.time()
    
    for i, beta in enumerate(beta_values):
        print(f"\n[Run {i+1}/{len(beta_values)}] β={beta:.2f}")
        results = run_training(beta, args.epochs, output_dir)
        all_results.append(results)
        
        # Print summary
        if results.get("stable", False):
            print(f"✅ STABLE - Final acc: {results['final_test_acc']:.4f}, Peak: {results['peak_test_acc']:.4f}")
        elif results.get("collapsed", False):
            print(f"❌ COLLAPSED at epoch {results.get('collapse_epoch', 'unknown')}")
        else:
            print(f"⚠️  INCOMPLETE or ERROR")
        
        # Estimate remaining time
        elapsed = time.time() - start_time
        avg_time_per_run = elapsed / (i + 1)
        remaining_runs = len(beta_values) - (i + 1)
        eta_seconds = avg_time_per_run * remaining_runs
        eta_hours = eta_seconds / 3600
        print(f"   Elapsed: {elapsed/3600:.1f}h, ETA: {eta_hours:.1f}h")
    
    # Save results
    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump({
            "experiment": "beta_stability_sweep",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "beta_min": args.beta_min,
                "beta_max": args.beta_max,
                "beta_step": args.beta_step,
                "epochs": args.epochs
            },
            "results": all_results
        }, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"✅ EXPERIMENT COMPLETE")
    print(f"{'='*70}")
    print(f"Total time: {(time.time() - start_time)/3600:.2f} hours")
    print(f"Results saved to: {results_file}")
    print(f"\nSummary:")
    
    # Print summary table
    print(f"\n{'β':>6} {'Status':>12} {'Final Acc':>12} {'Peak Acc':>12}")
    print(f"{'-'*50}")
    for r in all_results:
        status = "✅ STABLE" if r.get("stable") else ("❌ COLLAPSE" if r.get("collapsed") else "⚠️  ERROR")
        final = r.get("final_test_acc", 0.0)
        peak = r.get("peak_test_acc", 0.0)
        print(f"{r['beta']:>6.2f} {status:>12} {final:>12.4f} {peak:>12.4f}")
    
    # Identify threshold
    stable_betas = [r['beta'] for r in all_results if r.get('stable', False)]
    collapsed_betas = [r['beta'] for r in all_results if r.get('collapsed', False)]
    
    if stable_betas and collapsed_betas:
        threshold = (min(stable_betas) + max(collapsed_betas)) / 2
        print(f"\n💡 Estimated stability threshold: β ≈ {threshold:.3f}")
    
    # Find optimal β
    if stable_betas:
        optimal = max(all_results, key=lambda x: x.get('peak_test_acc', 0.0) if x.get('stable', False) else 0.0)
        print(f"🏆 Optimal β: {optimal['beta']:.2f} (peak acc: {optimal['peak_test_acc']:.4f})")
    
    print(f"\nNext step: Run analysis script")
    print(f"  python analyze_beta_sweep.py --results {results_file}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
