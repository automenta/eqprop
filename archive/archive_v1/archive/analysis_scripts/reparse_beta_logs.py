"""Reparse existing beta sweep logs with fixed parser."""

import json
from pathlib import Path
from datetime import datetime


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
    logs_dir = Path("logs/beta_sweep")
    beta_values = [0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26]
    
    results = []
    
    for beta in beta_values:
        log_file = logs_dir / f"beta_{beta:.2f}.log"
        if not log_file.exists():
            print(f"Warning: {log_file} not found")
            continue
        
        print(f"Parsing β={beta:.2f}...")
        parsed = parse_log(log_file)
        parsed["beta"] = beta
        results.append(parsed)
        
        status = "✅ STABLE" if parsed["stable"] else ("❌ COLLAPSE" if parsed["collapsed"] else "⚠️ ERROR")
        print(f"  {status} - Final: {parsed['final_test_acc']:.4f}, Peak: {parsed['peak_test_acc']:.4f}")
    
    # Save results
    output = {
        "experiment": "beta_stability_sweep",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "beta_min": 0.20,
            "beta_max": 0.26,
            "beta_step": 0.01,
            "epochs": 15
        },
        "results": results
    }
    
    output_file = logs_dir / "results.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✅ Reparsed results saved to {output_file}")
    
    # Print summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"{'β':>6} {'Status':>12} {'Final Acc':>12} {'Peak Acc':>12}")
    print(f"{'-'*50}")
    for r in results:
        status = "✅ STABLE" if r["stable"] else ("❌ COLLAPSE" if r["collapsed"] else "⚠️ ERROR")
        print(f"{r['beta']:>6.2f} {status:>12} {r['final_test_acc']:>12.4f} {r['peak_test_acc']:>12.4f}")
    
    # Find optimal
    if results:
        optimal = max(results, key=lambda x: x['peak_test_acc'])
        print(f"\n🏆 Optimal β: {optimal['beta']:.2f} (peak acc: {optimal['peak_test_acc']:.4f})")


if __name__ == "__main__":
    main()
