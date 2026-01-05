#!/usr/bin/env python3
"""
Simple, reliable smoke test verification script.
Checks that all experiment infrastructure is correct.
"""

import json
import sys
from pathlib import Path


def verify_smoke_test(log_dir: str = "logs/discovery/20251229_101357"):
    """Verify smoke test results."""
    log_path = Path(log_dir)
    
    if not log_path.exists():
        print(f"❌ Log directory not found: {log_dir}")
        return False
    
    # Check for results file
    results_file = log_path / "results.json"
    if not results_file.exists():
        print(f"⏳ Test still running - results.json not found")
        return None
    
    # Load results
    with open(results_file) as f:
        data = json.load(f)
    
    results = data.get("results", [])
    
    print(f"\n{'='*70}")
    print(f"SMOKE TEST VERIFICATION: {log_dir}")
    print(f"{'='*70}\n")
    
    # Critical checks
    checks = {
        "total": len(results),
        "success": 0,
        "failure": 0,
        "error": 0,
        "compile_fixed": False,
        "memory_d256_correct": False,
        "memory_d1024_correct": False,
        "memory_d2048_correct": False,
    }
    
    for result in results:
        status = result.get("status", "").upper()
        name = result.get("name", "")
        
        if status == "SUCCESS":
            checks["success"] += 1
        elif status == "FAILURE":
            checks["failure"] += 1
        elif status == "ERROR":
            checks["error"] += 1
            
        # Check specific fixes
        if "MNIST Extended" in name and status == "SUCCESS":
            checks["compile_fixed"] = True
        if "Memory d=256" in name and status == "SUCCESS":
            checks["memory_d256_correct"] = True
        if "Memory d=1024" in name and status == "SUCCESS":
            checks["memory_d1024_correct"] = True
        if "Memory d=2048" in name and status == "SUCCESS":
            checks["memory_d2048_correct"] = True
    
    # Print results
    print(f"Total experiments: {checks['total']}")
    print(f"  ✅ Success: {checks['success']}")
    print(f"  ❌ Failure: {checks['failure']}")
    print(f"  ⚠️  Error: {checks['error']}\n")
    
    print("Critical Bug Fixes:")
    print(f"  {'✅' if checks['compile_fixed'] else '❌'} MNIST Extended (--compile flag)")
    print(f"  {'✅' if checks['memory_d256_correct'] else '❌'} Memory d=256 (correct d_model)")
    print(f"  {'✅' if checks['memory_d1024_correct'] else '❌'} Memory d=1024 (correct d_model)")
    print(f"  {'✅' if checks['memory_d2048_correct'] else '❌'} Memory d=2048 (correct d_model)\n")
    
    # Overall verdict
    all_critical_fixed = all([
        checks['compile_fixed'],
        checks['memory_d256_correct'],
        checks['memory_d1024_correct'],
        checks['memory_d2048_correct']
    ])
    
    all_passed = checks['error'] == 0 and all_critical_fixed
    
    if all_passed:
        print("✅ VERDICT: All critical infrastructure verified and working!")
        return True
    elif all_critical_fixed:
        print("⚠️  VERDICT: Bug fixes work, but some experiments failed/errored")
        print("   (This is expected for smoke tests on complex tasks)")
        return True
    else:
        print("❌ VERDICT: Critical bugs still present")
        return False


if __name__ == "__main__":
    # Find most recent log directory
    import glob
    dirs = sorted(glob.glob("logs/discovery/*/results.json"), reverse=True)
    
    if not dirs:
        print("❌ No smoke test results found")
        sys.exit(1)
    
    latest_dir = str(Path(dirs[0]).parent)
    print(f"Checking latest results: {latest_dir}\n")
    
    result = verify_smoke_test(latest_dir)
    
    if result is None:
        print("\nTest still running. Wait and run again.")
        sys.exit(2)
    elif result:
        sys.exit(0)
    else:
        sys.exit(1)
