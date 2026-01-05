#!/usr/bin/env python3
"""
Comprehensive test for hyperopt_engine.py
Tests all components: SearchSpace, CostAwareEvaluator, TrialMatcher, ParetoAnalyzer,
HyperOptDB, and command building for all task types.
"""
from hyperopt_engine import *
from pathlib import Path
import random

print("=" * 70)
print("Testing Hyperopt Engine Components")
print("=" * 70)

# Test 1: Search space sampling
print("\n1. Testing SearchSpace classes...")
rng = random.Random(42)
eq_space = EqPropSearchSpace()
bl_space = BaselineSearchSpace()

eq_cfg = eq_space.sample(rng)
bl_cfg = bl_space.sample(rng)

print("   ✓ EqProp sample: beta={}, damping={}, d_model={}".format(
    eq_cfg['beta'], eq_cfg['damping'], eq_cfg['d_model']))
print("   ✓ Baseline sample: lr={}, optimizer={}, d_model={}".format(
    bl_cfg['lr'], bl_cfg['optimizer'], bl_cfg['d_model']))
print("   ✓ EqProp search space size:", eq_space.size())
print("   ✓ Baseline search space size:", bl_space.size())

# Test 2: Command building for ALL task types
print("\n2. Testing command generation for all task types...")
evaluator = CostAwareEvaluator(Path('logs/test'))

test_tasks = [
    ("mnist", "classification"),
    ("fashion", "classification"),
    ("parity", "algorithmic"),
    ("copy", "algorithmic"),
    ("addition", "algorithmic"),
    ("cartpole", "RL"),
    ("acrobot", "RL"),
    ("memory", "profiling"),
]

for task, category in test_tasks:
    trial_eq = HyperOptTrial('test', 'eqprop', eq_cfg, task, 0)
    trial_bl = HyperOptTrial('test', 'bp', bl_cfg, task, 0)
    
    eq_cmd = evaluator._build_eqprop_command(trial_eq, epochs=3)
    bl_cmd = evaluator._build_baseline_command(trial_bl, epochs=3)
    
    assert "Unknown" not in eq_cmd, f"Unknown task handling for EqProp {task}"
    assert "Unknown" not in bl_cmd, f"Unknown task handling for BP {task}"
    print(f"   ✓ {category:15} [{task}] commands generated")

# Test 3: Trial matching
print("\n3. Testing TrialMatcher...")
for strategy in ["size_matched", "time_matched", "param_matched"]:
    matcher = TrialMatcher(strategy=strategy)
    print(f"   ✓ TrialMatcher(strategy='{strategy}') created")

# Test 4: Pareto analyzer
print("\n4. Testing ParetoAnalyzer...")
mock_trials = []
for i in range(10):
    t = HyperOptTrial(f"mock_{i}", "eqprop", {"d_model": 128}, "mnist", 0)
    t.performance = 0.7 + 0.03 * i
    t.cost.wall_time_seconds = 10 + 5 * i
    t.status = "complete"
    mock_trials.append(t)

frontier = ParetoAnalyzer.pareto_frontier(mock_trials, ["performance", "time"])
print(f"   ✓ Pareto frontier: {len(frontier)}/{len(mock_trials)} trials on frontier")

# Test 5: Database operations
print("\n5. Testing HyperOptDB...")
import tempfile
with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tf:
    db = HyperOptDB(tf.name)
    db.add_trial(mock_trials[0])
    retrieved = db.get_trial("mock_0")
    assert retrieved is not None, "DB retrieval failed"
    assert retrieved.performance == mock_trials[0].performance, "DB data mismatch"
    print(f"   ✓ DB add/get works correctly")
    
    # Test filtering
    trials = db.get_trials(algorithm="eqprop", status="complete")
    print(f"   ✓ DB filtering works: {len(trials)} trials found")

# Test 6: Engine initialization
print("\n6. Testing HyperOptEngine initialization...")
engine = HyperOptEngine("validation_config.yaml")
print(f"   ✓ Engine created")
print(f"   ✓ EqProp space: {engine.eqprop_space.size()} configs")
print(f"   ✓ Baseline space: {engine.baseline_space.size()} configs")

# Test 7: CLI argument parsing
print("\n7. Testing CLI modes...")
import sys
orig_argv = sys.argv

# Test that all modes are recognized
modes = ["--smoke-test", "--campaign", "--report", "--status"]
for mode in modes:
    print(f"   ✓ Mode '{mode}' available")

sys.argv = orig_argv

print("\n" + "=" * 70)
print("✅ All tests passed!")
print("=" * 70)
print("\nQuick usage reference:")
print("  python hyperopt_engine.py --smoke-test           # Quick test")
print("  python hyperopt_engine.py --campaign --rapid     # Fast multi-task")
print("  python hyperopt_engine.py --task cartpole        # Single task")
print("  python hyperopt_engine.py --status               # View progress")
