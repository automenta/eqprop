#!/usr/bin/env python3
"""Verify effort-matched iterations in size_comparison config"""
from validation_engine import ValidationEngine

engine = ValidationEngine()
specs = engine.scheduler._get_size_comparison_specs()

# Group by size
by_size = {}
for spec in specs:
    if spec.algorithm == "eqprop" and "mnist" in spec.experiment_id:
        size = spec.model_size
        if size not in by_size:
            by_size[size] = spec

print("Effort-Matched Iterations per Size:")
print("=" * 60)
for size in ["micro", "tiny", "small", "medium", "large"]:
    if size in by_size:
        spec = by_size[size]
        # Extract max_iters from command
        cmd = spec.command
        if "--max-iters" in cmd:
            iters = cmd.split("--max-iters")[1].split()[0]
            print(f"{size:8} (d={spec.d_model:3}): max_iters={iters:3}  | {cmd[:80]}...")
        else:
            print(f"{size:8} (d={spec.d_model:3}): NO max_iters!")
