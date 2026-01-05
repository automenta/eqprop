#!/usr/bin/env python3
"""Test optimized kernel performance."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kernel.eqprop_kernel import EqPropKernel
import numpy as np
import time

print("=" * 70)
print("Testing Kernel Optimizations")
print("=" * 70)

# Test configurations
configs = [
    ("Baseline (max_steps=15, adaptive=False)", dict(max_steps=15, adaptive_epsilon=False)),
    ("Optimized (max_steps=10, adaptive=True)", dict(max_steps=10, adaptive_epsilon=True)),
    ("Aggressive (max_steps=8, adaptive=True)", dict(max_steps=8, adaptive_epsilon=True)),
]

x = np.random.randn(64, 784).astype(np.float32)
y = np.random.randint(0, 10, size=64)

results = {}

for name, config in configs:
    print(f"\n{name}")
    print("-" * 70)
    
    kernel = EqPropKernel(784, 256, 10, use_gpu=True, 
                         use_spectral_norm=True, **config)
    
    # Warmup
    for _ in range(5):
        kernel.train_step(x, y)
    
    # Benchmark
    times = []
    convergence_info = []
    for _ in range(30):
        start = time.perf_counter()
        metrics = kernel.train_step(x, y)
        times.append(time.perf_counter() - start)
        convergence_info.append((metrics['free_steps'], metrics['nudged_steps']))
    
    avg_time = np.mean(times) * 1000
    avg_free = np.mean([c[0] for c in convergence_info])
    avg_nudged = np.mean([c[1] for c in convergence_info])
    
    results[name] = avg_time
    
    print(f"  Time: {avg_time:.2f}ms/step")
    print(f"  Avg free steps: {avg_free:.1f}")
    print(f"  Avg nudged steps: {avg_nudged:.1f}")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
baseline = results[configs[0][0]]
for name, time_ms in results.items():
    speedup = baseline / time_ms
    improvement = (baseline - time_ms) / baseline * 100
    print(f"{name:50s}: {time_ms:6.2f}ms ({speedup:.2f}x, +{improvement:4.1f}%)")

print("\nComparison to PyTorch GPU (33.9ms):")
optimized = results[configs[1][0]]
vs_pytorch = optimized / 33.9
print(f"  Optimized kernel: {optimized:.2f}ms ({vs_pytorch:.2f}x PyTorch)")
print("=" * 70)
