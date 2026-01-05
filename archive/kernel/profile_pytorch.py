#!/usr/bin/env python3
"""
Profile PyTorch implementation for comparison with kernel.

Usage:
    CUDA_PATH=/opt/cuda python kernel/profile_pytorch.py
"""

import torch
import time
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import LoopedMLP
from src.training import EqPropTrainer

print("=" * 70)
print("PyTorch Implementation Profiling")
print("=" * 70)

# Setup
batch_size = 64
hidden_dim = 256
model = LoopedMLP(784, hidden_dim, 10, use_spectral_norm=True).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
trainer = EqPropTrainer(model, optimizer, beta=0.22)

x = torch.randn(batch_size, 784).cuda()
y = torch.randint(0, 10, (batch_size,)).cuda()

# Warmup
for _ in range(5):
    trainer.step(x, y)

# Component timing with PyTorch profiler
print("\nUsing torch.profiler for detailed analysis...")

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    for _ in range(10):
        trainer.step(x, y)

# Print top operations
print("\nTop 15 operations by CUDA time:")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

print("\nTop 15 operations by CPU time:")
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))

# Manual component timing
print("\n" + "=" * 70)
print("Manual Component Timing")
print("=" * 70)

components = {}

# Full training step
times = []
for _ in range(50):
    start = time.perf_counter()
    trainer.step(x, y)
    torch.cuda.synchronize()
    times.append(time.perf_counter() - start)
avg_time = np.mean(times) * 1000

print(f"Full training step: {avg_time:.2f}ms")

# Forward pass (free phase)
model.train()
times = []
for _ in range(50):
    start = time.perf_counter()
    with torch.no_grad():
        _ = model(x)
    torch.cuda.synchronize()
    times.append(time.perf_counter() - start)
components['forward_pass'] = np.mean(times) * 1000

# Equilibrium solving
from src.training.equilibrium import EquilibriumSolver
solver = EquilibriumSolver(max_steps=15)
times = []
for _ in range(50):
    start = time.perf_counter()
    _, _ = solver.solve(model, x)
    torch.cuda.synchronize()
    times.append(time.perf_counter() - start)
components['equilibrium'] = np.mean(times) * 1000

# Optimizer step
optimizer.zero_grad()
loss = torch.nn.functional.cross_entropy(model(x), y)
loss.backward()
times = []
for _ in range(50):
    optimizer.zero_grad()
    loss = torch.nn.functional.cross_entropy(model(x), y)
    loss.backward()
    start = time.perf_counter()
    optimizer.step()
    torch.cuda.synchronize()
    times.append(time.perf_counter() - start)
components['optimizer_step'] = np.mean(times) * 1000

print("\nComponent timings:")
for name, ms in sorted(components.items(), key=lambda x: -x[1]):
    print(f"  {name:20s}: {ms:6.2f}ms")

print("=" * 70)
