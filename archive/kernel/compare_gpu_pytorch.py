#!/usr/bin/env python3
"""Compare CuPy GPU kernel vs PyTorch GPU."""

import torch
import time
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from kernel.eqprop_kernel import EqPropKernel
from src.models import LoopedMLP
from src.training import EqPropTrainer

print('='*60)
print('GPU Comparison: CuPy Kernel vs PyTorch')
print('='*60)

# Setup
batch_size = 64
hidden_dim = 256
num_steps = 50

# Data
x_np = np.random.randn(batch_size, 784).astype(np.float32)
y_np = np.random.randint(0, 10, size=batch_size)
x_torch = torch.from_numpy(x_np).cuda()
y_torch = torch.from_numpy(y_np).cuda()

# CuPy Kernel
print('\n[1/2] Benchmarking CuPy kernel on GPU...')
kernel = EqPropKernel(784, hidden_dim, 10, use_gpu=True)
for _ in range(5):
    kernel.train_step(x_np, y_np)

kernel_times = []
for i in range(num_steps):
    start = time.perf_counter()
    kernel.train_step(x_np, y_np)
    kernel_times.append(time.perf_counter() - start)
    if (i + 1) % 10 == 0:
        print(f"  Step {i+1}/{num_steps}")

kernel_avg = np.mean(kernel_times) * 1000

# PyTorch GPU
print('\n[2/2] Benchmarking PyTorch on GPU...')
model = LoopedMLP(784, hidden_dim, 10, use_spectral_norm=True).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
trainer = EqPropTrainer(model, optimizer, beta=0.22)

for _ in range(5):
    trainer.step(x_torch, y_torch)

torch_times = []
for i in range(num_steps):
    start = time.perf_counter()
    trainer.step(x_torch, y_torch)
    torch_times.append(time.perf_counter() - start)
    if (i + 1) % 10 == 0:
        print(f"  Step {i+1}/{num_steps}")

torch_avg = np.mean(torch_times) * 1000

# Results
print(f'\n{'='*60}')
print('RESULTS')
print(f'{'='*60}')
print(f'CuPy Kernel (GPU): {kernel_avg:.2f}ms/step')
print(f'PyTorch (GPU):     {torch_avg:.2f}ms/step')
speedup = torch_avg / kernel_avg
print(f'Speedup:           {speedup:.2f}x {"(kernel faster)" if speedup > 1 else "(PyTorch faster)"}')
print(f'Throughput:        {batch_size/kernel_avg*1000:.0f} vs {batch_size/torch_avg*1000:.0f} samples/sec')
print(f'{'='*60}')
