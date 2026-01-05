#!/usr/bin/env python3
"""
Profile EqProp Kernel to identify performance bottlenecks.

Usage:
    python kernel/profile_kernel.py --backend cpu
    python kernel/profile_kernel.py --backend gpu
"""

import numpy as np
import time
import sys
from pathlib import Path
import argparse
import cProfile
import pstats
from io import StringIO

sys.path.insert(0, str(Path(__file__).parent.parent))

from kernel.eqprop_kernel import EqPropKernel


def profile_kernel_components(use_gpu=False, batch_size=64):
    """Profile individual kernel components."""
    print("=" * 70)
    print(f"Profiling Kernel Components ({'GPU' if use_gpu else 'CPU'})")
    print("=" * 70)
    
    kernel = EqPropKernel(784, 256, 10, use_gpu=use_gpu, max_steps=15)
    x = np.random.randn(batch_size, 784).astype(np.float32)
    y = np.random.randint(0, 10, size=batch_size)
    
    # Component timing
    components = {}
    
    # 1. Spectral normalization
    print("\n[1/6] Profiling spectral normalization...")
    times = []
    for _ in range(20):
        start = time.perf_counter()
        _ = kernel._get_normalized_weights()
        times.append(time.perf_counter() - start)
    components['spectral_norm'] = np.mean(times) * 1000
    
    # 2. Forward step
    print("[2/6] Profiling forward step...")
    weights = kernel._get_normalized_weights()
    xp = kernel.xp
    
    # Convert to appropriate backend
    if use_gpu:
        x_backend = xp.asarray(x)
        x_emb = x_backend @ weights['embed'].T + kernel.biases['embed']
    else:
        x_emb = x @ kernel.weights['embed'].T + kernel.biases['embed']
    
    h = xp.zeros((batch_size, 256), dtype=xp.float32)
    
    times = []
    for _ in range(20):
        start = time.perf_counter()
        _, _ = kernel.forward_step(h, x_emb, weights)
        times.append(time.perf_counter() - start)
    components['forward_step'] = np.mean(times) * 1000
    
    # 3. Free phase equilibrium
    print("[3/6] Profiling free phase equilibrium...")
    times = []
    for _ in range(10):
        start = time.perf_counter()
        _, _, _ = kernel.solve_equilibrium(x, nudge_grad=None)
        times.append(time.perf_counter() - start)
    components['free_phase'] = np.mean(times) * 1000
    
    # 4. Nudged phase equilibrium
    print("[4/6] Profiling nudged phase equilibrium...")
    h_free, _, _ = kernel.solve_equilibrium(x, nudge_grad=None)
    logits = kernel.compute_output(h_free)
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(batch_size), y] = 1.0
    d_logits = probs - one_hot
    nudge_grad = d_logits @ kernel.weights['head']
    if use_gpu:
        nudge_grad = xp.asarray(nudge_grad)
    
    times = []
    for _ in range(10):
        start = time.perf_counter()
        _, _, _ = kernel.solve_equilibrium(x, nudge_grad=nudge_grad)
        times.append(time.perf_counter() - start)
    components['nudged_phase'] = np.mean(times) * 1000
    
    # 5. Hebbian update
    print("[5/6] Profiling Hebbian update computation...")
    h_free, act_free, _ = kernel.solve_equilibrium(x, nudge_grad=None)
    h_nudged, act_nudged, _ = kernel.solve_equilibrium(x, nudge_grad=nudge_grad)
    
    times = []
    for _ in range(20):
        start = time.perf_counter()
        _ = kernel.compute_hebbian_update(act_free[-1], act_nudged[-1])
        times.append(time.perf_counter() - start)
    components['hebbian_update'] = np.mean(times) * 1000
    
    # 6. Adam update
    print("[6/6] Profiling Adam optimizer update...")
    grads = kernel.compute_hebbian_update(act_free[-1], act_nudged[-1])
    grads['head'] = d_logits.T @ h_free / batch_size
    
    times = []
    for _ in range(20):
        start = time.perf_counter()
        kernel.adam_update(grads)
        times.append(time.perf_counter() - start)
    components['adam_update'] = np.mean(times) * 1000
    
    # Print results
    print("\n" + "=" * 70)
    print("COMPONENT TIMINGS")
    print("=" * 70)
    total_estimated = 0
    for name, ms in sorted(components.items(), key=lambda x: -x[1]):
        print(f"{name:25s}: {ms:8.3f}ms")
        total_estimated += ms
    print("-" * 70)
    print(f"{'Estimated total':25s}: {total_estimated:8.3f}ms")
    
    # Full train step for comparison
    print("\n[*] Running full train_step for comparison...")
    times = []
    for _ in range(10):
        start = time.perf_counter()
        kernel.train_step(x, y)
        times.append(time.perf_counter() - start)
    actual_time = np.mean(times) * 1000
    
    print(f"{'Actual train_step':25s}: {actual_time:8.3f}ms")
    print(f"{'Overhead':25s}: {actual_time - total_estimated:8.3f}ms ({(actual_time - total_estimated)/actual_time*100:.1f}%)")
    print("=" * 70)
    
    return components, actual_time


def profile_with_cprofile(use_gpu=False, batch_size=64, num_steps=20):
    """Use cProfile for detailed profiling."""
    print("\n" + "=" * 70)
    print(f"cProfile Analysis ({'GPU' if use_gpu else 'CPU'})")
    print("=" * 70)
    
    kernel = EqPropKernel(784, 256, 10, use_gpu=use_gpu)
    x = np.random.randn(batch_size, 784).astype(np.float32)
    y = np.random.randint(0, 10, size=batch_size)
    
    # Profile
    profiler = cProfile.Profile()
    profiler.enable()
    
    for _ in range(num_steps):
        kernel.train_step(x, y)
    
    profiler.disable()
    
    # Print stats
    s = StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(30)  # Top 30 functions
    
    print(s.getvalue())
    
    return profiler


def compare_equilibrium_solvers():
    """Compare equilibrium solving strategies."""
    print("\n" + "=" * 70)
    print("Equilibrium Solver Analysis")
    print("=" * 70)
    
    batch_size = 64
    x = np.random.randn(batch_size, 784).astype(np.float32)
    
    # Test different max_steps
    for max_steps in [5, 10, 15, 20, 25]:
        kernel = EqPropKernel(784, 256, 10, use_gpu=False, max_steps=max_steps)
        
        times = []
        convergence_steps = []
        for _ in range(10):
            start = time.perf_counter()
            _, _, info = kernel.solve_equilibrium(x)
            times.append(time.perf_counter() - start)
            convergence_steps.append(info['steps'])
        
        avg_time = np.mean(times) * 1000
        avg_steps = np.mean(convergence_steps)
        
        print(f"max_steps={max_steps:2d}: {avg_time:6.2f}ms, "
              f"avg_converge={avg_steps:5.1f}, "
              f"ms/step={avg_time/avg_steps:5.2f}")
    
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Profile EqProp kernel")
    parser.add_argument('--backend', choices=['cpu', 'gpu', 'both'], default='cpu')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--cprofile', action='store_true', 
                       help='Run cProfile analysis')
    parser.add_argument('--equilibrium', action='store_true',
                       help='Analyze equilibrium solver')
    
    args = parser.parse_args()
    
    backends = ['cpu', 'gpu'] if args.backend == 'both' else [args.backend]
    
    for backend in backends:
        use_gpu = (backend == 'gpu')
        
        # Component profiling
        profile_kernel_components(use_gpu=use_gpu, batch_size=args.batch_size)
        
        # cProfile if requested
        if args.cprofile:
            profile_with_cprofile(use_gpu=use_gpu, batch_size=args.batch_size)
    
    # Equilibrium analysis
    if args.equilibrium:
        compare_equilibrium_solvers()


if __name__ == "__main__":
    main()
