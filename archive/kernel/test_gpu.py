"""
GPU Testing for EqProp Kernel

Reusable test suite comparing CPU (NumPy) vs GPU (CuPy) performance.

Usage:
    python kernel/test_gpu.py --backend cpu
    python kernel/test_gpu.py --backend gpu
    python kernel/test_gpu.py --backend both
"""

import numpy as np
import time
import argparse
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kernel.eqprop_kernel import EqPropKernel, HAS_CUPY


def test_backend_availability():
    """Test which backends are available."""
    print("=" * 60)
    print("Backend Availability Test")
    print("=" * 60)
    print(f"NumPy (CPU): ✓ Available")
    print(f"CuPy (GPU):  {'✓ Available' if HAS_CUPY else '✗ Not installed'}")
    
    if HAS_CUPY:
        import cupy as cp
        print(f"  CuPy version: {cp.__version__}")
        try:
            device_count = cp.cuda.runtime.getDeviceCount()
            print(f"  GPU devices: {device_count}")
            for i in range(device_count):
                props = cp.cuda.runtime.getDeviceProperties(i)
                print(f"    Device {i}: {props['name'].decode()}")
        except Exception as e:
            print(f"  Error querying GPU: {e}")
    print("=" * 60)
    return HAS_CUPY


def benchmark_backend(use_gpu=False, batch_size=64, hidden_dim=256, num_steps=50):
    """Benchmark a specific backend."""
    backend_name = "GPU (CuPy)" if use_gpu else "CPU (NumPy)"
    
    if use_gpu and not HAS_CUPY:
        print(f"\n{backend_name}: Skipped (CuPy not available)")
        return None
    
    print(f"\n{'='*60}")
    print(f"Benchmarking {backend_name}")
    print(f"{'='*60}")
    print(f"Batch size: {batch_size}")
    print(f"Hidden dim: {hidden_dim}")
    print(f"Steps: {num_steps}")
    
    # Create kernel
    kernel = EqPropKernel(784, hidden_dim, 10, use_gpu=use_gpu)
    
    # Generate data
    x = np.random.randn(batch_size, 784).astype(np.float32)
    y = np.random.randint(0, 10, size=batch_size)
    
    # Warmup
    print("Warming up...")
    for _ in range(5):
        kernel.train_step(x, y)
    
    # Benchmark
    print("Benchmarking...")
    times = []
    losses = []
    accuracies = []
    
    for i in range(num_steps):
        start = time.perf_counter()
        metrics = kernel.train_step(x, y)
        elapsed = time.perf_counter() - start
        
        times.append(elapsed)
        losses.append(metrics['loss'])
        accuracies.append(metrics['accuracy'])
        
        if (i + 1) % 10 == 0:
            avg_time = np.mean(times[-10:])
            print(f"  Step {i+1:3d}/{num_steps}: {avg_time*1000:.2f}ms/step, "
                  f"loss={metrics['loss']:.4f}, acc={metrics['accuracy']:.2%}")
    
    results = {
        'backend': backend_name,
        'avg_time_ms': np.mean(times) * 1000,
        'std_time_ms': np.std(times) * 1000,
        'min_time_ms': np.min(times) * 1000,
        'max_time_ms': np.max(times) * 1000,
        'final_loss': losses[-1],
        'final_acc': accuracies[-1],
        'throughput': batch_size / np.mean(times),
    }
    
    print(f"\nResults:")
    print(f"  Time: {results['avg_time_ms']:.2f} ± {results['std_time_ms']:.2f} ms/step")
    print(f"  Range: [{results['min_time_ms']:.2f}, {results['max_time_ms']:.2f}] ms")
    print(f"  Throughput: {results['throughput']:.0f} samples/sec")
    print(f"  Final loss: {results['final_loss']:.4f}")
    print(f"  Final accuracy: {results['final_acc']:.2%}")
    print("=" * 60)
    
    return results


def compare_backends(batch_size=64, hidden_dim=256, num_steps=50):
    """Compare CPU vs GPU performance."""
    print("\n" + "=" * 60)
    print("CPU vs GPU Comparison")
    print("=" * 60)
    
    # CPU benchmark
    cpu_results = benchmark_backend(use_gpu=False, batch_size=batch_size, 
                                    hidden_dim=hidden_dim, num_steps=num_steps)
    
    # GPU benchmark
    gpu_results = benchmark_backend(use_gpu=True, batch_size=batch_size, 
                                    hidden_dim=hidden_dim, num_steps=num_steps)
    
    # Comparison
    if cpu_results and gpu_results:
        speedup = cpu_results['avg_time_ms'] / gpu_results['avg_time_ms']
        
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        print(f"CPU: {cpu_results['avg_time_ms']:.2f}ms/step")
        print(f"GPU: {gpu_results['avg_time_ms']:.2f}ms/step")
        print(f"Speedup: {speedup:.2f}x {'(GPU faster)' if speedup > 1 else '(CPU faster)'}")
        print(f"Throughput improvement: {(gpu_results['throughput'] / cpu_results['throughput']):.2f}x")
        print("=" * 60)
        
        return {'cpu': cpu_results, 'gpu': gpu_results, 'speedup': speedup}
    
    return {'cpu': cpu_results, 'gpu': gpu_results, 'speedup': None}


def test_correctness(use_gpu=False):
    """Test that GPU and CPU produce similar results."""
    print("\n" + "=" * 60)
    print("Correctness Test")
    print("=" * 60)
    
    if use_gpu and not HAS_CUPY:
        print("GPU not available, skipping correctness test")
        return
    
    # Create both kernels with same seed
    np.random.seed(42)
    cpu_kernel = EqPropKernel(784, 128, 10, use_gpu=False)
    
    if use_gpu:
        np.random.seed(42)
        gpu_kernel = EqPropKernel(784, 128, 10, use_gpu=True)
    
    # Test data
    x = np.random.randn(32, 784).astype(np.float32)
    y = np.random.randint(0, 10, size=32)
    
    # Train CPU
    cpu_metrics = cpu_kernel.train_step(x, y)
    
    if use_gpu:
        # Train GPU
        gpu_metrics = gpu_kernel.train_step(x, y)
        
        # Compare
        loss_diff = abs(cpu_metrics['loss'] - gpu_metrics['loss'])
        acc_diff = abs(cpu_metrics['accuracy'] - gpu_metrics['accuracy'])
        
        print(f"CPU loss: {cpu_metrics['loss']:.6f}")
        print(f"GPU loss: {gpu_metrics['loss']:.6f}")
        print(f"Loss diff: {loss_diff:.6f}")
        print(f"\nCPU acc: {cpu_metrics['accuracy']:.4f}")
        print(f"GPU acc: {gpu_metrics['accuracy']:.4f}")
        print(f"Acc diff: {acc_diff:.4f}")
        
        # Allow small numerical differences
        if loss_diff < 0.01 and acc_diff < 0.01:
            print("\n✓ Correctness test passed (within tolerance)")
            return True
        else:
            print("\n✗ Correctness test failed (differences too large)")
            return False
    
    print(f"CPU only - loss: {cpu_metrics['loss']:.6f}, acc: {cpu_metrics['accuracy']:.4f}")
    print("✓ CPU test passed")
    return True


def main():
    parser = argparse.ArgumentParser(description="Test EqProp GPU performance")
    parser.add_argument('--backend', choices=['cpu', 'gpu', 'both'], default='both',
                       help='Which backend to test')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--num-steps', type=int, default=50, help='Number of benchmark steps')
    parser.add_argument('--test-correctness', action='store_true', 
                       help='Test CPU/GPU correctness')
    
    args = parser.parse_args()
    
    # Check availability
    has_gpu = test_backend_availability()
    
    # Correctness test
    if args.test_correctness:
        test_correctness(use_gpu=has_gpu)
        return
    
    # Benchmarks
    if args.backend == 'both':
        compare_backends(args.batch_size, args.hidden_dim, args.num_steps)
    elif args.backend == 'cpu':
        benchmark_backend(use_gpu=False, batch_size=args.batch_size,
                        hidden_dim=args.hidden_dim, num_steps=args.num_steps)
    elif args.backend == 'gpu':
        if not has_gpu:
            print("\nError: GPU backend requested but CuPy not available")
            sys.exit(1)
        benchmark_backend(use_gpu=True, batch_size=args.batch_size,
                        hidden_dim=args.hidden_dim, num_steps=args.num_steps)


if __name__ == "__main__":
    main()
