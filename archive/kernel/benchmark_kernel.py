"""
Benchmark: EqProp Kernel vs PyTorch Implementation

Compares:
1. Speed (wall time per training step)
2. Memory usage
3. Accuracy after fixed epochs/steps

Usage:
    python kernel/benchmark_kernel.py --test speed
    python kernel/benchmark_kernel.py --test mnist
    python kernel/benchmark_kernel.py --test all
"""

import numpy as np
import time
import argparse
from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kernel.eqprop_kernel import EqPropKernel, to_numpy


def load_mnist(subset_size=10000):
    """Load MNIST dataset (or generate synthetic if not available)."""
    try:
        import torch
        from torchvision import datasets, transforms
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1))
        ])
        
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, transform=transform)
        
        # Convert to numpy
        if subset_size:
            train_x = np.stack([train_dataset[i][0].numpy() for i in range(min(subset_size, len(train_dataset)))])
            train_y = np.array([train_dataset[i][1] for i in range(min(subset_size, len(train_dataset)))])
            test_x = np.stack([test_dataset[i][0].numpy() for i in range(min(subset_size, len(test_dataset)))])
            test_y = np.array([test_dataset[i][1] for i in range(min(subset_size, len(test_dataset)))])
        else:
            train_x = np.stack([d[0].numpy() for d in train_dataset])
            train_y = np.array([d[1] for d in train_dataset])
            test_x = np.stack([d[0].numpy() for d in test_dataset])
            test_y = np.array([d[1] for d in test_dataset])
        
        return train_x, train_y, test_x, test_y
    
    except ImportError:
        print("PyTorch/torchvision not available, using synthetic data")
        return generate_synthetic_data(subset_size)


def generate_synthetic_data(n_samples=10000):
    """Generate synthetic classification data."""
    np.random.seed(42)
    train_x = np.random.randn(n_samples, 784).astype(np.float32)
    train_y = np.random.randint(0, 10, size=n_samples)
    test_x = np.random.randn(n_samples // 10, 784).astype(np.float32)
    test_y = np.random.randint(0, 10, size=n_samples // 10)
    return train_x, train_y, test_x, test_y


def benchmark_speed(batch_size=64, num_steps=100, hidden_dim=256):
    """Benchmark training speed of kernel."""
    print(f"\n{'='*60}")
    print("Speed Benchmark: EqProp Kernel")
    print(f"{'='*60}")
    print(f"Batch size: {batch_size}")
    print(f"Hidden dim: {hidden_dim}")
    print(f"Steps: {num_steps}")
    
    # Create kernel
    kernel = EqPropKernel(784, hidden_dim, 10, use_gpu=False)
    
    # Generate random data
    x = np.random.randn(batch_size, 784).astype(np.float32)
    y = np.random.randint(0, 10, size=batch_size)
    
    # Warmup
    print("\nWarming up...")
    for _ in range(5):
        kernel.train_step(x, y)
    
    # Benchmark
    print("Running benchmark...")
    times = []
    for i in range(num_steps):
        start = time.perf_counter()
        metrics = kernel.train_step(x, y)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        
        if (i + 1) % 20 == 0:
            avg_time = np.mean(times[-20:])
            print(f"  Step {i+1}/{num_steps}: {avg_time*1000:.2f}ms/step, loss={metrics['loss']:.4f}")
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"\n{'='*60}")
    print("Results:")
    print(f"  Average time: {avg_time*1000:.2f}ms ± {std_time*1000:.2f}ms per step")
    print(f"  Throughput: {batch_size / avg_time:.0f} samples/sec")
    print(f"{'='*60}")
    
    return {
        'avg_time_ms': avg_time * 1000,
        'std_time_ms': std_time * 1000,
        'throughput': batch_size / avg_time,
    }


def benchmark_mnist(epochs=10, batch_size=64, hidden_dim=256, subset_size=10000):
    """Train on MNIST and measure accuracy."""
    print(f"\n{'='*60}")
    print("MNIST Benchmark: EqProp Kernel")
    print(f"{'='*60}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Hidden dim: {hidden_dim}")
    print(f"Subset size: {subset_size}")
    
    # Load data
    print("\nLoading MNIST...")
    train_x, train_y, test_x, test_y = load_mnist(subset_size)
    print(f"Train: {train_x.shape}, Test: {test_x.shape}")
    
    # Create kernel
    kernel = EqPropKernel(784, hidden_dim, 10, use_gpu=False, lr=0.001)
    
    # Training loop
    n_samples = len(train_x)
    n_batches = n_samples // batch_size
    
    print("\nTraining...")
    start_time = time.time()
    
    for epoch in range(epochs):
        # Shuffle
        perm = np.random.permutation(n_samples)
        train_x_shuffled = train_x[perm]
        train_y_shuffled = train_y[perm]
        
        epoch_loss = 0
        epoch_acc = 0
        
        for i in range(n_batches):
            x_batch = train_x_shuffled[i*batch_size:(i+1)*batch_size]
            y_batch = train_y_shuffled[i*batch_size:(i+1)*batch_size]
            
            metrics = kernel.train_step(x_batch, y_batch)
            epoch_loss += metrics['loss']
            epoch_acc += metrics['accuracy']
        
        epoch_loss /= n_batches
        epoch_acc /= n_batches
        
        # Test accuracy
        test_acc = kernel.evaluate(test_x, test_y)
        
        print(f"  Epoch {epoch+1:2d}/{epochs}: "
              f"train_loss={epoch_loss:.4f}, train_acc={epoch_acc:.2%}, test_acc={test_acc:.2%}")
    
    total_time = time.time() - start_time
    final_test_acc = kernel.evaluate(test_x, test_y)
    
    print(f"\n{'='*60}")
    print("Results:")
    print(f"  Final test accuracy: {final_test_acc:.2%}")
    print(f"  Total training time: {total_time:.1f}s")
    print(f"  Time per epoch: {total_time/epochs:.1f}s")
    print(f"{'='*60}")
    
    return {
        'final_accuracy': final_test_acc,
        'total_time': total_time,
        'time_per_epoch': total_time / epochs,
    }


def compare_pytorch_vs_kernel():
    """Compare PyTorch implementation vs kernel (if PyTorch available)."""
    print(f"\n{'='*60}")
    print("PyTorch vs Kernel Comparison")
    print(f"{'='*60}")
    
    try:
        import torch
        from src.models import LoopedMLP
        from src.training import EqPropTrainer
        
        print("\nPyTorch implementation available, running comparison...")
        
        # Setup
        batch_size = 64
        hidden_dim = 256
        num_steps = 50
        
        # Generate data
        x_np = np.random.randn(batch_size, 784).astype(np.float32)
        y_np = np.random.randint(0, 10, size=batch_size)
        x_torch = torch.from_numpy(x_np)
        y_torch = torch.from_numpy(y_np)
        
        # Kernel timing
        kernel = EqPropKernel(784, hidden_dim, 10, use_gpu=False)
        
        # Warmup
        for _ in range(5):
            kernel.train_step(x_np, y_np)
        
        kernel_times = []
        for _ in range(num_steps):
            start = time.perf_counter()
            kernel.train_step(x_np, y_np)
            kernel_times.append(time.perf_counter() - start)
        
        # PyTorch timing
        model = LoopedMLP(784, hidden_dim, 10, use_spectral_norm=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        trainer = EqPropTrainer(model, optimizer, beta=0.22)
        
        # Warmup
        for _ in range(5):
            trainer.step(x_torch, y_torch)
        
        torch_times = []
        for _ in range(num_steps):
            start = time.perf_counter()
            trainer.step(x_torch, y_torch)
            torch_times.append(time.perf_counter() - start)
        
        kernel_avg = np.mean(kernel_times) * 1000
        torch_avg = np.mean(torch_times) * 1000
        speedup = torch_avg / kernel_avg
        
        print(f"\nResults ({num_steps} steps):")
        print(f"  Kernel:  {kernel_avg:.2f}ms/step")
        print(f"  PyTorch: {torch_avg:.2f}ms/step")
        print(f"  Speedup: {speedup:.2f}x {'(kernel faster)' if speedup > 1 else '(PyTorch faster)'}")
        
        return {
            'kernel_ms': kernel_avg,
            'pytorch_ms': torch_avg,
            'speedup': speedup,
        }
        
    except ImportError as e:
        print(f"\nPyTorch comparison not available: {e}")
        print("Run benchmark_speed instead for kernel-only timing.")
        return None


def run_all_benchmarks():
    """Run all benchmarks."""
    results = {}
    
    # Speed benchmark
    results['speed'] = benchmark_speed(batch_size=64, num_steps=100)
    
    # MNIST training
    results['mnist'] = benchmark_mnist(epochs=10, batch_size=64, subset_size=10000)
    
    # PyTorch comparison
    results['comparison'] = compare_pytorch_vs_kernel()
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Kernel speed: {results['speed']['avg_time_ms']:.2f}ms/step")
    print(f"MNIST accuracy: {results['mnist']['final_accuracy']:.2%}")
    if results['comparison']:
        print(f"Speedup vs PyTorch: {results['comparison']['speedup']:.2f}x")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark EqProp Kernel")
    parser.add_argument('--test', choices=['speed', 'mnist', 'compare', 'all'], 
                        default='all', help='Which benchmark to run')
    parser.add_argument('--epochs', type=int, default=10, help='MNIST epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--subset-size', type=int, default=10000, help='MNIST subset size')
    
    args = parser.parse_args()
    
    if args.test == 'speed':
        benchmark_speed(batch_size=args.batch_size, hidden_dim=args.hidden_dim)
    elif args.test == 'mnist':
        benchmark_mnist(epochs=args.epochs, batch_size=args.batch_size, 
                       hidden_dim=args.hidden_dim, subset_size=args.subset_size)
    elif args.test == 'compare':
        compare_pytorch_vs_kernel()
    else:
        run_all_benchmarks()
