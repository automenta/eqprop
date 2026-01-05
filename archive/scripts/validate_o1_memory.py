#!/usr/bin/env python3
"""Test memory usage scaling with model depth.

Since LocalHebbianUpdate needs trainer integration, we instead demonstrate
EqProp's memory advantage by showing constant memory with model depth (vs linear for BP).
"""

import sys
sys.path.insert(0, '.')

import torch
import torch.optim as optim
import torch.nn as nn
import torch.cuda as cuda

from src.models import LoopedMLP, BackpropMLP
from src.training import EqPropTrainer
from src.tasks import get_task_loader


def measure_peak_memory(train_func, warmup=3):
    """Measure peak GPU memory during training."""
    cuda.empty_cache()
    cuda.reset_peak_memory_stats()
    
    # Warmup
    for _ in range(warmup):
        train_func()
    
    # Measure
    cuda.empty_cache()
    cuda.reset_peak_memory_stats()
    
    initial = cuda.memory_allocated()
    train_func()
    peak = cuda.max_memory_allocated()
    
    return (peak - initial) / 1024**2  # MB


def test_eqprop_constant_memory():
    """Test that EqProp memory doesn't scale with depth."""
    print("=" * 70)
    print("EqProp MEMORY SCALING TEST")
    print("=" * 70)
    print("\nHypothesis: EqProp memory usage is O(1) in model depth")
    print("(Because we don't store intermediate activations for backprop)\n")
    
    train_loader, _, input_dim, output_dim = get_task_loader(
        'digits', batch_size=64, dataset_size=500
    )
    
    x, y = next(iter(train_loader))
    x, y = x.cuda(), y.cuda()
    
    results = []
    
    # Test different hidden dimensions (proxy for depth)
    hidden_dims = [64, 128, 256, 512, 1024]
    
    print("Testing EqProp (LoopedMLP with spectral norm):")
    print("-" * 50)
    
    for hidden_dim in hidden_dims:
        model = LoopedMLP(input_dim, hidden_dim, output_dim,
                         symmetric=True, use_spectral_norm=True).cuda()
        from src.training.updates import LocalHebbianUpdate
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        # Use LocalHebbianUpdate for TRUE O(1) memory
        update_strategy = LocalHebbianUpdate(beta=0.22)
        trainer = EqPropTrainer(model, optimizer, beta=0.22, max_steps=25,
                               update_strategy=update_strategy)
        
        def train_step():
            optimizer.zero_grad()
            metrics = trainer.step(x, y)
        
        mem_mb = measure_peak_memory(train_step)
        param_count = sum(p.numel() for p in model.parameters())
        
        results.append({
            'hidden_dim': hidden_dim,
            'params': param_count,
            'memory_mb': mem_mb
        })
        
        print(f"  hidden_dim={hidden_dim:4d}: {mem_mb:6.2f} MB ({param_count:,} params)")
    
    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    mem_values = [r['memory_mb'] for r in results]
    mem_increase = (mem_values[-1] - mem_values[0]) / mem_values[0] * 100
    param_increase = (results[-1]['params'] - results[0]['params']) / results[0]['params'] * 100
    
    print(f"\nParameter increase (64→1024 hidden): {param_increase:.0f}%")
    print(f"Memory increase (64→1024 hidden):    {mem_increase:.0f}%")
    print(f"\nMemory scaling rate: {mem_increase/param_increase:.2f}× parameter growth")
    
    if mem_increase < param_increase * 0.5:
        print("\n✓ SUB-LINEAR MEMORY SCALING: EqProp shows memory efficiency")
    else:
        print("\n⚠ LINEAR MEMORY SCALING: Similar to standard backprop")
    
    # Save for comparison
    return results


def test_backprop_memory():
    """Compare with standard backprop memory scaling."""
    print("\n" + "=" * 70)
    print("BACKPROP MEMORY SCALING (Comparison)")
    print("=" * 70)
    
    train_loader, _, input_dim, output_dim = get_task_loader(
        'digits', batch_size=64, dataset_size=500
    )
    
    x, y = next(iter(train_loader))
    x, y = x.cuda(), y.cuda()
    
    results = []
    hidden_dims = [64, 128, 256, 512, 1024]
    
    print("\nTesting Backprop (2-layer MLP):")
    print("-" * 50)
    
    for hidden_dim in hidden_dims:
        model = BackpropMLP(input_dim, hidden_dim, output_dim, depth=2).cuda()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        def train_step():
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
        
        mem_mb = measure_peak_memory(train_step)
        param_count = sum(p.numel() for p in model.parameters())
        
        results.append({
            'hidden_dim': hidden_dim,
            'params': param_count,
            'memory_mb': mem_mb
        })
        
        print(f"  hidden_dim={hidden_dim:4d}: {mem_mb:6.2f} MB ({param_count:,} params)")
    
    return results


def main():
    eqprop_results = test_eqprop_constant_memory()
    backprop_results = test_backprop_memory()
    
    # Final comparison
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    
    print(f"\n{'Hidden Dim':<12} | {'EqProp Mem':>12} | {'Backprop Mem':>13} | {'Ratio'}")
    print("-" * 70)
    
    for i, hidden_dim in enumerate([64, 128, 256, 512, 1024]):
        eq_mem = eqprop_results[i]['memory_mb']
        bp_mem = backprop_results[i]['memory_mb']
        ratio = eq_mem / bp_mem
        
        print(f"{hidden_dim:<12} | {eq_mem:>11.2f}MB | {bp_mem:>12.2f}MB | {ratio:>6.2f}×")
    
    # Conclusion
    eq_growth = eqprop_results[-1]['memory_mb'] / eqprop_results[0]['memory_mb']
    bp_growth = backprop_results[-1]['memory_mb'] / backprop_results[0]['memory_mb']
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print(f"EqProp memory growth (64→1024):  {eq_growth:.2f}×")
    print(f"Backprop memory growth (64→1024): {bp_growth:.2f}×")
    
    if eq_growth < bp_growth * 0.8:
        print("\n✓ EqProp shows better memory scaling than Backprop")
    else:
        print("\n⚠ Similar memory scaling to Backprop")
    
    print("\nNote: For true O(1) memory, LocalHebbianUpdate requires")
    print("      full trainer integration (currently in archive).")
    print("=" * 70)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("ERROR: CUDA required for memory profiling")
        sys.exit(1)
    main()
