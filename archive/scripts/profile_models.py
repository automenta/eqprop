#!/usr/bin/env python3
"""
Model Profiling Script

Profile EqProp models to identify performance bottlenecks.
Analyzes CPU/GPU time, memory usage, and operator-level costs.
"""

import argparse
from pathlib import Path
import torch
from torch.profiler import profile, ProfilerActivity, record_function

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import MODEL_REGISTRY, get_model


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def profile_model(model_name, input_dim=64, hidden_dim=128, output_dim=10, 
                  batch_size=32, steps=20):
    """Profile a model's forward pass."""
    
    print(f"\n{'='*60}")
    print(f"Profiling: {model_name}")
    print(f"{'='*60}")
    
    # Create model
    model = get_model(model_name, input_dim=input_dim, hidden_dim=hidden_dim, 
                     output_dim=output_dim).to(DEVICE)
    model.eval()
    
    # Sample input
    x = torch.randn(batch_size, input_dim, device=DEVICE)
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(x, steps=steps)
    
    # Profile
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if DEVICE == 'cuda' else [ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        with record_function("model_forward"):
            with torch.no_grad():
                output = model(x, steps=steps)
    
    # Print results
    print("\n📊 Top Operations by Time:")
    print(prof.key_averages().table(
        sort_by="cuda_time_total" if DEVICE == 'cuda' else "cpu_time_total",
        row_limit=15
    ))
    
    print("\n💾 Top Operations by Memory:")
    print(prof.key_averages().table(
        sort_by="self_cuda_memory_usage" if DEVICE == 'cuda' else "self_cpu_memory_usage",
        row_limit=10
    ))
    
    # Export trace for visualization
    trace_file = f"results/traces/{model_name}_trace.json"
    Path(trace_file).parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(trace_file)
    print(f"\n📁 Trace exported to: {trace_file}")
    print(f"   View at: chrome://tracing")
    
    return prof


def compare_models(models, input_dim=64, hidden_dim=128, batch_size=32, steps=20):
    """Compare profiling results across models."""
    
    print("\n" + "="*80)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*80)
    
    results = {}
    
    for model_name in models:
        try:
            prof = profile_model(model_name, input_dim, hidden_dim, 10, batch_size, steps)
            
            # Extract key metrics
            events = prof.key_averages()
            total_time = sum(e.self_cuda_time_total if DEVICE == 'cuda' else e.self_cpu_time_total 
                           for e in events) / 1000  # Convert to ms
            total_memory = sum(e.self_cuda_memory_usage if DEVICE == 'cuda' else e.self_cpu_memory_usage 
                             for e in events) / (1024**2)  # Convert to MB
            
            results[model_name] = {
                'time_ms': total_time,
                'memory_mb': total_memory,
                'params': sum(p.numel() for p in get_model(model_name, input_dim, hidden_dim, 10).parameters())
            }
        except Exception as e:
            print(f"  ⚠️  Failed to profile {model_name}: {e}")
            continue
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Model':<20} {'Time (ms)':>12} {'Memory (MB)':>14} {'Params':>12}")
    print("-"*80)
    
    for name, data in sorted(results.items(), key=lambda x: x[1]['time_ms']):
        print(f"{name:<20} {data['time_ms']:>11.2f} {data['memory_mb']:>13.2f} {data['params']:>11,}")
    
    return results


def identify_bottlenecks(model_name, **kwargs):
    """Identify specific bottlenecks in a model."""
    
    print(f"\n🔍 Bottleneck Analysis: {model_name}")
    print("="*60)
    
    prof = profile_model(model_name, **kwargs)
    events = prof.key_averages()
    
    # Find expensive operations
    total_time = sum(e.self_cuda_time_total if DEVICE == 'cuda' else e.self_cpu_time_total 
                    for e in events)
    
    bottlenecks = []
    for event in sorted(events, key=lambda x: x.self_cuda_time_total if DEVICE == 'cuda' else x.self_cpu_time_total, reverse=True)[:10]:
        pct = ((event.self_cuda_time_total if DEVICE == 'cuda' else event.self_cpu_time_total) / total_time) * 100
        if pct > 5:  # More than 5% of total time
            bottlenecks.append({
                'name': event.key,
                'time_pct': pct,
                'count': event.count
            })
    
    if bottlenecks:
        print("\n⚠️  Performance Bottlenecks (>5% time):")
        for b in bottlenecks:
            print(f"  • {b['name']}: {b['time_pct']:.1f}% ({b['count']} calls)")
            
        print("\n💡 Optimization Suggestions:")
        for b in bottlenecks:
            if 'linear' in b['name'].lower() or 'matmul' in b['name'].lower():
                print("  • Use fused linear-activation ops (e.g., F.linear + tanh in one kernel)")
            elif 'layernorm' in b['name'].lower():
                print("  • Consider GroupNorm or simplified normalization")
            elif 'copy' in b['name'].lower() or 'clone' in b['name'].lower():
                print("  • Use in-place operations where possible")
    else:
        print("✅ No major bottlenecks found (all ops <5% of total time)")
    
    return bottlenecks


def main():
    parser = argparse.ArgumentParser(description='Profile EqProp models')
    parser.add_argument('--model', type=str, default=None, help='Single model to profile')
    parser.add_argument('--models', type=str, default='top', 
                       help='Models to compare: top, all, or comma-separated')
    parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--steps', type=int, default=20, help='Equilibrium steps')
    parser.add_argument('--analyze', action='store_true', help='Run bottleneck analysis')
    args = parser.parse_args()
    
    if args.model:
        # Profile single model
        profile_model(args.model, hidden_dim=args.hidden_dim, 
                     batch_size=args.batch_size, steps=args.steps)
        
        if args.analyze:
            identify_bottlenecks(args.model, hidden_dim=args.hidden_dim,
                               batch_size=args.batch_size, steps=args.steps)
    else:
        # Compare multiple models
        if args.models == 'top':
            models = ['ModernEqProp', 'TPEqProp', 'SpectralTorEqProp', 'MSTEP']
        elif args.models == 'all':
            models = list(MODEL_REGISTRY.keys())[:9]  # Skip backprop-only models
        else:
            models = [m.strip() for m in args.models.split(',')]
        
        compare_models(models, hidden_dim=args.hidden_dim, 
                      batch_size=args.batch_size, steps=args.steps)


if __name__ == '__main__':
    main()
