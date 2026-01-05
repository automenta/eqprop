#!/usr/bin/env python3
"""Simplified profiling via manual timing."""

import sys
sys.path.insert(0, '.')

import torch
import torch.optim as optim
import time

from src.models import ModernEqProp, BackpropMLP
from src.training import EqPropTrainer, EquilibriumSolver
from src.tasks import get_task_loader


def time_component(name, func, *args, **kwargs):
    """Time a function call."""
    torch.cuda.synchronize()
    t0 = time.time()
    result = func(*args, **kwargs)
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    return result, elapsed


def profile_eqprop_vs_backprop():
    """Compare EqProp vs Backprop timing."""
    print("=" * 70)
    print("TIMING ANALYSIS: EqProp vs Backprop")
    print("=" * 70)
    
    # Setup
    train_loader, _, input_dim, output_dim = get_task_loader(
        'digits', batch_size=64, dataset_size=500
    )
    
    x, y = next(iter(train_loader))
    x, y = x.cuda(), y.cuda()
    
    # Warmup
    for _ in range(5):
        _ = x + 1
    
    # === BACKPROP ===
    print("\n## Backprop MLP")
    print("-" * 50)
    
    bp_model = BackpropMLP(input_dim, 256, output_dim, depth=2).cuda()
    bp_optimizer = optim.Adam(bp_model.parameters(), lr=0.001)
    bp_criterion = torch.nn.CrossEntropyLoss()
    
    times_bp = []
    for _ in range(10):
        bp_optimizer.zero_grad()
        _, t = time_component("forward", bp_model, x)
        out = bp_model(x)
        loss = bp_criterion(out, y)
        _, t_back = time_component("backward", loss.backward)
        _, t_opt = time_component("optimizer", bp_optimizer.step)
        times_bp.append(t + t_back + t_opt)
    
    avg_bp = sum(times_bp) / len(times_bp)
    print(f"Average time per batch: {avg_bp*1000:.2f}ms")
    
    # === EQPROP ===
    print("\n## EqProp (ModernEqProp)")
    print("-" * 50)
    
    eq_model = ModernEqProp(input_dim, 256, output_dim, use_spectral_norm=True).cuda()
    eq_optimizer = optim.Adam(eq_model.parameters(), lr=0.001)
    eq_trainer = EqPropTrainer(eq_model, eq_optimizer, beta=0.22, max_steps=25)
    
    times_eq = []
    for _ in range(10):
        eq_optimizer.zero_grad()
        _, t = time_component("trainer.step", eq_trainer.step, x, y)
        times_eq.append(t)
    
    avg_eq = sum(times_eq) / len(times_eq)
    print(f"Average time per batch: {avg_eq*1000:.2f}ms")
    
    # === BREAKDOWN ===
    print("\n## EqProp Component Breakdown")
    print("-" * 50)
    
    solver = EquilibriumSolver(epsilon=1e-5, max_steps=25)
    
    # Free phase
    (h_free, metrics_free), t_free = time_component("free_phase", solver.solve, eq_model, x)
    print(f"Free phase (25 steps):    {t_free*1000:.2f}ms")
    
    # Nudged phase
    h_free_var = h_free.detach().requires_grad_(True)
    y_hat = eq_model.Head(h_free_var)
    loss = torch.nn.functional.cross_entropy(y_hat, y)
    loss.backward()
    dL_dh = h_free_var.grad.detach()
    
    (h_nudged, metrics_nudged), t_nudged = time_component("nudged_phase", solver.solve,
        eq_model, x, h_init=h_free, nudging=True,
        target_grads=dL_dh, beta=0.22
    )
    print(f"Nudged phase (25 steps):  {t_nudged*1000:.2f}ms")
    
    # Backward
    def backward_step():
        E_free = eq_model.energy(h_free, x)
        E_nudged = eq_model.energy(h_nudged, x)
        surrogate = (E_nudged - E_free) / 0.22
        surrogate.backward()
    
    _, t_backward = time_component("backward", backward_step)
    print(f"Backward pass:            {t_backward*1000:.2f}ms")
    
    # Optimizer
    _, t_optim = time_component("optimizer", eq_optimizer.step)
    print(f"Optimizer step:           {t_optim*1000:.2f}ms")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Backprop:                 {avg_bp*1000:.2f}ms")
    print(f"EqProp:                   {avg_eq*1000:.2f}ms")
    print(f"Slowdown:                 {avg_eq/avg_bp:.1f}×")
    print()
    print(f"EqProp breakdown:")
    total_eq = t_free + t_nudged + t_backward + t_optim
    print(f"  Free+Nudged (50 steps): {(t_free+t_nudged)*1000:.2f}ms ({(t_free+t_nudged)/total_eq*100:.0f}%)")
    print(f"  Backward:               {t_backward*1000:.2f}ms ({t_backward/total_eq*100:.0f}%)")
    print(f"  Optimizer:              {t_optim*1000:.2f}ms ({t_optim/total_eq*100:.0f}%)")
    print()
    print(f"Bottleneck: {(t_free+t_nudged)/total_eq*100:.0f}% of time spent running 50 forward_step iterations")


def test_early_stopping():
    """Test speed improvement with early stopping."""
    print("\n" + "=" * 70)
    print("TESTING EARLY STOPPING OPTIMIZATION")
    print("=" * 70)
    
    train_loader, _, input_dim, output_dim = get_task_loader(
        'digits', batch_size=64, dataset_size=500
    )
    
    x, y = next(iter(train_loader))
    x, y = x.cuda(), y.cuda()
    
    model = ModernEqProp(input_dim, 256, output_dim, use_spectral_norm=True).cuda()
    
    # Test with different epsilon values
    configs = [
        (25, 1e-5, "Normal (max_steps=25, ε=1e-5)"),
        (50, 1e-5, "More steps (max_steps=50, ε=1e-5)"),
        (25, 1e-4, "Early stop (max_steps=25, ε=1e-4)"),
        (25, 1e-3, "Aggressive stop (max_steps=25, ε=1e-3)"),
    ]
    
    for max_steps, epsilon, desc in configs:
        solver = EquilibriumSolver(epsilon=epsilon, max_steps=max_steps)
        
        times = []
        steps_taken = []
        for _ in range(10):
            (h, metrics), t = time_component("solve", solver.solve, model, x)
            times.append(t)
            steps_taken.append(metrics['steps'])
        
        avg_time = sum(times) / len(times)
        avg_steps = sum(steps_taken) / len(steps_taken)
        
        print(f"{desc:40s}: {avg_time*1000:6.2f}ms, {avg_steps:4.1f} steps avg")


if __name__ == "__main__":
    profile_eqprop_vs_backprop()
    test_early_stopping()
