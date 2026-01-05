#!/usr/bin/env python3
"""Quick comparison of all models with and without spectral norm."""

import sys
sys.path.insert(0, '.')

import torch
import torch.optim as optim

import argparse
import numpy as np

from src.models import LoopedMLP, ToroidalMLP, ModernEqProp
from src.training import EqPropTrainer
from src.analysis import XORTask


def quick_train(model, task, epochs=15, lr=0.01, beta=0.22):
    """Train and return final (Lipschitz, accuracy)."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    trainer = EqPropTrainer(model, optimizer, beta=beta, max_steps=30)
    
    X, Y = task.generate_data(400, device='cpu')
    X_test, Y_test = task.generate_data(100, device='cpu')
    
    for epoch in range(epochs):
        model.train()
        batch_size = 32
        for i in range(0, len(X), batch_size):
            trainer.step(X[i:i+batch_size], Y[i:i+batch_size])
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        output = model(X_test, steps=30)
        preds = output.argmax(dim=-1)
        acc = (preds == Y_test).float().mean().item()
    
    # Measure Lipschitz
    L = measure_lipschitz(model, X[:32])
    
    return L, acc


def measure_lipschitz(model, x, n_samples=30):
    """Quick Lipschitz estimate."""
    hidden_dim = model.hidden_dim
    device = x.device
    batch_size = x.size(0)
    
    max_ratio = 0.0
    with torch.no_grad():
        for _ in range(n_samples):
            h1 = torch.randn(batch_size, hidden_dim, device=device)
            h2 = h1 + torch.randn_like(h1) * 0.1
            
            # Handle stacked state for ToroidalMLP
            if hasattr(model, 'buffer_size'):
                zeros = torch.zeros(batch_size, model.buffer_size, hidden_dim, device=device)
                h1 = torch.cat([h1.unsqueeze(1), zeros], dim=1)
                h2 = torch.cat([h2.unsqueeze(1), zeros], dim=1)
            
            f_h1, _ = model.forward_step(h1, x, None)
            f_h2, _ = model.forward_step(h2, x, None)
            
            if f_h1.dim() == 3:
                f_h1, f_h2 = f_h1[:, 0], f_h2[:, 0]
                h1, h2 = h1[:, 0], h2[:, 0]
            
            dist_h = torch.norm(h2 - h1, dim=-1)
            dist_f = torch.norm(f_h2 - f_h1, dim=-1)
            
            ratio = (dist_f / (dist_h + 1e-8)).max().item()
            max_ratio = max(max_ratio, ratio)
    
    return max_ratio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=1, help='Number of seeds')
    args = parser.parse_args()

    print("=" * 70)
    print(f"SPECTRAL NORM: UNIVERSAL BENEFIT TEST ({args.seeds} seeds)")
    print("=" * 70)
    
    task = XORTask(n_bits=4)
    
    results = []
    
    for name, ModelClass, kwargs_base in [
        ("LoopedMLP", LoopedMLP, {"symmetric": True}),
        ("ToroidalMLP", ToroidalMLP, {}),
        ("ModernEqProp", ModernEqProp, {}),
    ]:
        print(f"\n## {name}")
        print("-" * 50)
        
        l_no_list, acc_no_list = [], []
        l_yes_list, acc_yes_list = [], []
        
        for seed in range(42, 42 + args.seeds):
            # Without SN
            torch.manual_seed(seed)
            model_no = ModelClass(task.input_dim, 64, task.output_dim, 
                                  use_spectral_norm=False, **kwargs_base)
            L_no, acc_no = quick_train(model_no, task)
            l_no_list.append(L_no)
            acc_no_list.append(acc_no)
            
            # With SN
            torch.manual_seed(seed)
            model_yes = ModelClass(task.input_dim, 64, task.output_dim, 
                                   use_spectral_norm=True, **kwargs_base)
            L_yes, acc_yes = quick_train(model_yes, task)
            l_yes_list.append(L_yes)
            acc_yes_list.append(acc_yes)
        
        # Average
        avg_L_no = np.mean(l_no_list)
        avg_acc_no = np.mean(acc_no_list)
        avg_L_yes = np.mean(l_yes_list)
        avg_acc_yes = np.mean(acc_yes_list)
        
        print(f"  Without SN: L={avg_L_no:.3f}, Acc={avg_acc_no:.1%}")
        print(f"  With SN:    L={avg_L_yes:.3f}, Acc={avg_acc_yes:.1%}")
        print(f"  L reduced:  {avg_L_no - avg_L_yes:.3f}")
        print(f"  Contraction: {'✓' if avg_L_yes < 1 else '✗'}")
        
        results.append({
            "model": name,
            "L_without_sn": avg_L_no,
            "L_with_sn": avg_L_yes,
            "acc_without_sn": avg_acc_no,
            "acc_with_sn": avg_acc_yes,
            "contraction_maintained": avg_L_yes < 1
        })
    
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Model':<15} | {'L (no SN)':<10} | {'L (SN)':<10} | {'Δ':<8} | {'Contraction'}")
    print("-" * 70)
    for r in results:
        delta = r["L_without_sn"] - r["L_with_sn"]
        status = "✓" if r["contraction_maintained"] else "✗"
        print(f"{r['model']:<15} | {r['L_without_sn']:<10.3f} | {r['L_with_sn']:<10.3f} | {delta:<8.3f} | {status}")
    
    all_maintained = all(r["contraction_maintained"] for r in results)
    print("\n" + "=" * 70)
    if all_maintained:
        print("✓ SPECTRAL NORM UNIVERSALLY MAINTAINS CONTRACTION")
    else:
        print("⚠ Some models still break contraction - investigate further")
    print("=" * 70)

    # Save results
    import json
    
    # Convert numpy types to python types
    clean_results = []
    for r in results:
        clean_r = {}
        for k, v in r.items():
            if hasattr(v, 'item'):
                clean_r[k] = v.item()
            else:
                clean_r[k] = v
        clean_results.append(clean_r)

    with open('/tmp/lipschitz_analysis.json', 'w') as f:
        json.dump(clean_results, f, indent=2)
    print("Results saved to /tmp/lipschitz_analysis.json")

if __name__ == "__main__":
    main()
