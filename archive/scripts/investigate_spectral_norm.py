#!/usr/bin/env python3
"""Investigate whether spectral normalization prevents Lipschitz explosion.

Key hypothesis: Training without spectral norm breaks contraction (L > 1).
Spectral norm should maintain L < 1 throughout training.
"""

import sys
sys.path.insert(0, '.')

import torch
import torch.optim as optim
import numpy as np

from src.models import LoopedMLP
from src.training import EqPropTrainer
from src.analysis import IterationAnalyzer, XORTask


def measure_lipschitz(model, x, n_samples=50):
    """Estimate Lipschitz constant via sampling."""
    batch_size = x.size(0)
    hidden_dim = model.hidden_dim
    device = x.device
    
    max_ratio = 0.0
    
    with torch.no_grad():
        for _ in range(n_samples):
            h1 = torch.randn(batch_size, hidden_dim, device=device)
            h2 = h1 + torch.randn_like(h1) * 0.1
            
            f_h1, _ = model.forward_step(h1, x, None)
            f_h2, _ = model.forward_step(h2, x, None)
            
            dist_h = torch.norm(h2 - h1, dim=-1)
            dist_f = torch.norm(f_h2 - f_h1, dim=-1)
            
            ratio = (dist_f / (dist_h + 1e-8)).max().item()
            max_ratio = max(max_ratio, ratio)
    
    return max_ratio


def train_and_track(model, task, epochs=30, lr=0.01, beta=0.22):
    """Train model and track Lipschitz throughout training."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    trainer = EqPropTrainer(model, optimizer, beta=beta, max_steps=30)
    
    X, Y = task.generate_data(500, device='cpu')
    X_test, Y_test = task.generate_data(100, device='cpu')
    
    history = {
        'epoch': [],
        'lipschitz': [],
        'accuracy': [],
        'loss': []
    }
    
    # Initial measurement
    model.eval()
    L_init = measure_lipschitz(model, X[:32])
    history['epoch'].append(0)
    history['lipschitz'].append(L_init)
    history['accuracy'].append(0)
    history['loss'].append(float('inf'))
    
    print(f"  Initial L = {L_init:.3f}")
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        
        # Mini-batch training
        batch_size = 32
        for i in range(0, len(X), batch_size):
            x_batch = X[i:i+batch_size]
            y_batch = Y[i:i+batch_size]
            metrics = trainer.step(x_batch, y_batch)
            total_loss += metrics['loss']
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            output = model(X_test, steps=30)
            preds = output.argmax(dim=-1)
            acc = (preds == Y_test).float().mean().item()
        
        # Measure Lipschitz
        L = measure_lipschitz(model, X[:32])
        
        history['epoch'].append(epoch)
        history['lipschitz'].append(L)
        history['accuracy'].append(acc)
        history['loss'].append(total_loss)
        
        if epoch % 5 == 0:
            print(f"  Epoch {epoch}: L={L:.3f}, Acc={acc:.1%}, Loss={total_loss:.2f}")
    
    return history


def main():
    print("=" * 70)
    print("SPECTRAL NORMALIZATION INVESTIGATION")
    print("=" * 70)
    
    task = XORTask(n_bits=4)
    
    # Test 1: Without spectral norm
    print("\n## WITHOUT Spectral Normalization")
    print("-" * 50)
    model_no_sn = LoopedMLP(task.input_dim, 64, task.output_dim, 
                            symmetric=True, use_spectral_norm=False)
    history_no_sn = train_and_track(model_no_sn, task, epochs=30)
    
    # Test 2: With spectral norm
    print("\n## WITH Spectral Normalization")
    print("-" * 50)
    model_sn = LoopedMLP(task.input_dim, 64, task.output_dim, 
                         symmetric=True, use_spectral_norm=True)
    history_sn = train_and_track(model_sn, task, epochs=30)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\nWithout Spectral Norm:")
    print(f"  Initial L:     {history_no_sn['lipschitz'][0]:.3f}")
    print(f"  Final L:       {history_no_sn['lipschitz'][-1]:.3f}")
    print(f"  Max L:         {max(history_no_sn['lipschitz']):.3f}")
    print(f"  Final Acc:     {history_no_sn['accuracy'][-1]:.1%}")
    print(f"  Contraction:   {'✓' if history_no_sn['lipschitz'][-1] < 1 else '✗ BROKEN'}")
    
    print(f"\nWith Spectral Norm:")
    print(f"  Initial L:     {history_sn['lipschitz'][0]:.3f}")
    print(f"  Final L:       {history_sn['lipschitz'][-1]:.3f}")
    print(f"  Max L:         {max(history_sn['lipschitz']):.3f}")
    print(f"  Final Acc:     {history_sn['accuracy'][-1]:.1%}")
    print(f"  Contraction:   {'✓' if history_sn['lipschitz'][-1] < 1 else '✗ BROKEN'}")
    
    # Save results
    import json
    results = {
        'without_spectral_norm': history_no_sn,
        'with_spectral_norm': history_sn
    }
    with open('/tmp/spectral_norm_investigation.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to /tmp/spectral_norm_investigation.json")
    
    # Recommendation
    print("\n" + "=" * 70)
    sn_maintains = history_sn['lipschitz'][-1] < 1
    if sn_maintains:
        print("✓ RECOMMENDATION: Use spectral_norm=True to maintain stability")
    else:
        print("⚠ Spectral norm alone is insufficient - investigate damping factor")
    print("=" * 70)


if __name__ == "__main__":
    main()
