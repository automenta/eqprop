"""Script to test gradient equivalence at various beta values."""

import torch
import torch.nn as nn
from src.models import LoopedTransformerBlock
from src.solver import EquilibriumSolver
from src.trainer import EqPropTrainer
import numpy as np
import matplotlib.pyplot as plt


def compute_bp_gradient(model, output_head, x, y, h_eq):
    """Compute standard backprop gradient through equilibrium."""
    # Enable gradients for model parameters
    for param in model.parameters():
        param.requires_grad = True
    for param in output_head.parameters():
        param.requires_grad = True
    
    # Forward through equilibrium (already computed)
    y_pred = output_head(h_eq.mean(dim=0))
    loss = nn.functional.cross_entropy(y_pred, y)
    
    # Backward
    loss.backward()
    
    # Collect gradients
    bp_grads = []
    for param in model.parameters():
        if param.grad is not None:
            bp_grads.append(param.grad.clone().flatten())
    
    # Clear gradients
    model.zero_grad()
    output_head.zero_grad()
    
    return torch.cat(bp_grads) if bp_grads else torch.tensor([])


def compute_eqprop_gradient(model, solver, output_head, x, y, beta):
    """Compute EqProp gradient via contrastive Hebbian learning."""
    trainer = EqPropTrainer(model, solver, output_head, beta=beta, lr=1e-3)
    trainer.optimizer.zero_grad()
    
    # Run one EqProp step
    _ = trainer.train_step(x, y)
    
    # Collect gradients
    eqprop_grads = []
    for param in model.parameters():
        if param.grad is not None:
            eqprop_grads.append(param.grad.clone().flatten())
    
    return torch.cat(eqprop_grads) if eqprop_grads else torch.tensor([])


def test_gradient_equivalence(betas=[0.5, 0.1, 0.05, 0.01, 0.001]):
    """Test gradient equivalence across different beta values."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create simple model
    d_model = 64
    model = LoopedTransformerBlock(d_model, n_heads=4, d_ff=256, 
                                   attention_type='linear', symmetric=False).to(device)
    output_head = nn.Linear(d_model, 10).to(device)
    solver = EquilibriumSolver(max_iters=50, tol=1e-5, damping=0.9)
    
    # Create dummy data
    batch_size = 32
    x = torch.randn(1, batch_size, d_model).to(device)
    y = torch.randint(0, 10, (batch_size,)).to(device)
    
    # Compute equilibrium once
    with torch.no_grad():
        h0 = torch.zeros_like(x)
        h_eq, _ = solver.solve(model, h0, x)
    
    results = []
    
    print("="*70)
    print("Gradient Equivalence Test: EqProp vs Backprop")
    print("="*70)
    print(f"{'Beta':<10} {'Cosine Sim':<15} {'L2 Error':<15} {'Status':<10}")
    print("-"*70)
    
    for beta in betas:
        # Compute BP gradient
        bp_grad = compute_bp_gradient(model.copy() if hasattr(model, 'copy') else model, 
                                       output_head, x, y, h_eq)
        
        # Compute EqProp gradient
        eq_grad = compute_eqprop_gradient(model, solver, output_head, x, y, beta)
        
        # Scale EqProp gradient by beta (theory: should match BP as beta->0)
        eq_grad_scaled = eq_grad * beta
        
        # Compute similarity
        if bp_grad.numel() > 0 and eq_grad_scaled.numel() > 0:
            cosine_sim = nn.functional.cosine_similarity(
                bp_grad.unsqueeze(0), eq_grad_scaled.unsqueeze(0)
            ).item()
            l2_error = (bp_grad - eq_grad_scaled).norm().item()
        else:
            cosine_sim = 0.0
            l2_error = float('inf')
        
        status = "✓" if cosine_sim > 0.95 else "✗"
        
        print(f"{beta:<10.4f} {cosine_sim:<15.4f} {l2_error:<15.4f} {status:<10}")
        
        results.append({
            'beta': beta,
            'cosine_sim': cosine_sim,
            'l2_error': l2_error
        })
    
    print("-"*70)
    print(f"\nSuccess Criterion: Cosine similarity > 0.95 at small beta")
    
    return results


def plot_results(results, save_path="figures/gradient_equivalence.png"):
    """Plot gradient equivalence vs beta."""
    import os
    os.makedirs("figures", exist_ok=True)
    
    betas = [r['beta'] for r in results]
    cosine_sims = [r['cosine_sim'] for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.semilogx(betas, cosine_sims, 'o-', linewidth=2, markersize=8)
    plt.axhline(y=0.99, color='g', linestyle='--', label='Target (0.99)')
    plt.axhline(y=0.95, color='orange', linestyle='--', label='Acceptable (0.95)')
    plt.xlabel('Beta (nudge strength)', fontsize=12)
    plt.ylabel('Cosine Similarity', fontsize=12)
    plt.title('Gradient Equivalence: EqProp vs Backprop', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\nPlot saved to {save_path}")


if __name__ == "__main__":
    results = test_gradient_equivalence()
    plot_results(results)
