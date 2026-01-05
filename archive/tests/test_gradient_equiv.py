"""Gradient Equivalence Test for Equilibrium Propagation.

Verifies that EqProp gradients approximate backpropagation gradients
with high cosine similarity (target: >0.99 for symmetric models).

Reference: Scellier & Bengio (2017), "Equilibrium Propagation: Bridging the
Gap between Energy-Based Models and Backpropagation"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.models import LoopedMLP
from src.training import EquilibriumSolver


def cosine_similarity(v1, v2):
    """Compute cosine similarity between two tensors."""
    v1_flat = v1.flatten()
    v2_flat = v2.flatten()
    dot = torch.dot(v1_flat, v2_flat)
    norm1 = torch.norm(v1_flat)
    norm2 = torch.norm(v2_flat)
    return (dot / (norm1 * norm2 + 1e-8)).item()


def test_gradient_equivalence(symmetric=False, verbose=True):
    """Test EqProp gradient equivalence with backpropagation.
    
    Args:
        symmetric: If True, use symmetric weights (required for EqProp theory)
        verbose: Print detailed results
        
    Returns:
        Average cosine similarity across all parameters
    """
    torch.manual_seed(42)
    
    # Small model for testing
    input_dim = 64
    hidden_dim = 32
    output_dim = 10
    batch_size = 16
    beta = 0.1
    
    # Create model
    model = LoopedMLP(input_dim, hidden_dim, output_dim, alpha=0.5, symmetric=symmetric)
    solver = EquilibriumSolver(epsilon=1e-5, max_steps=100)
    
    # Generate test data
    x = torch.randn(batch_size, input_dim)
    y = torch.randint(0, output_dim, (batch_size,))
    
    # ===== Method 1: Standard Backpropagation =====
    model_bp = LoopedMLP(input_dim, hidden_dim, output_dim, alpha=0.5, symmetric=symmetric)
    # Copy weights
    model_bp.load_state_dict(model.state_dict())
    
    # Forward with many steps (simulating equilibrium)
    out_bp = model_bp(x, steps=50)
    loss_bp = F.cross_entropy(out_bp, y)
    loss_bp.backward()
    
    bp_grads = {}
    for name, param in model_bp.named_parameters():
        if param.grad is not None:
            bp_grads[name] = param.grad.clone()
    
    # ===== Method 2: Equilibrium Propagation =====
    model.zero_grad()
    
    # Free phase: find equilibrium
    h_free, info_free = solver.solve(model, x)
    h_free = h_free.detach()
    
    # Compute nudging gradient
    h_free_var = h_free.clone().requires_grad_(True)
    y_hat = model.Head(h_free_var)
    loss_eq = F.cross_entropy(y_hat, y)
    loss_eq.backward()
    dL_dh = h_free_var.grad.detach()
    
    # Nudged phase
    h_nudged, info_nudged = solver.solve(
        model, x, h_init=h_free, nudging=True, target_grads=dL_dh, beta=beta
    )
    h_nudged = h_nudged.detach()
    
    # Compute contrastive gradients via energy
    E_free = model.energy(h_free, x)
    E_nudged = model.energy(h_nudged, x)
    surrogate_loss = (E_nudged - E_free) / beta
    surrogate_loss.backward()
    
    eq_grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            eq_grads[name] = param.grad.clone()
    
    # ===== Compare Gradients =====
    similarities = []
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Gradient Equivalence Test ({'Symmetric' if symmetric else 'Non-Symmetric'} Mode)")
        print(f"{'='*60}")
        print(f"Free phase converged: {info_free.get('converged', 'N/A')} in {info_free['steps']} steps")
        print(f"Nudged phase: {info_nudged['steps']} steps")
        print(f"\n{'Parameter':<30} {'Cosine Sim':>12} {'BP Norm':>12} {'EQ Norm':>12}")
        print("-" * 70)
    
    for name in bp_grads:
        if name in eq_grads:
            sim = cosine_similarity(bp_grads[name], eq_grads[name])
            similarities.append(sim)
            
            if verbose:
                bp_norm = torch.norm(bp_grads[name]).item()
                eq_norm = torch.norm(eq_grads[name]).item()
                status = "✓" if sim > 0.9 else "✗"
                print(f"{name:<30} {sim:>12.4f} {bp_norm:>12.4f} {eq_norm:>12.4f} {status}")
    
    avg_sim = np.mean(similarities) if similarities else 0.0
    
    if verbose:
        print("-" * 70)
        print(f"{'Average Cosine Similarity:':<30} {avg_sim:>12.4f}")
        print(f"\nTarget: >0.99 for symmetric mode, >0.90 for non-symmetric")
        
        if symmetric and avg_sim > 0.99:
            print("✓ PASSED: Symmetric mode gradient equivalence verified")
        elif not symmetric and avg_sim > 0.90:
            print("✓ PASSED: Non-symmetric mode shows reasonable alignment")
        else:
            print("✗ FAILED: Gradient equivalence below threshold")
    
    return avg_sim


def test_symmetric_vs_nonsymmetric():
    """Compare gradient equivalence between symmetric and non-symmetric modes."""
    print("\n" + "=" * 70)
    print("COMPARISON: Symmetric vs Non-Symmetric Gradient Equivalence")
    print("=" * 70)
    
    sim_nonsym = test_gradient_equivalence(symmetric=False, verbose=True)
    sim_sym = test_gradient_equivalence(symmetric=True, verbose=True)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Non-Symmetric Mode: {sim_nonsym:.4f}")
    print(f"Symmetric Mode:     {sim_sym:.4f}")
    print(f"Improvement:        {(sim_sym - sim_nonsym):.4f}")
    
    return sim_nonsym, sim_sym


if __name__ == "__main__":
    test_symmetric_vs_nonsymmetric()
