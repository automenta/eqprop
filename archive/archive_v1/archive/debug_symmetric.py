"""Quick diagnostic script to check symmetric vs non-symmetric dynamics."""
import torch
from src.models import LoopedTransformerBlock
from src.solver import EquilibriumSolver

torch.manual_seed(42)

# Test symmetric dynamics
model = LoopedTransformerBlock(64, 4, 256, attention_type='linear', symmetric=True)
solver = EquilibriumSolver(max_iters=50, tol=1e-5, damping=0.9)

x = torch.randn(1, 8, 64)
h0 = torch.zeros_like(x)

h_eq, iters = solver.solve(model, h0, x)
print(f'Symmetric mode:')
print(f'  Iterations: {iters}')
print(f'  Mean: {h_eq.mean().item():.4f}, Std: {h_eq.std().item():.4f}')
print(f'  Min: {h_eq.min().item():.4f}, Max: {h_eq.max().item():.4f}')

# Check saturation
saturated = (h_eq.abs() > 0.9).float().mean()
print(f'  Saturation (|h|>0.9): {saturated.item()*100:.1f}%')

# Non-symmetric for comparison
print()
model2 = LoopedTransformerBlock(64, 4, 256, attention_type='linear', symmetric=False) 
h_eq2, iters2 = solver.solve(model2, h0, x)
print(f'Non-symmetric mode:')
print(f'  Iterations: {iters2}')
print(f'  Mean: {h_eq2.mean().item():.4f}, Std: {h_eq2.std().item():.4f}')
print(f'  Min: {h_eq2.min().item():.4f}, Max: {h_eq2.max().item():.4f}')
