"""
Enhanced comparison: Linearly separable problem for clearer demonstration
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.kernel import EqPropKernel, cross_entropy

np.random.seed(42)
torch.manual_seed(42)

# Simple linearly separable data (binary classification)
X_np = np.random.randn(100, 10).astype(np.float32)
w_true = np.random.randn(10).astype(np.float32)
y_np = (X_np @ w_true > 0).astype(np.int64)

X_torch = torch.from_numpy(X_np)
y_torch = torch.from_numpy(y_np)

input_dim, hidden_dim, output_dim = 10, 16, 2

# ============================================================================
# PyTorch Implementation  
# ============================================================================

class SimplePyTorchEqProp(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_in = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_rec = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_out = nn.Linear(hidden_dim, output_dim, bias=False)
        
        nn.init.xavier_uniform_(self.W_in.weight, gain=0.5)
        nn.init.xavier_uniform_(self.W_rec.weight, gain=0.5)
        nn.init.xavier_uniform_(self.W_out.weight, gain=0.5)
    
    def forward(self, x, steps=20):
        h = torch.zeros(x.size(0), hidden_dim)
        x_proj = self.W_in(x)
        for _ in range(steps):
            h = torch.tanh(x_proj + self.W_rec(h))
        return self.W_out(h)

pt_model = SimplePyTorchEqProp()
pt_opt = torch.optim.SGD(pt_model.parameters(), lr=0.05)

# ============================================================================
# Kernel Implementation (same initialization)
# ============================================================================

kernel = EqPropKernel(input_dim, hidden_dim, output_dim, lr=0.05, max_steps=20)

kernel.W_in = pt_model.W_in.weight.detach().numpy().copy()
kernel.W_rec = pt_model.W_rec.weight.detach().numpy().copy()
kernel.W_out = pt_model.W_out.weight.detach().numpy().copy()

# ============================================================================
# Side-by-side training
# ============================================================================

print("=" * 80)
print("SIDE-BY-SIDE TRAINING COMPARISON")
print("=" * 80)
print(f"{'Epoch':<8} {'PyTorch Loss':<15} {'PyTorch Acc':<15} {'Kernel Loss':<15} {'Kernel Acc'}")
print("-" * 80)

for epoch in range(20):
    # PyTorch step
    pt_opt.zero_grad()
    pt_logits = pt_model(X_torch)
    pt_loss = F.cross_entropy(pt_logits, y_torch)
    pt_loss.backward()
    pt_opt.step()
    
    with torch.no_grad():
        pt_preds = pt_logits.argmax(dim=1)
        pt_acc = (pt_preds == y_torch).float().mean()
    
    # Kernel step
    kernel_result = kernel.train_step(X_np, y_np)
    
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"{epoch+1:<8} {pt_loss.item():<15.4f} {pt_acc.item()*100:<14.1f}% "
              f"{kernel_result['loss']:<15.4f} {kernel_result['accuracy']*100:.1f}%")

print("\n" + "=" * 80)
print("FINAL COMPARISON")
print("=" * 80)

# Final evaluation
with torch.no_grad():
    pt_final_logits = pt_model(X_torch)
    pt_final_preds = pt_final_logits.argmax(dim=1)
    pt_final_acc = (pt_final_preds == y_torch).float().mean()
    pt_final_loss = F.cross_entropy(pt_final_logits, y_torch)

kernel_final = kernel.evaluate(X_np, y_np)

print(f"PyTorch: Loss={pt_final_loss.item():.4f}, Acc={pt_final_acc.item()*100:.1f}%")
print(f"Kernel:  Loss={kernel_final['loss']:.4f}, Acc={kernel_final['accuracy']*100:.1f}%")

# Show weight difference
print("\n" + "=" * 80)
print("WEIGHT DIVERGENCE")
print("=" * 80)

W_out_diff = np.linalg.norm(pt_model.W_out.weight.detach().numpy() - kernel.W_out)
W_rec_diff = np.linalg.norm(pt_model.W_rec.weight.detach().numpy() - kernel.W_rec)

print(f"W_out difference: {W_out_diff:.4f}")
print(f"W_rec difference: {W_rec_diff:.4f}")
