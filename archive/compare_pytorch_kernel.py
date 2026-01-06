"""
Side-by-side comparison of PyTorch EqProp vs NumPy Kernel on minimal problem.

Goal: Understand any differences in learning behavior.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# MINIMAL PROBLEM: 2D input, 4D hidden, 2 classes
# ============================================================================

np.random.seed(42)
torch.manual_seed(42)

# Simple linearly separable data
X_np = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
], dtype=np.float32)

y_np = np.array([0, 1, 1, 0])  # XOR-like

X_torch = torch.from_numpy(X_np)
y_torch = torch.from_numpy(y_np)

input_dim, hidden_dim, output_dim = 2, 4, 2

# ============================================================================
# PyTorch Implementation
# ============================================================================

class SimplePyTorchEqProp(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_in = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_rec = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_out = nn.Linear(hidden_dim, output_dim, bias=False)
        
        # Initialize to match kernel
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
pt_opt = torch.optim.SGD(pt_model.parameters(), lr=0.1)

print("=" * 70)
print("PYTORCH TRAINING")
print("=" * 70)

for epoch in range(10):
    pt_opt.zero_grad()
    logits = pt_model(X_torch)
    loss = F.cross_entropy(logits, y_torch)
    loss.backward()
    pt_opt.step()
    
    with torch.no_grad():
        preds = logits.argmax(dim=1)
        acc = (preds == y_torch).float().mean()
    
    print(f"Epoch {epoch+1}: Loss={loss.item():.4f}, Acc={acc.item()*100:.1f}%")

# ============================================================================
# NumPy Kernel Implementation
# ============================================================================

from models.kernel import EqPropKernel

kernel = EqPropKernel(input_dim, hidden_dim, output_dim, beta=0.5, lr=0.1, 
                      use_spectral_norm=False, max_steps=20)

# Match PyTorch initialization (note: PyTorch Linear has weight as [out, in])
# Kernel expects [out, in] as well, so just copy directly
kernel.W_in = pt_model.W_in.weight.detach().numpy().copy()
kernel.W_rec = pt_model.W_rec.weight.detach().numpy().copy()
kernel.W_out = pt_model.W_out.weight.detach().numpy().copy()

print("\n" + "=" * 70)
print("NUMPY KERNEL TRAINING (same initialization)")
print("=" * 70)

for epoch in range(10):
    result = kernel.train_step(X_np, y_np)
    print(f"Epoch {epoch+1}: Loss={result['loss']:.4f}, Acc={result['accuracy']*100:.1f}%")

# ============================================================================
# DETAILED COMPARISON OF SINGLE STEP
# ============================================================================

print("\n" + "=" * 70)
print("DETAILED SINGLE-STEP COMPARISON")
print("=" * 70)

# Reset both to same weights
torch.manual_seed(42)
np.random.seed(42)

pt_model2 = SimplePyTorchEqProp()
kernel2 = EqPropKernel(input_dim, hidden_dim, output_dim, beta=0.5, lr=0.1,
                       use_spectral_norm=False, max_steps=20)

# Copy weights
kernel2.W_in = pt_model2.W_in.weight.detach().numpy().copy()
kernel2.W_rec = pt_model2.W_rec.weight.detach().numpy().copy()
kernel2.W_out = pt_model2.W_out.weight.detach().numpy().copy()

print("\n1. Initial forward pass:")
with torch.no_grad():
    pt_logits = pt_model2(X_torch)
    pt_loss = F.cross_entropy(pt_logits, y_torch)
    print(f"   PyTorch: Loss={pt_loss.item():.4f}")

kernel_logits = kernel2.predict(X_np)
from models.kernel import cross_entropy
kernel_loss = cross_entropy(kernel_logits, y_np)
print(f"   Kernel:  Loss={kernel_loss:.4f}")

print("\n2. Computing gradients:")

# PyTorch gradients
pt_model2.zero_grad()
pt_logits = pt_model2(X_torch)
pt_loss = F.cross_entropy(pt_logits, y_torch)
pt_loss.backward()

print(f"   PyTorch W_out grad norm: {pt_model2.W_out.weight.grad.norm().item():.4f}")
print(f"   PyTorch W_rec grad norm: {pt_model2.W_rec.weight.grad.norm().item():.4f}")

# Kernel gradients (one step)
result = kernel2.train_step(X_np, y_np)
print(f"   Kernel loss after step: {result['loss']:.4f}")

print("\n3. After one update:")
pt_opt2 = torch.optim.SGD(pt_model2.parameters(), lr=0.1)
pt_opt2.step()

with torch.no_grad():
    pt_logits_new = pt_model2(X_torch)
    pt_loss_new = F.cross_entropy(pt_logits_new, y_torch)
    print(f"   PyTorch: Loss={pt_loss_new.item():.4f} (Δ={pt_loss_new.item()-pt_loss.item():+.4f})")

print(f"   Kernel:  Loss={result['loss']:.4f} (Δ={result['loss']-kernel_loss:+.4f})")
