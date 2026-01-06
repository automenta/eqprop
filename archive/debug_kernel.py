"""Debug kernel gradients in detail"""
import numpy as np
from models.kernel import EqPropKernel, softmax

np.random.seed(42)
X = np.random.randn(10, 64).astype(np.float32)  # Small batch
y = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])  # One per class

kernel = EqPropKernel(64, 32, 10, beta=0.5, lr=0.1, max_steps=20)

# Single step with detailed output
W_in, W_rec, W_out = kernel._get_weights()

# Free phase
h_free, _ = kernel.solve_equilibrium(X)
logits_free = h_free @ W_out.T + kernel.b_out
loss_free = -np.mean(np.log(softmax(logits_free)[np.arange(10), y] + 1e-8))

# Supervised gradient
probs = softmax(logits_free)
one_hot = np.zeros_like(probs)
one_hot[np.arange(10), y] = 1.0
d_logits = probs - one_hot

# Output gradient (this should decrease loss)
dW_out_supervised = d_logits.T @ h_free / 10
print(f"Loss (free): {loss_free:.3f}")
print(f"dW_out norm: {np.linalg.norm(dW_out_supervised):.3f}")

# Apply just W_out update
kernel.W_out -= 0.1 * dW_out_supervised
logits_new = h_free @ kernel.W_out.T + kernel.b_out
loss_new = -np.mean(np.log(softmax(logits_new)[np.arange(10), y] + 1e-8))
print(f"Loss (after W_out update): {loss_new:.3f}")
print(f"Loss decreased: {loss_free - loss_new > 0}")

# Reset and try nudged phase
kernel.W_out = W_out.copy()
nudge_grad = d_logits @ W_out
h_nudged, _ = kernel.solve_equilibrium(X, nudge_grad)

dh = (h_nudged - h_free) / kernel.beta
print(f"\ndh norm: {np.linalg.norm(dh):.3f}")
print(f"dh mean: {dh.mean():.3f}")

# Test W_rec update
dW_rec = dh.T @ h_free / 10
print(f"dW_rec norm: {np.linalg.norm(dW_rec):.3f}")
