"""Test kernel learning"""
import numpy as np
from models.kernel import EqPropKernel

# Test kernel with simple data
np.random.seed(42)
X = np.random.randn(100, 64).astype(np.float32)
y = np.random.randint(0, 10, 100)

kernel = EqPropKernel(64, 64, 10, beta=0.22, lr=0.01, max_steps=30)

print('Initial loss:')
result = kernel.train_step(X, y)
print(f'Loss: {result["loss"]:.3f}, Acc: {result["accuracy"]*100:.1f}%')

print('\nAfter 20 steps:')
for i in range(20):
    result = kernel.train_step(X, y)
    if i % 5 == 4:
        print(f'Step {i+1}: Loss: {result["loss"]:.3f}, Acc: {result["accuracy"]*100:.1f}%')

print('\nFinal:')
print(f'Loss: {result["loss"]:.3f}, Acc: {result["accuracy"]*100:.1f}%')
