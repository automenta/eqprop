import sys
sys.path.insert(0, '.')
import torch
from src.tasks import get_task_loader
from src.models.looped_mlp import LoopedMLP
from src.training import EqPropTrainer
import time

print("Testing Fashion-MNIST loader...")
try:
    train_loader, test_loader, in_dim, out_dim = get_task_loader(
        'fashion-mnist', batch_size=64, dataset_size=1000
    )
    print(f"✅ Loader success: {in_dim}->{out_dim}")
except Exception as e:
    print(f"❌ Loader failed: {e}")
    exit(1)

print("\nInitializing model...")
model = LoopedMLP(in_dim, 128, out_dim, symmetric=True, use_spectral_norm=True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
trainer = EqPropTrainer(model, optimizer, beta=0.22, max_steps=20)

print("\nRunning training loop (1 epoch)...")
model.train()
start = time.time()
for i, (x, y) in enumerate(train_loader):
    trainer.step(x, y)
    if i % 5 == 0:
        print(f"  Batch {i} processed")
elapsed = time.time() - start
print(f"✅ Training loop success ({elapsed:.2f}s)")
