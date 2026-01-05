#!/usr/bin/env python3
"""
Quick test to verify training works and see immediate output.
"""
import sys
import os

# Force unbuffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

print("🚀 Starting quick training test...", flush=True)
print("="*70, flush=True)

from src.config import TorEqPropConfig

# Override sys.argv for testing
sys.argv = ['test', '--dataset', 'mnist', '--epochs', '1', '--rapid', '--no-compile']
config = TorEqPropConfig.from_args()

print(f"✓ Config loaded: d_model={config.d_model}, beta={config.beta}, tol={config.tol} (type: {type(config.tol)})", flush=True)
print(f"✓ Device: {config.device}", flush=True)

# Import training components
print("Loading data...", flush=True)
from src.datasets import get_data_loaders
train_loader, test_loader, input_dim, num_classes = get_data_loaders(
    config.dataset, batch_size=config.batch_size, num_workers=0
)
print(f"✓ Data loaded: {len(train_loader)} batches", flush=True)

# Create model
print("Creating model...", flush=True)
import torch
import torch.nn as nn
from src.models import LoopedTransformerBlock
from src.solver import EquilibriumSolver
from src.trainer import EqPropTrainer

embedding = nn.Linear(input_dim, config.d_model).to(config.device)
model = LoopedTransformerBlock(
    config.d_model, config.n_heads, config.d_ff,
    dropout=config.dropout,
    attention_type=config.attention_type,
    symmetric=config.symmetric
).to(config.device)
output_head = nn.Linear(config.d_model, num_classes).to(config.device)
print(f"✓ Model created", flush=True)

# Create solver
print(f"Creating solver (tol={config.tol}, type={type(config.tol)})...", flush=True)
solver = EquilibriumSolver(
    max_iters=config.max_iters,
    tol=config.tol,
    damping=config.damping
)
print(f"✓ Solver created (solver.tol={solver.tol}, type={type(solver.tol)})", flush=True)

# Create trainer  
trainer = EqPropTrainer(
    model, solver, output_head,
    beta=config.beta, lr=config.lr,
    update_mode=config.update_mode
)
trainer.optimizer.add_param_group({'params': embedding.parameters()})
print(f"✓ Trainer created", flush=True)

# Train one epoch
print("="*70, flush=True)
print("Training epoch 0...", flush=True)
print("="*70, flush=True)

model.train()
embedding.train()
output_head.train()

for batch_idx, (data, target) in enumerate(train_loader):
    if batch_idx >= 5:  # Just do 5 batches for quick test
        break
    
    data, target = data.to(config.device), target.to(config.device)
    x_emb = embedding(data).unsqueeze(0)
    
    print(f"  Batch {batch_idx}: Running train_step...", flush=True)
    metrics = trainer.train_step(x_emb, target)
    
    print(f"  ✓ Batch {batch_idx}: Loss={metrics['loss']:.4f}, "
          f"Acc={metrics['accuracy']:.4f}, "
          f"Iters={metrics['iters_free']}/{metrics['iters_nudged']}", flush=True)

print("="*70, flush=True)
print("✅ SUCCESS! Training works!", flush=True)
print("="*70, flush=True)
