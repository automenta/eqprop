"""Training script for micro tasks - designed for rapid EqProp exploration.

Use tiny models (d_model=8-32) and micro tasks (XOR, AND, etc.) to quickly
explore EqProp's hyperparameter space and understand its behavior.

Usage:
    # Quick XOR test
    python train_micro.py --task xor --d-model 8 --epochs 10
    
    # Test with EqProp
    python train_micro.py --task xor3 --d-model 16 --epochs 20 --beta 0.22
    
    # Baseline comparison
    python train_micro.py --task tiny_lm --d-model 32 --epochs 30 --use-bp
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from pathlib import Path

from src.micro_tasks import get_micro_loader, MICRO_TASK_INFO
from src.models import LoopedTransformerBlock
from src.solver import EquilibriumSolver
from src.trainer import EqPropTrainer

# Global PyTorch optimizations
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


def create_tiny_model(input_dim: int, d_model: int, num_classes: int, device: str):
    """Create a tiny model for micro tasks."""
    # For very small d_model, use fewer heads
    n_heads = max(1, min(4, d_model // 4))  # 1-4 heads depending on d_model
    d_ff = max(d_model, d_model * 2)  # At least d_model for FFN
    
    embedding = nn.Linear(input_dim, d_model).to(device)
    model = LoopedTransformerBlock(
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=0.0,  # No dropout for tiny tasks
        attention_type="linear",
        symmetric=False
    ).to(device)
    output_head = nn.Linear(d_model, num_classes).to(device)
    
    return embedding, model, output_head, n_heads, d_ff


def train_eqprop(args, train_loader, test_loader, input_dim, num_classes):
    """Train with Equilibrium Propagation."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    embedding, model, output_head, n_heads, d_ff = create_tiny_model(
        input_dim, args.d_model, num_classes, device
    )
    
    solver = EquilibriumSolver(
        max_iters=args.max_iters,
        tol=args.tol,
        damping=args.damping
    )
    
    trainer = EqPropTrainer(
        model, solver, output_head,
        beta=args.beta,
        lr=args.lr,
        update_mode=args.update_mode
    )
    trainer.optimizer.add_param_group({'params': embedding.parameters()})
    
    print(f"\n{'='*60}")
    print(f"Training MICRO TASK with EqProp")
    print(f"{'='*60}")
    print(f"Task: {args.task}")
    print(f"Model: d_model={args.d_model}, n_heads={n_heads}, d_ff={d_ff}")
    print(f"EqProp: β={args.beta}, damping={args.damping}, max_iters={args.max_iters}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    best_acc = 0
    start_time = time.time()
    
    for epoch in range(args.epochs):
        model.train()
        embedding.train()
        output_head.train()
        
        total_loss = 0
        total_acc = 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            x_emb = embedding(data).unsqueeze(0)
            metrics = trainer.train_step(x_emb, target)
            total_loss += metrics["loss"]
            total_acc += metrics["accuracy"]
        
        train_loss = total_loss / len(train_loader)
        train_acc = total_acc / len(train_loader)
        
        # Evaluate
        model.eval()
        test_acc = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                x_emb = embedding(data).unsqueeze(0)
                h0 = torch.zeros_like(x_emb)
                h_fixed, _ = solver.solve(model, h0, x_emb)
                y_pred = output_head(h_fixed.mean(dim=0))
                test_acc += (y_pred.argmax(-1) == target).float().mean().item()
        
        test_acc /= len(test_loader)
        if test_acc > best_acc:
            best_acc = test_acc
        
        print(f"Epoch {epoch:3d}: Train Loss={train_loss:.4f} Train Acc={train_acc:.4f} Test Acc={test_acc:.4f}")
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Complete! Best Test Accuracy: {best_acc:.4f}")
    print(f"Total Time: {total_time:.2f}s ({total_time/args.epochs:.2f}s/epoch)")
    print(f"{'='*60}")
    
    return best_acc


def train_bp(args, train_loader, test_loader, input_dim, num_classes):
    """Train with standard backpropagation."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    embedding, model, output_head, n_heads, d_ff = create_tiny_model(
        input_dim, args.d_model, num_classes, device
    )
    
    solver = EquilibriumSolver(
        max_iters=args.max_iters,
        tol=args.tol,
        damping=args.damping
    )
    
    optimizer = torch.optim.Adam(
        list(embedding.parameters()) + list(model.parameters()) + list(output_head.parameters()),
        lr=args.lr
    )
    
    print(f"\n{'='*60}")
    print(f"Training MICRO TASK with Backprop (BP)")
    print(f"{'='*60}")
    print(f"Task: {args.task}")
    print(f"Model: d_model={args.d_model}, n_heads={n_heads}, d_ff={d_ff}")
    print(f"BP: lr={args.lr}, max_iters={args.max_iters}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    best_acc = 0
    start_time = time.time()
    
    for epoch in range(args.epochs):
        model.train()
        embedding.train()
        output_head.train()
        
        total_loss = 0
        total_acc = 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            x_emb = embedding(data).unsqueeze(0)
            h0 = torch.zeros_like(x_emb)
            h_fixed, _ = solver.solve(model, h0, x_emb)
            
            y_pred = output_head(h_fixed.mean(dim=0))
            loss = F.cross_entropy(y_pred, target)
            
            loss.backward()
            optimizer.step()
            
            acc = (y_pred.argmax(-1) == target).float().mean().item()
            total_loss += loss.item()
            total_acc += acc
        
        train_loss = total_loss / len(train_loader)
        train_acc = total_acc / len(train_loader)
        
        # Evaluate
        model.eval()
        test_acc = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                x_emb = embedding(data).unsqueeze(0)
                h0 = torch.zeros_like(x_emb)
                h_fixed, _ = solver.solve(model, h0, x_emb)
                y_pred = output_head(h_fixed.mean(dim=0))
                test_acc += (y_pred.argmax(-1) == target).float().mean().item()
        
        test_acc /= len(test_loader)
        if test_acc > best_acc:
            best_acc = test_acc
        
        print(f"Epoch {epoch:3d}: Train Loss={train_loss:.4f} Train Acc={train_acc:.4f} Test Acc={test_acc:.4f}")
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Complete! Best Test Accuracy: {best_acc:.4f}")
    print(f"Total Time: {total_time:.2f}s ({total_time/args.epochs:.2f}s/epoch)")
    print(f"{'='*60}")
    
    return best_acc


def main():
    parser = argparse.ArgumentParser(
        description="Train on micro tasks for rapid EqProp exploration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Task
    parser.add_argument("--task", type=str, default="xor",
                       choices=list(MICRO_TASK_INFO.keys()),
                       help="Micro task to train on")
    parser.add_argument("--n-samples", type=int, default=2000,
                       help="Number of training samples")
    
    # Model
    parser.add_argument("--d-model", type=int, default=16,
                       help="Model dimension (8, 16, 32 for micro tasks)")
    
    # Training
    parser.add_argument("--epochs", type=int, default=20,
                       help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="Learning rate")
    
    # EqProp specific
    parser.add_argument("--beta", type=float, default=0.22,
                       help="EqProp nudge strength (0.20-0.25 recommended)")
    parser.add_argument("--damping", type=float, default=0.8,
                       help="Equilibrium damping factor")
    parser.add_argument("--max-iters", type=int, default=20,
                       help="Max equilibrium iterations")
    parser.add_argument("--tol", type=float, default=1e-4,
                       help="Convergence tolerance")
    parser.add_argument("--update-mode", type=str, default="mse_proxy",
                       choices=["mse_proxy", "vector_field", "local_hebbian"],
                       help="EqProp update mode")
    
    # Algorithm choice
    parser.add_argument("--use-bp", action="store_true",
                       help="Use backpropagation instead of EqProp")
    
    # System
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Load data
    train_loader, input_dim, num_classes = get_micro_loader(
        args.task, train=True, batch_size=args.batch_size, n_samples=args.n_samples
    )
    test_loader, _, _ = get_micro_loader(
        args.task, train=False, batch_size=args.batch_size, n_samples=args.n_samples // 5
    )
    
    print(f"\nTask: {args.task} | Input dim: {input_dim} | Classes: {num_classes}")
    print(f"Train samples: {len(train_loader.dataset)} | Test samples: {len(test_loader.dataset)}")
    
    # Train
    if args.use_bp:
        train_bp(args, train_loader, test_loader, input_dim, num_classes)
    else:
        train_eqprop(args, train_loader, test_loader, input_dim, num_classes)


if __name__ == "__main__":
    main()
