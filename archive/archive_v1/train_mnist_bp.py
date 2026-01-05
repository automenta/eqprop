"""Baseline training with standard backpropagation for fair comparison.

Uses the same LoopedTransformerBlock architecture but trains with standard BPTT
through the equilibrium solver instead of contrastive Equilibrium Propagation.
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.models import LoopedTransformerBlock
from src.solver import EquilibriumSolver
import torch.nn.functional as F
import time

# Global PyTorch optimizations (same as EqProp for fairness)
torch.backends.cudnn.benchmark = True  # Optimize conv/attention for fixed input sizes
torch.set_float32_matmul_precision('high')  # Use TensorCores if available


def train_bp(config):
    """Train model with standard backpropagation."""
    
    print(f"Training Baseline (BP) on {config['device']}")
    print(f"Model: d_model={config['d_model']}, epochs={config['epochs']}, lr={config['lr']}")

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(784))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    # Use persistent workers and pin_memory for optimal data loading
    persistent = config["num_workers"] > 0
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["batch_size"], 
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
        persistent_workers=persistent
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config["batch_size"], 
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
        persistent_workers=persistent
    )

    # Model (same architecture as EqProp)
    embedding = nn.Linear(784, config["d_model"]).to(config["device"])
    model = LoopedTransformerBlock(
        config["d_model"], 
        config["n_heads"], 
        config["d_ff"]
    ).to(config["device"])
    output_head = nn.Linear(config["d_model"], 10).to(config["device"])

    solver = EquilibriumSolver(
        max_iters=config["max_iters"],
        tol=1e-5,
        damping=config["damping"]
    )

    optimizer = optim.Adam(
        list(embedding.parameters()) + list(model.parameters()) + list(output_head.parameters()),
        lr=config["lr"]
    )

    best_acc = 0
    for epoch in range(config["epochs"]):
        model.train()
        embedding.train()
        output_head.train()

        total_loss = 0
        total_acc = 0
        start_time = time.time()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(config["device"]), target.to(config["device"])

            optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()

            x_emb = embedding(data).unsqueeze(0)
            h0 = torch.zeros_like(x_emb)

            # Forward (BPTT through solver - this is the key difference from EqProp)
            h_fixed, iters = solver.solve(model, h0, x_emb)

            y_pred = output_head(h_fixed.mean(dim=0))
            loss = F.cross_entropy(y_pred, target)

            loss.backward()
            optimizer.step()

            acc = (y_pred.argmax(-1) == target).float().mean().item()
            total_loss += loss.item()
            total_acc += acc

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f} Acc: {acc:.4f} Iters: {iters}")

        avg_loss = total_loss / len(train_loader)
        avg_acc = total_acc / len(train_loader)
        duration = time.time() - start_time

        print(f"Epoch {epoch} Completed in {duration:.2f}s. Avg Loss: {avg_loss:.4f} Avg Acc: {avg_acc:.4f}")

        # Validation
        model.eval()
        embedding.eval()
        output_head.eval()
        test_acc = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(config["device"]), target.to(config["device"])
                x_emb = embedding(data).unsqueeze(0)
                h0 = torch.zeros_like(x_emb)
                h_fixed, _ = solver.solve(model, h0, x_emb)
                y_pred = output_head(h_fixed.mean(dim=0))
                test_acc += (y_pred.argmax(-1) == target).float().mean().item()

        test_acc /= len(test_loader)
        if test_acc > best_acc:
            best_acc = test_acc
        print(f"Test Accuracy: {test_acc:.4f}")

    print(f"\nBest Test Accuracy: {best_acc:.4f}")
    return best_acc


def main():
    parser = argparse.ArgumentParser(
        description="Train baseline BP model for comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model
    parser.add_argument("--d-model", type=int, default=128, help="Model dimension")
    parser.add_argument("--n-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--d-ff", type=int, default=512, help="FFN hidden dimension")
    
    # Training
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    
    # Solver
    parser.add_argument("--max-iters", type=int, default=50, help="Max equilibrium iterations")
    parser.add_argument("--damping", type=float, default=0.9, help="Damping factor")
    
    # System
    parser.add_argument("--seed", type=int, default=-1, help="Random seed (-1 = no seed)")
    parser.add_argument("--num-workers", type=int, default=4, help="Data loader workers")
    parser.add_argument("--dataset", type=str, default="mnist", help="Dataset (for compatibility)")
    
    args = parser.parse_args()
    
    # Set seed if specified
    if args.seed >= 0:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    
    config = {
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "d_ff": args.d_ff,
        "batch_size": args.batch_size,
        "max_iters": args.max_iters,
        "damping": args.damping,
        "lr": args.lr,
        "epochs": args.epochs,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "num_workers": args.num_workers,
    }
    
    train_bp(config)


if __name__ == "__main__":
    main()
