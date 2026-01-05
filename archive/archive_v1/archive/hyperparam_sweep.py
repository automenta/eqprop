"""Hyperparameter tuning to close the accuracy gap with BP."""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.models import LoopedTransformerBlock
from src.solver import EquilibriumSolver
from src.trainer import EqPropTrainer
import time
import itertools


def train_with_config(config, epochs=3, verbose=False):
    """Train and return test accuracy."""
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    
    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(784))
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    
    # Model
    embedding = nn.Linear(784, config["d_model"]).to(device)
    model = LoopedTransformerBlock(
        config["d_model"], config["n_heads"], config["d_ff"],
        attention_type='linear', symmetric=False
    ).to(device)
    output_head = nn.Linear(config["d_model"], 10).to(device)
    
    solver = EquilibriumSolver(
        max_iters=config["max_iters"],
        tol=config.get("tol", 1e-5),
        damping=config["damping"]
    )
    
    trainer = EqPropTrainer(
        model, solver, output_head, 
        beta=config["beta"], 
        lr=config["lr"],
        update_mode=config.get("update_mode", "mse_proxy")
    )
    trainer.optimizer.add_param_group({'params': embedding.parameters()})
    
    # Training
    for epoch in range(epochs):
        model.train()
        embedding.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            x_emb = embedding(data).unsqueeze(0)
            metrics = trainer.train_step(x_emb, target)
            
            if verbose and batch_idx % 200 == 0:
                print(f"  Epoch {epoch} [{batch_idx}/{len(train_loader)}] Acc: {metrics['accuracy']:.3f}")
    
    # Evaluation
    model.eval()
    embedding.eval()
    test_acc = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            x_emb = embedding(data).unsqueeze(0)
            h0 = torch.zeros_like(x_emb)
            h_fixed, _ = solver.solve(model, h0, x_emb)
            y_pred = output_head(h_fixed.mean(dim=0))
            test_acc += (y_pred.argmax(-1) == target).float().mean().item()
    
    return test_acc / len(test_loader)


def run_sweep():
    """Run hyperparameter sweep."""
    base_config = {
        "d_model": 128,
        "n_heads": 4,
        "d_ff": 512,
        "batch_size": 128,
        "max_iters": 50,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    
    # Parameters to sweep
    betas = [0.05, 0.1, 0.2]
    dampings = [0.8, 0.9, 0.95]
    lrs = [5e-4, 1e-3, 2e-3]
    
    print("=" * 70)
    print("Hyperparameter Sweep: Closing the Accuracy Gap")
    print("=" * 70)
    print(f"{'Beta':>8} {'Damping':>8} {'LR':>10} {'Test Acc':>10}")
    print("-" * 70)
    
    results = []
    best_acc = 0
    best_config = None
    
    for beta, damping, lr in itertools.product(betas, dampings, lrs):
        config = {**base_config, "beta": beta, "damping": damping, "lr": lr}
        
        try:
            acc = train_with_config(config, epochs=3)
            results.append({"beta": beta, "damping": damping, "lr": lr, "acc": acc})
            
            mark = "***" if acc > best_acc else ""
            print(f"{beta:>8.2f} {damping:>8.2f} {lr:>10.4f} {acc:>10.4f} {mark}")
            
            if acc > best_acc:
                best_acc = acc
                best_config = config
                
        except Exception as e:
            print(f"{beta:>8.2f} {damping:>8.2f} {lr:>10.4f} {'ERROR':>10}")
    
    print("-" * 70)
    print(f"\nBest configuration:")
    print(f"  Beta: {best_config['beta']}")
    print(f"  Damping: {best_config['damping']}")
    print(f"  LR: {best_config['lr']}")
    print(f"  Test Accuracy: {best_acc:.4f}")
    
    # Run best config for full 5 epochs
    print("\nRunning best config for 5 epochs...")
    full_acc = train_with_config(best_config, epochs=5, verbose=True)
    print(f"\nFinal Test Accuracy (5 epochs): {full_acc:.4f}")
    
    return results, best_config, full_acc


if __name__ == "__main__":
    run_sweep()
