
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional
from datetime import datetime

def progress_bar(current: int, total: int, width: int = 20) -> str:
    filled = int(width * current / total)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {current}/{total}"


def create_synthetic_dataset(n_samples: int, input_dim: int, n_classes: int, seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    centers = torch.randn(n_classes, input_dim) * 2
    samples_per_class = n_samples // n_classes
    X, y = [], []
    
    for c in range(n_classes):
        class_samples = centers[c] + torch.randn(samples_per_class, input_dim) * 0.5
        X.append(class_samples)
        y.append(torch.full((samples_per_class,), c, dtype=torch.long))
    
    X, y = torch.cat(X), torch.cat(y)
    perm = torch.randperm(len(y))
    return X[perm], y[perm]


def train_model(model: nn.Module, X: torch.Tensor, y: torch.Tensor, 
                epochs: int = 50, lr: float = 0.01, name: str = "Model", verifier=None, track_id=0, seed=0) -> List[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(X)
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if verifier:
            verifier.record_metric(track_id, seed, epoch, f"{name}_loss", loss.item())
            # Optionally record gradient norm
            grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
            verifier.record_metric(track_id, seed, epoch, f"{name}_grad_norm", grad_norm)
        
        acc = (out.argmax(dim=1) == y).float().mean().item() * 100
        print(f"\r  {name}: {progress_bar(epoch+1, epochs)} loss={loss.item():.3f} acc={acc:.1f}%", end="", flush=True)
    
    print()
    return losses


def evaluate_accuracy(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        out = model(X)
        acc = (out.argmax(dim=1) == y).float().mean().item()
    model.train()
    return acc
