"""
Multi-Task Benchmark Suite

Unified framework for testing IDEA models across diverse tasks:
- Classification: digits, MNIST, Fashion-MNIST
- Sequences: copy, parity (future)
- RL: CartPole (future)

Features:
- Standardized train/val/test splits
- Consistent metric reporting
- Automatic baseline comparisons
- Per-task optimal hyperparameters
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class TaskConfig:
    """Configuration for a benchmark task."""
    name: str
    input_dim: int
    output_dim: int
    hidden_dim: int = 64
    epochs: int = 10
    batch_size: int = 32
    lr: float = 1e-3
    beta: float = 0.22


@dataclass
class TaskResult:
    """Results from a benchmark task."""
    task_name: str
    model_name: str
    train_acc: float
    val_acc: float
    test_acc: float
    train_time_s: float
    params: int
    config: TaskConfig


class BenchmarkTask(ABC):
    """Abstract base class for benchmark tasks."""
    
    @abstractmethod
    def get_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Return train, val, test dataloaders."""
        pass
    
    @abstractmethod
    def get_config(self) -> TaskConfig:
        """Return task configuration."""
        pass


class DigitsTask(BenchmarkTask):
    """Sklearn digits (8x8) classification."""
    
    def __init__(self, batch_size=32, seed=42):
        self.batch_size = batch_size
        self.seed = seed
    
    def get_config(self) -> TaskConfig:
        return TaskConfig(
            name='digits',
            input_dim=64,
            output_dim=10,
            hidden_dim=64,
            epochs=10,
            batch_size=self.batch_size,
            lr=3e-3,  # Optimal from sweep
            beta=0.22
        )
    
    def get_loaders(self):
        digits = load_digits()
        X, y = digits.data, digits.target
        
        # Normalize
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Split: 60% train, 20% val, 20% test
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.4, random_state=self.seed, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=self.seed, stratify=y_temp
        )
        
        # To tensors
        def make_loader(X, y):
            X_t = torch.tensor(X, dtype=torch.float32)
            y_t = torch.tensor(y, dtype=torch.long)
            return DataLoader(TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=True)
        
        return make_loader(X_train, y_train), make_loader(X_val, y_val), make_loader(X_test, y_test)


class MNISTTask(BenchmarkTask):
    """MNIST digit classification."""
    
    def __init__(self, batch_size=64, seed=42):
        self.batch_size = batch_size
        self.seed = seed
    
    def get_config(self) -> TaskConfig:
        return TaskConfig(
            name='mnist',
            input_dim=784,
            output_dim=10,
            hidden_dim=256,
            epochs=10,
            batch_size=self.batch_size,
            lr=3e-3,  # Optimized (was 1e-3)
            beta=0.25  # Slightly higher for larger inputs
        )
    
    def get_loaders(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.view(-1))  # Flatten
        ])
        
        train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_data = datasets.MNIST('./data', train=False, transform=transform)
        
        # Split train into train/val (90%/10%)
        train_size = int(0.9 * len(train_data))
        val_size = len(train_data) - train_size
        train_subset, val_subset = torch.utils.data.random_split(
            train_data, [train_size, val_size],
            generator=torch.Generator().manual_seed(self.seed)
        )
        
        train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader


class FashionMNISTTask(BenchmarkTask):
    """Fashion-MNIST classification."""
    
    def __init__(self, batch_size=64, seed=42):
        self.batch_size = batch_size
        self.seed = seed
    
    def get_config(self) -> TaskConfig:
        return TaskConfig(
            name='fashion_mnist',
            input_dim=784,
            output_dim=10,
            hidden_dim=256,
            epochs=15,
            batch_size=self.batch_size,
            lr=1e-3,
            beta=0.22
        )
    
    def get_loaders(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
            transforms.Lambda(lambda x: x.view(-1))
        ])
        
        train_data = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
        test_data = datasets.FashionMNIST('./data', train=False, transform=transform)
        
        # Split train into train/val
        train_size = int(0.9 * len(train_data))
        val_size = len(train_data) - train_size
        train_subset, val_subset = torch.utils.data.random_split(
            train_data, [train_size, val_size],
            generator=torch.Generator().manual_seed(self.seed)
        )
        
        train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader


# Task registry
TASK_REGISTRY = {
    'digits': DigitsTask,
    'mnist': MNISTTask,
    'fashion': FashionMNISTTask,
}


def get_task(task_name: str, **kwargs) -> BenchmarkTask:
    """Get task instance by name."""
    if task_name not in TASK_REGISTRY:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(TASK_REGISTRY.keys())}")
    return TASK_REGISTRY[task_name](**kwargs)
