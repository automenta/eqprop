"""
Intermediate-Scale Tasks for EqProp Evaluation

Creates tasks between digits (64d) and MNIST (784d) to understand scaling.
"""

from dataclasses import dataclass
from typing import Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import torchvision.transforms.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np


@dataclass
class TaskConfig:
    """Task configuration."""
    name: str
    input_dim: int
    output_dim: int
    hidden_dim: int = 128
    epochs: int = 10
    batch_size: int = 32
    lr: float = 1e-3
    beta: float = 0.22


class DownsampledMNIST:
    """MNIST downsampled to 14x14 (196 dims) - intermediate difficulty."""
    
    def __init__(self, batch_size=64, seed=42):
        self.batch_size = batch_size
        self.seed = seed
    
    def get_config(self) -> TaskConfig:
        return TaskConfig(
            name='mnist_14x14',
            input_dim=196,
            output_dim=10,
            hidden_dim=196,
            epochs=10,
            batch_size=self.batch_size,
            lr=1e-3,
            beta=0.20
        )
    
    def get_loaders(self):
        """Get MNIST downsampled to 14x14."""
        transform = transforms.Compose([
            transforms.Resize(14),  # Downsample to 14x14
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.view(-1))
        ])
        
        train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_data = datasets.MNIST('./data', train=False, transform=transform)
        
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


class BinaryMNIST:
    """Binary MNIST (0 vs 1) - same dims but simpler task."""
    
    def __init__(self, batch_size=64, seed=42):
        self.batch_size = batch_size
        self.seed = seed
    
    def get_config(self) -> TaskConfig:
        return TaskConfig(
            name='mnist_binary',
            input_dim=784,
            output_dim=2,
            hidden_dim=256,
            epochs=10,
            batch_size=self.batch_size,
            lr=1e-3,
            beta=0.22
        )
    
    def get_loaders(self):
        """Get MNIST filtered for digits 0 and 1 only."""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.view(-1))
        ])
        
        full_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
        full_test = datasets.MNIST('./data', train=False, transform=transform)
        
        # Filter for 0 and 1
        def filter_binary(dataset):
            indices = [i for i, (_, label) in enumerate(dataset) if label in [0, 1]]
            return torch.utils.data.Subset(dataset, indices)
        
        train_data = filter_binary(full_train)
        test_data = filter_binary(full_test)
        
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


# Add to existing registry
INTERMEDIATE_TASKS = {
    'mnist_14x14': DownsampledMNIST,
    'mnist_binary': BinaryMNIST,
}


def get_intermediate_task(task_name: str, **kwargs):
    """Get intermediate task by name."""
    if task_name not in INTERMEDIATE_TASKS:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(INTERMEDIATE_TASKS.keys())}")
    return INTERMEDIATE_TASKS[task_name](**kwargs)
