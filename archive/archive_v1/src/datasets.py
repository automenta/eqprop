"""Unified dataset loaders for TorEqProp experiments.

Supports multiple classification datasets with consistent interface.
"""

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from typing import Tuple, Optional
import numpy as np


def get_dataset(
    name: str,
    train: bool = True,
    data_dir: str = "./data"
) -> Tuple[Dataset, int, int]:
    """Get a dataset by name.
    
    Args:
        name: Dataset name (mnist, fashion, cifar10, svhn, emnist)
        train: Whether to get train or test split
        data_dir: Directory to download/load data from
        
    Returns:
        (dataset, input_dim, num_classes)
    """
    name = name.lower()
    
    if name == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.view(784))
        ])
        dataset = datasets.MNIST(data_dir, train=train, download=True, transform=transform)
        return dataset, 784, 10
        
    elif name in ["fashion", "fashionmnist", "fashion_mnist"]:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
            transforms.Lambda(lambda x: x.view(784))
        ])
        dataset = datasets.FashionMNIST(data_dir, train=train, download=True, transform=transform)
        return dataset, 784, 10
        
    elif name == "cifar10":
        # Flatten CIFAR-10 for now (patch embeddings can come later)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            transforms.Lambda(lambda x: x.view(3072))  # 3*32*32
        ])
        dataset = datasets.CIFAR10(data_dir, train=train, download=True, transform=transform)
        return dataset, 3072, 10
        
    elif name == "svhn":
        # Street View House Numbers
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
            transforms.Lambda(lambda x: x.view(3072))  # 3*32*32
        ])
        split = "train" if train else "test"
        dataset = datasets.SVHN(data_dir, split=split, download=True, transform=transform)
        return dataset, 3072, 10
        
    elif name == "emnist":
        # Extended MNIST with letters (balanced split: 47 classes)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1751,), (0.3332,)),
            transforms.Lambda(lambda x: x.view(784))
        ])
        dataset = datasets.EMNIST(data_dir, split="balanced", train=train, download=True, transform=transform)
        return dataset, 784, 47
        
    else:
        raise ValueError(f"Unknown dataset: {name}. Supported: mnist, fashion, cifar10, svhn, emnist")


def get_data_loaders(
    dataset_name: str,
    batch_size: int = 128,
    num_workers: int = 4,
    data_dir: str = "./data"
) -> Tuple[DataLoader, DataLoader, int, int]:
    """Get train and test loaders for a dataset.
    
    Args:
        dataset_name: Name of dataset
        batch_size: Batch size for both loaders
        num_workers: Number of data loading workers
        data_dir: Data directory
        
    Returns:
        (train_loader, test_loader, input_dim, num_classes)
    """
    train_dataset, input_dim, num_classes = get_dataset(dataset_name, train=True, data_dir=data_dir)
    test_dataset, _, _ = get_dataset(dataset_name, train=False, data_dir=data_dir)
    
    # Use persistent workers for faster data loading when num_workers > 0
    persistent = num_workers > 0
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    return train_loader, test_loader, input_dim, num_classes


# Dataset info for quick reference
DATASET_INFO = {
    "mnist": {"input_dim": 784, "num_classes": 10, "description": "Handwritten digits"},
    "fashion": {"input_dim": 784, "num_classes": 10, "description": "Clothing items"},
    "cifar10": {"input_dim": 3072, "num_classes": 10, "description": "32x32 color images"},
    "svhn": {"input_dim": 3072, "num_classes": 10, "description": "Street view house numbers"},
    "emnist": {"input_dim": 784, "num_classes": 47, "description": "Extended MNIST with letters"},
}


def list_datasets() -> None:
    """Print available datasets and their info."""
    print("Available datasets:")
    print("-" * 60)
    for name, info in DATASET_INFO.items():
        print(f"  {name:12} | dim={info['input_dim']:5} | classes={info['num_classes']:2} | {info['description']}")
    print("-" * 60)


if __name__ == "__main__":
    # Quick test
    list_datasets()
    print("\nTesting dataset loading...")
    for name in ["mnist", "fashion", "cifar10"]:
        train_loader, test_loader, input_dim, num_classes = get_data_loaders(name, batch_size=32)
        x, y = next(iter(train_loader))
        print(f"  {name}: input shape {x.shape}, labels {y.shape}, classes {num_classes}")
