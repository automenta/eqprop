"""
Task registry for TEP experiments.

Defines tasks for each phase and provides data loaders.

Phase 1 (Rapid Signal Detection):
- xor: 2-bit XOR (non-linear separability test)
- digits_8x8: 8x8 MNIST digits from scikit-learn (64 pixels)

Phase 2 (Validation):
- mnist_28x28: Full 28x28 MNIST
- cartpole_v1: Gym CartPole-v1

Phase 3 (Comprehensive):
- cifar10: CIFAR-10
- sequence_copy: Copy task
- acrobot_v1: Gym Acrobot-v1
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, Callable
from enum import Enum
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


@dataclass
class TaskSpec:
    """Specification for an experiment task."""
    name: str
    phase: int
    input_dim: int
    output_dim: int
    task_type: str  # "classification", "rl", "sequence"
    
    # Data loading
    get_train_loader: Callable
    get_test_loader: Callable
    
    # Performance metric
    metric_name: str = "accuracy"  # or "reward" for RL
    metric_higher_is_better: bool = True
    
    # Suggested training settings
    default_epochs: int = 10
    default_batch_size: int = 64


# =============================================================================
# PHASE 1 DATASETS
# =============================================================================

class XORDataset(Dataset):
    """XOR of 2 bits - simplest non-linear classification test."""
    
    def __init__(self, n_samples: int = 2000, seed: int = 42):
        self.n_samples = n_samples
        rng = np.random.RandomState(seed)
        
        # All 4 possible XOR inputs/outputs
        base_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
        base_labels = np.array([0, 1, 1, 0], dtype=np.int64)
        
        indices = rng.randint(0, 4, size=n_samples)
        self.data = base_inputs[indices]
        self.labels = base_labels[indices]
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]), self.labels[idx]


class Digits8x8Dataset(Dataset):
    """8x8 MNIST digits from scikit-learn (64 pixels).
    
    Uses sklearn.datasets.load_digits() for true 8x8 resolution,
    not downsampled 28x28 MNIST.
    """
    
    def __init__(self, train: bool = True, seed: int = 42):
        try:
            from sklearn.datasets import load_digits
            from sklearn.model_selection import train_test_split
        except ImportError:
            raise ImportError("scikit-learn required for 8x8 digits. Install with: pip install scikit-learn")
        
        digits = load_digits()
        X = digits.data.astype(np.float32)
        y = digits.target.astype(np.int64)
        
        # Normalize to [0, 1]
        X = X / 16.0  # sklearn digits are 0-16
        
        # Train/test split (80/20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )
        
        if train:
            self.data = X_train
            self.labels = y_train
        else:
            self.data = X_test
            self.labels = y_test
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]), self.labels[idx]


# =============================================================================
# PHASE 2 DATASETS
# =============================================================================

class MNIST28x28Dataset(Dataset):
    """Full 28x28 MNIST using torchvision."""
    
    def __init__(self, train: bool = True, root: str = "./data"):
        try:
            from torchvision import datasets, transforms
        except ImportError:
            raise ImportError("torchvision required for MNIST. Install with: pip install torchvision")
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        dataset = datasets.MNIST(root=root, train=train, download=True, transform=transform)
        self.data = dataset.data.float().view(-1, 784) / 255.0
        self.labels = dataset.targets
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# =============================================================================
# DATA LOADER FACTORIES
# =============================================================================

def get_xor_train_loader(batch_size: int = 64, n_samples: int = 2000, seed: int = 42) -> DataLoader:
    dataset = XORDataset(n_samples=n_samples, seed=seed)
    # Small dataset - multiprocessing overhead > benefit
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)


def get_xor_test_loader(batch_size: int = 64, n_samples: int = 400, seed: int = 42) -> DataLoader:
    dataset = XORDataset(n_samples=n_samples, seed=seed + 1000)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def get_digits8x8_train_loader(batch_size: int = 64, seed: int = 42) -> DataLoader:
    dataset = Digits8x8Dataset(train=True, seed=seed)
    # Small dataset (~1400 samples) - keep simple
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)


def get_digits8x8_test_loader(batch_size: int = 64, seed: int = 42) -> DataLoader:
    dataset = Digits8x8Dataset(train=False, seed=seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def get_mnist_train_loader(batch_size: int = 128, root: str = "./data") -> DataLoader:
    dataset = MNIST28x28Dataset(train=True, root=root)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, 
                     persistent_workers=True, pin_memory=True)


def get_mnist_test_loader(batch_size: int = 128, root: str = "./data") -> DataLoader:
    dataset = MNIST28x28Dataset(train=False, root=root)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2,
                     persistent_workers=True, pin_memory=True)


# =============================================================================
# RL TASK FACTORIES
# =============================================================================

def get_cartpole_env():
    """Get CartPole-v1 environment."""
    try:
        import gymnasium as gym
    except ImportError:
        try:
            import gym
        except ImportError:
            raise ImportError("gymnasium or gym required for RL tasks. Install with: pip install gymnasium")
    return gym.make("CartPole-v1")


def get_acrobot_env():
    """Get Acrobot-v1 environment."""
    try:
        import gymnasium as gym
    except ImportError:
        import gym
    return gym.make("Acrobot-v1")


# =============================================================================
# TASK REGISTRY
# =============================================================================

TASK_REGISTRY: Dict[str, TaskSpec] = {
    # Phase 1: Rapid Signal Detection
    "xor": TaskSpec(
        name="xor",
        phase=1,
        input_dim=2,
        output_dim=2,
        task_type="classification",
        get_train_loader=get_xor_train_loader,
        get_test_loader=get_xor_test_loader,
        metric_name="accuracy",
        default_epochs=20,
        default_batch_size=32,
    ),
    "digits_8x8": TaskSpec(
        name="digits_8x8",
        phase=1,
        input_dim=64,
        output_dim=10,
        task_type="classification",
        get_train_loader=get_digits8x8_train_loader,
        get_test_loader=get_digits8x8_test_loader,
        metric_name="accuracy",
        default_epochs=30,
        default_batch_size=64,
    ),
    
    # Phase 2: Validation
    "mnist_28x28": TaskSpec(
        name="mnist_28x28",
        phase=2,
        input_dim=784,
        output_dim=10,
        task_type="classification",
        get_train_loader=get_mnist_train_loader,
        get_test_loader=get_mnist_test_loader,
        metric_name="accuracy",
        default_epochs=20,
        default_batch_size=128,
    ),
    "cartpole_v1": TaskSpec(
        name="cartpole_v1",
        phase=2,
        input_dim=4,
        output_dim=2,
        task_type="rl",
        get_train_loader=lambda **kw: None,  # RL doesn't use data loaders
        get_test_loader=lambda **kw: None,
        metric_name="reward",
        default_epochs=100,  # episodes
        default_batch_size=1,
    ),
    
    # Phase 3: Comprehensive
    "acrobot_v1": TaskSpec(
        name="acrobot_v1",
        phase=3,
        input_dim=6,
        output_dim=3,
        task_type="rl",
        get_train_loader=lambda **kw: None,
        get_test_loader=lambda **kw: None,
        metric_name="reward",
        default_epochs=200,
        default_batch_size=1,
    ),
}


def get_task(name: str) -> TaskSpec:
    """Get a task specification by name."""
    if name not in TASK_REGISTRY:
        raise ValueError(f"Unknown task: {name}. Available: {list(TASK_REGISTRY.keys())}")
    return TASK_REGISTRY[name]


def get_phase_tasks(phase: int) -> Dict[str, TaskSpec]:
    """Get all tasks for a given phase."""
    return {name: spec for name, spec in TASK_REGISTRY.items() if spec.phase == phase}
