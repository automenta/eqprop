"""Micro tasks for rapid experimentation with tiny models.

These tasks are designed to be solved in seconds with d_model=8-32,
allowing rapid exploration of EqProp's hyperparameter space.

Tasks:
- xor: XOR of 2 bits (simplest possible task)
- and_gate: AND of 2 bits
- or_gate: OR of 2 bits
- xor3: XOR of 3 bits (slightly harder)
- majority: Majority vote of N bits
- identity: Echo single input (baseline)
- tiny_lm: Next-token prediction on tiny vocabulary
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List
import numpy as np


class XORDataset(Dataset):
    """XOR of 2 bits - simplest binary classification.
    
    Should be trivially learnable. If this fails, something is wrong.
    """
    
    def __init__(self, n_samples: int = 1000, seed: int = 42):
        self.n_samples = n_samples
        rng = np.random.RandomState(seed)
        
        # All 4 possible inputs, repeated to get n_samples
        base_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
        base_labels = np.array([0, 1, 1, 0], dtype=np.int64)
        
        indices = rng.randint(0, 4, size=n_samples)
        self.data = base_inputs[indices]
        self.labels = base_labels[indices]
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]), torch.tensor(self.labels[idx])


class ANDDataset(Dataset):
    """AND of 2 bits - linearly separable."""
    
    def __init__(self, n_samples: int = 1000, seed: int = 42):
        self.n_samples = n_samples
        rng = np.random.RandomState(seed)
        
        base_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
        base_labels = np.array([0, 0, 0, 1], dtype=np.int64)
        
        indices = rng.randint(0, 4, size=n_samples)
        self.data = base_inputs[indices]
        self.labels = base_labels[indices]
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]), torch.tensor(self.labels[idx])


class ORDataset(Dataset):
    """OR of 2 bits - linearly separable."""
    
    def __init__(self, n_samples: int = 1000, seed: int = 42):
        self.n_samples = n_samples
        rng = np.random.RandomState(seed)
        
        base_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
        base_labels = np.array([0, 1, 1, 1], dtype=np.int64)
        
        indices = rng.randint(0, 4, size=n_samples)
        self.data = base_inputs[indices]
        self.labels = base_labels[indices]
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]), torch.tensor(self.labels[idx])


class XOR3Dataset(Dataset):
    """XOR of 3 bits - requires 2 nested XORs."""
    
    def __init__(self, n_samples: int = 1000, seed: int = 42):
        self.n_samples = n_samples
        rng = np.random.RandomState(seed)
        
        # All 8 possible inputs
        base_inputs = np.array([
            [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
            [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]
        ], dtype=np.float32)
        base_labels = (base_inputs.sum(axis=1) % 2).astype(np.int64)
        
        indices = rng.randint(0, 8, size=n_samples)
        self.data = base_inputs[indices]
        self.labels = base_labels[indices]
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]), torch.tensor(self.labels[idx])


class MajorityDataset(Dataset):
    """Majority vote of N bits - output 1 if more than half are 1."""
    
    def __init__(self, n_samples: int = 1000, n_bits: int = 5, seed: int = 42):
        self.n_samples = n_samples
        self.n_bits = n_bits
        rng = np.random.RandomState(seed)
        
        self.data = rng.randint(0, 2, size=(n_samples, n_bits)).astype(np.float32)
        self.labels = (self.data.sum(axis=1) > n_bits / 2).astype(np.int64)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]), torch.tensor(self.labels[idx])


class IdentityDataset(Dataset):
    """Identity task - echo single input (baseline sanity check)."""
    
    def __init__(self, n_samples: int = 1000, n_classes: int = 4, seed: int = 42):
        self.n_samples = n_samples
        self.n_classes = n_classes
        rng = np.random.RandomState(seed)
        
        self.labels = rng.randint(0, n_classes, size=n_samples).astype(np.int64)
        # One-hot encode
        self.data = np.zeros((n_samples, n_classes), dtype=np.float32)
        self.data[np.arange(n_samples), self.labels] = 1.0
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]), torch.tensor(self.labels[idx])


class TinyLMDataset(Dataset):
    """Tiny language modeling: next-token prediction on simple patterns.
    
    Vocabulary: digits 0-9 (10 tokens)
    Patterns:
    - Repeat last: "1 2 3" -> 3
    - Count up: "1 2 3" -> 4
    - Alternating: "1 2 1" -> 2
    
    Context length: 3-5 tokens
    """
    
    def __init__(self, n_samples: int = 2000, context_len: int = 4, 
                 vocab_size: int = 10, seed: int = 42):
        self.n_samples = n_samples
        self.context_len = context_len
        self.vocab_size = vocab_size
        rng = np.random.RandomState(seed)
        
        self.data = []
        self.labels = []
        
        for _ in range(n_samples):
            pattern_type = rng.randint(0, 3)
            
            if pattern_type == 0:
                # Repeat: output last token
                context = rng.randint(0, vocab_size, size=context_len)
                target = context[-1]
            
            elif pattern_type == 1:
                # Count up: output next in sequence
                start = rng.randint(0, vocab_size - context_len - 1)
                context = np.arange(start, start + context_len)
                target = (start + context_len) % vocab_size
            
            else:
                # Alternating: A B A B -> next in pattern
                a, b = rng.randint(0, vocab_size, size=2)
                context = np.array([a, b, a, b])[:context_len]
                target = a if context_len % 2 == 0 else b
            
            self.data.append(context.astype(np.int64))
            self.labels.append(target)
        
        self.data = np.array(self.data)
        self.labels = np.array(self.labels, dtype=np.int64)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        # One-hot encode context
        x = torch.zeros(self.context_len, self.vocab_size)
        for i, token in enumerate(self.data[idx]):
            x[i, token] = 1.0
        return x.view(-1), torch.tensor(self.labels[idx])


class Digits8x8Dataset(Dataset):
    """8x8 MNIST digits from scikit-learn (64 pixels).
    
    Uses sklearn.datasets.load_digits() for true 8x8 resolution,
    not downsampled 28x28 MNIST. This is the Phase 1 task for
    rapid signal detection.
    """
    
    def __init__(self, train: bool = True, seed: int = 42):
        try:
            from sklearn.datasets import load_digits
            from sklearn.model_selection import train_test_split
        except ImportError:
            raise ImportError(
                "scikit-learn required for 8x8 digits. "
                "Install with: pip install scikit-learn"
            )
        
        digits = load_digits()
        X = digits.data.astype(np.float32)
        y = digits.target.astype(np.int64)
        
        # Normalize to [0, 1] (sklearn digits are 0-16)
        X = X / 16.0
        
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
        return torch.from_numpy(self.data[idx]), torch.tensor(self.labels[idx])


def get_micro_loader(
    task: str,
    train: bool = True,
    batch_size: int = 64,
    n_samples: int = 1000,
    **kwargs
) -> Tuple[DataLoader, int, int]:
    """Get data loader for a micro task.
    
    Args:
        task: Task name
        train: If True, use training seed
        batch_size: Batch size
        n_samples: Number of samples
        
    Returns:
        (data_loader, input_dim, num_classes)
    """
    seed = 42 if train else 1337
    
    task = task.lower()
    
    if task == "xor":
        dataset = XORDataset(n_samples, seed)
        input_dim = 2
        num_classes = 2
    
    elif task == "and" or task == "and_gate":
        dataset = ANDDataset(n_samples, seed)
        input_dim = 2
        num_classes = 2
    
    elif task == "or" or task == "or_gate":
        dataset = ORDataset(n_samples, seed)
        input_dim = 2
        num_classes = 2
    
    elif task == "xor3":
        dataset = XOR3Dataset(n_samples, seed)
        input_dim = 3
        num_classes = 2
    
    elif task == "majority":
        n_bits = kwargs.get("n_bits", 5)
        dataset = MajorityDataset(n_samples, n_bits, seed)
        input_dim = n_bits
        num_classes = 2
    
    elif task == "identity":
        n_classes = kwargs.get("n_classes", 4)
        dataset = IdentityDataset(n_samples, n_classes, seed)
        input_dim = n_classes
        num_classes = n_classes
    
    elif task == "tiny_lm":
        context_len = kwargs.get("context_len", 4)
        vocab_size = kwargs.get("vocab_size", 10)
        dataset = TinyLMDataset(n_samples, context_len, vocab_size, seed)
        input_dim = context_len * vocab_size
        num_classes = vocab_size
    
    elif task == "digits_8x8" or task == "digits8x8":
        # 8x8 MNIST from sklearn (Phase 1 task)
        train_mode = train if isinstance(train, bool) else True
        dataset = Digits8x8Dataset(train=train_mode, seed=seed)
        input_dim = 64
        num_classes = 10
    
    else:
        raise ValueError(f"Unknown micro task: {task}")
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train)
    return loader, input_dim, num_classes


# Task metadata for quick reference
MICRO_TASK_INFO = {
    "xor": {"input_dim": 2, "classes": 2, "complexity": "trivial"},
    "and": {"input_dim": 2, "classes": 2, "complexity": "trivial (linear)"},
    "or": {"input_dim": 2, "classes": 2, "complexity": "trivial (linear)"},
    "xor3": {"input_dim": 3, "classes": 2, "complexity": "easy"},
    "majority": {"input_dim": 5, "classes": 2, "complexity": "easy"},
    "identity": {"input_dim": 4, "classes": 4, "complexity": "trivial"},
    "tiny_lm": {"input_dim": 40, "classes": 10, "complexity": "moderate"},
    "digits_8x8": {"input_dim": 64, "classes": 10, "complexity": "phase1"},
}


if __name__ == "__main__":
    print("Micro Tasks for Rapid EqProp Exploration")
    print("=" * 50)
    
    for task_name, info in MICRO_TASK_INFO.items():
        loader, input_dim, num_classes = get_micro_loader(task_name, n_samples=100)
        x, y = next(iter(loader))
        print(f"\n{task_name}:")
        print(f"  Input shape: {x.shape}, Labels shape: {y.shape}")
        print(f"  Input dim: {input_dim}, Classes: {num_classes}")
        print(f"  Complexity: {info['complexity']}")
