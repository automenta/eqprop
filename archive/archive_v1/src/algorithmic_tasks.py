"""Algorithmic reasoning tasks for TorEqProp.

These tasks test the adaptive compute hypothesis: can equilibrium models
allocate more iterations to harder problem instances?

Tasks:
- Parity: XOR of N bits (requires N sequential operations)
- Reversal: Reverse a sequence
- Copy: Echo the input
- Addition: Add two N-digit numbers with carry propagation
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional
import numpy as np


class ParityDataset(Dataset):
    """Parity (XOR) task: output 1 if odd number of 1s, else 0.
    
    Difficulty scales with sequence length (requires N sequential ops).
    """
    
    def __init__(self, n_samples: int = 10000, seq_len: int = 8, seed: int = 42):
        self.n_samples = n_samples
        self.seq_len = seq_len
        
        rng = np.random.RandomState(seed)
        self.data = rng.randint(0, 2, size=(n_samples, seq_len)).astype(np.float32)
        self.labels = (self.data.sum(axis=1) % 2).astype(np.int64)
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.data[idx]), torch.tensor(self.labels[idx])


class ReversalDataset(Dataset):
    """Reversal task: reverse the input sequence.
    
    Tests memory and sequential processing.
    """
    
    def __init__(self, n_samples: int = 10000, seq_len: int = 8, vocab_size: int = 10, seed: int = 42):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
        rng = np.random.RandomState(seed)
        # Input: sequence of integers [0, vocab_size)
        self.data = rng.randint(0, vocab_size, size=(n_samples, seq_len)).astype(np.int64)
        # Output: reversed sequence
        self.labels = self.data[:, ::-1].copy()
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # One-hot encode input
        x = torch.zeros(self.seq_len, self.vocab_size)
        x[torch.arange(self.seq_len), self.data[idx]] = 1.0
        return x.view(-1), torch.from_numpy(self.labels[idx])


class CopyDataset(Dataset):
    """Copy task: echo the input sequence.
    
    Tests basic memory capacity.
    """
    
    def __init__(self, n_samples: int = 10000, seq_len: int = 8, vocab_size: int = 10, seed: int = 42):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
        rng = np.random.RandomState(seed)
        self.data = rng.randint(0, vocab_size, size=(n_samples, seq_len)).astype(np.int64)
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.zeros(self.seq_len, self.vocab_size)
        x[torch.arange(self.seq_len), self.data[idx]] = 1.0
        return x.view(-1), torch.from_numpy(self.data[idx])


class AdditionDataset(Dataset):
    """Addition task: add two N-digit numbers.
    
    Requires carry propagation (sequential operation).
    Harder instances have more carries.
    """
    
    def __init__(self, n_samples: int = 10000, n_digits: int = 4, seed: int = 42):
        self.n_samples = n_samples
        self.n_digits = n_digits
        
        rng = np.random.RandomState(seed)
        max_val = 10 ** n_digits - 1
        
        # Generate two numbers
        self.a = rng.randint(0, max_val + 1, size=n_samples)
        self.b = rng.randint(0, max_val + 1, size=n_samples)
        self.sums = self.a + self.b
        
        # Encode as digit sequences (for input)
        self.data = np.zeros((n_samples, 2 * n_digits), dtype=np.float32)
        for i in range(n_samples):
            a_digits = [int(d) for d in str(self.a[i]).zfill(n_digits)]
            b_digits = [int(d) for d in str(self.b[i]).zfill(n_digits)]
            self.data[i, :n_digits] = a_digits
            self.data[i, n_digits:] = b_digits
        
        # Output: sum modulo 10^n_digits (avoid overflow issues)
        self.labels = (self.sums % (10 ** n_digits)).astype(np.int64)
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Normalize input to [0, 1]
        x = torch.from_numpy(self.data[idx] / 9.0)
        return x, torch.tensor(self.labels[idx])


def get_algorithmic_loader(
    task: str,
    train: bool = True,
    batch_size: int = 128,
    seq_len: int = 8,
    n_samples: int = 10000
) -> Tuple[DataLoader, int, int]:
    """Get data loader for an algorithmic task.
    
    Args:
        task: Task name (parity, reversal, copy, addition)
        train: If True, use training seed; else test seed
        batch_size: Batch size
        seq_len: Sequence length for parity/reversal/copy
        n_samples: Number of samples
        
    Returns:
        (data_loader, input_dim, num_classes)
    """
    seed = 42 if train else 1337
    
    task = task.lower()
    if task == "parity":
        dataset = ParityDataset(n_samples, seq_len, seed)
        input_dim = seq_len
        num_classes = 2
        
    elif task == "reversal":
        vocab_size = 10
        dataset = ReversalDataset(n_samples, seq_len, vocab_size, seed)
        input_dim = seq_len * vocab_size
        num_classes = vocab_size  # Per-position classification
        
    elif task == "copy":
        vocab_size = 10
        dataset = CopyDataset(n_samples, seq_len, vocab_size, seed)
        input_dim = seq_len * vocab_size
        num_classes = vocab_size  # Per-position classification
        
    elif task == "addition":
        n_digits = seq_len // 2  # Use half of seq_len for each number
        dataset = AdditionDataset(n_samples, n_digits, seed)
        input_dim = 2 * n_digits
        num_classes = 10 ** n_digits  # Direct sum prediction
        
    else:
        raise ValueError(f"Unknown task: {task}. Supported: parity, reversal, copy, addition")
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train)
    return loader, input_dim, num_classes


# Task metadata
TASK_INFO = {
    "parity": {
        "description": "XOR of N bits",
        "difficulty": "Scales with N",
        "hypothesis": "More iterations for more 1s?"
    },
    "reversal": {
        "description": "Reverse input sequence",
        "difficulty": "Scales with length",
        "hypothesis": "Longer sequences need more iterations?"
    },
    "copy": {
        "description": "Echo input sequence",
        "difficulty": "Baseline (easy)",
        "hypothesis": "Should converge quickly"
    },
    "addition": {
        "description": "Add two N-digit numbers",
        "difficulty": "More carries = harder",
        "hypothesis": "More carries need more iterations?"
    },
}


def generate_parity(n_samples: int, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quick helper to generate parity data (for testing)."""
    dataset = ParityDataset(n_samples, seq_len)
    loader = DataLoader(dataset, batch_size=n_samples)
    return next(iter(loader))


if __name__ == "__main__":
    print("Algorithmic Tasks for TorEqProp")
    print("=" * 50)
    
    for task_name, info in TASK_INFO.items():
        loader, input_dim, num_classes = get_algorithmic_loader(task_name, n_samples=100)
        x, y = next(iter(loader))
        print(f"\n{task_name}:")
        print(f"  {info['description']}")
        print(f"  Input shape: {x.shape}, Labels shape: {y.shape}")
        print(f"  Input dim: {input_dim}, Classes: {num_classes}")
        print(f"  Hypothesis: {info['hypothesis']}")
