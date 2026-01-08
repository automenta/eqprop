"""
EqProp-Torch Dataset Utilities

HuggingFace datasets and tokenizers integration for easy LM and vision dataset loading.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, Dict, Any
import warnings


# =============================================================================
# Vision Datasets
# =============================================================================

def get_vision_dataset(
    name: str = "mnist",
    root: str = "./data",
    train: bool = True,
    download: bool = True,
    flatten: bool = False,
) -> Dataset:
    """
    Load a vision dataset with standard transforms.
    
    Args:
        name: Dataset name ('mnist', 'fashion_mnist', 'cifar10', 'kmnist', 'svhn')
        root: Data directory
        train: If True, load training set
        download: If True, download if not present
        flatten: If True, flatten images to 1D
        
    Returns:
        PyTorch Dataset
        
    Example:
        >>> train_data = get_vision_dataset('mnist', train=True)
        >>> test_data = get_vision_dataset('mnist', train=False)
    """
    from torchvision import datasets, transforms
    
    # Standard transforms
    transform_list = [transforms.ToTensor()]
    
    if name in ['mnist', 'fashion_mnist', 'kmnist']:
        # Normalize grayscale to [-1, 1] range
        transform_list.append(transforms.Normalize((0.5,), (0.5,)))
    elif name == 'cifar10':
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    
    if flatten:
        transform_list.append(transforms.Lambda(lambda x: x.view(-1)))
    
    transform = transforms.Compose(transform_list)
    
    # Select dataset
    dataset_map = {
        'mnist': datasets.MNIST,
        'fashion_mnist': datasets.FashionMNIST,
        'cifar10': datasets.CIFAR10,
        'kmnist': datasets.KMNIST,
    }
    
    if name not in dataset_map:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(dataset_map.keys())}")
    
    return dataset_map[name](root, train=train, download=download, transform=transform)


# =============================================================================
# Language Modeling Datasets
# =============================================================================

class CharDataset(Dataset):
    """Character-level language modeling dataset."""
    
    def __init__(self, text: str, seq_len: int = 128):
        self.seq_len = seq_len
        
        # Build vocabulary
        chars = sorted(set(text))
        self.char_to_idx = {c: i for i, c in enumerate(chars)}
        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}
        self.vocab_size = len(chars)
        
        # Encode text
        self.data = torch.tensor([self.char_to_idx[c] for c in text], dtype=torch.long)
    
    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_len - 1)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + 1:idx + self.seq_len + 1]
        return x, y
    
    def decode(self, indices: torch.Tensor) -> str:
        """Convert indices back to text."""
        return ''.join(self.idx_to_char[i.item()] for i in indices)


def get_lm_dataset(
    name: str = "tiny_shakespeare",
    seq_len: int = 128,
    batch_size: int = 64,
    split: str = "train",
) -> Tuple[DataLoader, Dict[str, Any]]:
    """
    Load a language modeling dataset with CharDataset wrapper.
    
    Args:
        name: Dataset name ('tiny_shakespeare', 'wikitext-2', 'ptb')
        seq_len: Sequence length for training
        batch_size: Batch size for DataLoader
        split: 'train', 'validation', or 'test'
        
    Returns:
        (DataLoader, vocab_info) where vocab_info contains char_to_idx, idx_to_char, vocab_size
        
    Example:
        >>> train_loader, vocab = get_lm_dataset('tiny_shakespeare', seq_len=128)
        >>> print(f"Vocab size: {vocab['vocab_size']}")
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "HuggingFace datasets required. Install with: pip install datasets"
        )
    
    # Load dataset
    if name == "tiny_shakespeare":
        # Shakespeare from HuggingFace
        try:
            dataset = load_dataset("tiny_shakespeare", trust_remote_code=True)
            text_key = "text"
            if split == "train":
                text = dataset["train"][text_key]
            elif split == "validation":
                text = dataset["validation"][text_key]
            else:
                text = dataset["test"][text_key]
            
            # tiny_shakespeare returns list, join it
            if isinstance(text, list):
                text = "\n".join(text)
        except Exception as e:
            # Fallback: load from URL
            warnings.warn(f"HuggingFace dataset failed, using fallback: {e}")
            import urllib.request
            url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            with urllib.request.urlopen(url) as response:
                text = response.read().decode('utf-8')
    
    elif name == "wikitext-2":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        text = "\n".join(dataset[split]["text"])
    
    elif name == "ptb":
        dataset = load_dataset("ptb_text_only")
        split_name = "train" if split == "train" else "validation" if split == "validation" else "test"
        text = " ".join(dataset[split_name]["sentence"])
    
    else:
        raise ValueError(f"Unknown LM dataset: {name}. Available: tiny_shakespeare, wikitext-2, ptb")
    
    # Create CharDataset
    char_dataset = CharDataset(text, seq_len=seq_len)
    
    # Create DataLoader
    loader = DataLoader(
        char_dataset, 
        batch_size=batch_size, 
        shuffle=(split == "train"),
        drop_last=True,
    )
    
    vocab_info = {
        'vocab_size': char_dataset.vocab_size,
        'char_to_idx': char_dataset.char_to_idx,
        'idx_to_char': char_dataset.idx_to_char,
        'decode': char_dataset.decode,
    }
    
    return loader, vocab_info


# =============================================================================
# Utility Functions
# =============================================================================

def create_data_loaders(
    dataset_name: str = "mnist",
    batch_size: int = 64,
    num_workers: int = 0,
    flatten: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test data loaders for a vision dataset.
    
    Args:
        dataset_name: Name of dataset
        batch_size: Batch size
        num_workers: Number of data loading workers
        flatten: Flatten images for MLP models
        
    Returns:
        (train_loader, test_loader)
    """
    train_data = get_vision_dataset(dataset_name, train=True, flatten=flatten)
    test_data = get_vision_dataset(dataset_name, train=False, flatten=flatten)
    
    train_loader = DataLoader(
        train_data, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_data, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
    )
    
    return train_loader, test_loader


__all__ = [
    'get_vision_dataset',
    'get_lm_dataset',
    'CharDataset',
    'create_data_loaders',
]
