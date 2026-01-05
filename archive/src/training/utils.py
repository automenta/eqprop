import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def get_mnist_loaders(batch_size=64, train_size=None, test_size=None):
    """
    Get MNIST train and test loaders.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])

    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)

    if train_size:
        indices = torch.randperm(len(train_data))[:train_size]
        train_data = Subset(train_data, indices)
    
    if test_size:
        indices = torch.randperm(len(test_data))[:test_size]
        test_data = Subset(test_data, indices)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
