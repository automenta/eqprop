import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, TensorDataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np

def get_task_loader(task_name, batch_size=64, dataset_size=1000):
    """
    Returns (train_loader, test_loader, input_dim, output_dim, is_classification)
    """
    if task_name == "mnist":
        return get_mnist(batch_size, dataset_size)
    elif task_name == "digits":
        return get_digits(batch_size)
    elif task_name == "fashion-mnist":
        return get_fashion_mnist(batch_size, dataset_size)
    elif task_name == "cartpole":
        return get_cartpole_bc(batch_size, dataset_size)
    elif task_name == "acrobot":
        return get_acrobot_bc(batch_size, dataset_size)
    elif task_name == "tiny-lm":
        return get_tiny_lm(batch_size, dataset_size)
    elif task_name == "cifar10":
        return get_cifar10(batch_size, dataset_size)
    else:
        raise ValueError(f"Unknown task: {task_name}")

def get_cifar10(batch_size, dataset_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    train_data = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    test_data = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
    
    # Subsample if dataset_size is specified and less than full dataset
    if dataset_size is not None and dataset_size < 50000:
        indices = torch.randperm(len(train_data))[:dataset_size]
        train_data = Subset(train_data, indices)
        
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Return 3 channels (not flattened)
    return train_loader, test_loader, 3, 10



def get_mnist(batch_size, dataset_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])
    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    # Subsample
    if dataset_size is not None and dataset_size < 60000:
        indices = torch.randperm(len(train_data))[:dataset_size]
        train_data = Subset(train_data, indices)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, 784, 10

def get_fashion_mnist(batch_size, dataset_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])
    train_data = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    test_data = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
    
    # Subsample
    if dataset_size is not None and dataset_size < 60000:
        indices = torch.randperm(len(train_data))[:dataset_size]
        train_data = Subset(train_data, indices)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, 784, 10

def get_digits(batch_size):
    digits = load_digits()
    X = digits.data.astype(np.float32)
    y = digits.target.astype(np.int64)
    # Normalize
    X /= 16.0
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, 64, 10

def get_cartpole_bc(batch_size, dataset_size):
    # Heuristic: if angle < 0, move left (0), else right (1)
    # State: [pos, vel, angle, angle_vel]
    input_dim = 4
    output_dim = 2
    
    X = np.random.uniform(low=-0.2, high=0.2, size=(dataset_size, input_dim)).astype(np.float32)
    # Bias towards balancing
    y = np.where(X[:, 2] < 0, 0, 1).astype(np.int64)
    
    # Split
    split = int(0.8 * dataset_size)
    train_ds = TensorDataset(torch.from_numpy(X[:split]), torch.from_numpy(y[:split]))
    test_ds = TensorDataset(torch.from_numpy(X[split:]), torch.from_numpy(y[split:]))
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, input_dim, output_dim

def get_acrobot_bc(batch_size, dataset_size):
    # Heuristic for acrobot: swing-up based on joint angles
    # State: [cos(θ1), sin(θ1), cos(θ2), sin(θ2), θ1_dot, θ2_dot]
    # Actions: 0 = -1 torque, 1 = 0, 2 = +1 torque
    input_dim = 6
    output_dim = 3
    
    # Generate states with realistic ranges
    X = np.zeros((dataset_size, input_dim), dtype=np.float32)
    theta1 = np.random.uniform(-np.pi, np.pi, dataset_size)
    theta2 = np.random.uniform(-np.pi, np.pi, dataset_size)
    X[:, 0] = np.cos(theta1)
    X[:, 1] = np.sin(theta1)
    X[:, 2] = np.cos(theta2)
    X[:, 3] = np.sin(theta2)
    X[:, 4] = np.random.uniform(-4, 4, dataset_size)  # θ1_dot
    X[:, 5] = np.random.uniform(-9, 9, dataset_size)  # θ2_dot
    
    # Swing-up heuristic: torque in direction of tip velocity to pump energy
    # tip_height indicator: cos(θ1) + cos(θ1 + θ2) (higher = better)
    tip_indicator = X[:, 0] + X[:, 0] * X[:, 2] - X[:, 1] * X[:, 3]  # cos(θ1+θ2) expansion
    # Velocity-based pumping: swing with momentum
    momentum = X[:, 4] + X[:, 5]
    y = np.where(tip_indicator > 0.5, 1,  # Near top: no torque (coast)
                 np.where(momentum > 0, 2, 0)).astype(np.int64)  # Pump with direction
    
    split = int(0.8 * dataset_size)
    train_ds = TensorDataset(torch.from_numpy(X[:split]), torch.from_numpy(y[:split]))
    test_ds = TensorDataset(torch.from_numpy(X[split:]), torch.from_numpy(y[split:]))
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, input_dim, output_dim

def get_tiny_lm(batch_size, dataset_size):
    # Next character prediction on dummy text
    text = "The quick brown fox jumps over the lazy dog. " * 100
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    
    # Input: 1-hot char, Output: next char class
    # Actually, EqProp usually takes vector input. 
    # We'll map char -> 1-hot vector.
    
    data_indices = [char_to_ix[c] for c in text]
    
    X_list = []
    y_list = []
    for i in range(len(data_indices) - 1):
        # Input: One-hot embedding of current char
        x_vec = np.zeros(vocab_size, dtype=np.float32)
        x_vec[data_indices[i]] = 1.0
        X_list.append(x_vec)
        y_list.append(data_indices[i+1])
        
    X = np.array(X_list)[:dataset_size]
    y = np.array(y_list)[:dataset_size]
    
    split = int(0.8 * len(X))
    train_ds = TensorDataset(torch.from_numpy(X[:split]), torch.from_numpy(y[:split]))
    test_ds = TensorDataset(torch.from_numpy(X[split:]), torch.from_numpy(y[split:]))
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, vocab_size, vocab_size
