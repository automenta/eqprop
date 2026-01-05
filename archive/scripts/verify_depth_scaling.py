#!/usr/bin/env python3
"""
Verify O(1) Memory Scaling with Depth (Time Steps).
Compares EqProp (LocalHebbian) vs BPTT (Standard Backprop Through Time).
"""

import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import torch.cuda as cuda
import torch.optim as optim


from src.models import ModernEqProp
from src.training import EqPropTrainer
from src.training.updates import LocalHebbianUpdate

# Standard RNN for BPTT comparison
class SimpleRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.rnn = nn.RNNCell(input_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, steps):
        h = torch.zeros(x.size(0), self.rnn.hidden_size, device=x.device)
        for _ in range(steps):
            h = self.rnn(x, h) # Re-inject x at each step? Or just once?
            # EqProp usually has constant input injection.
            # Let's match EqProp: x injected every step.
        return self.head(h)

def measure_peak_memory(func):
    cuda.empty_cache()
    cuda.reset_peak_memory_stats()
    func()
    return cuda.max_memory_allocated() / 1024**2

def run_test():
    input_dim = 128
    hidden_dim = 512 # Large enough to see activation memory
    output_dim = 10
    batch_size = 64
    
    x = torch.randn(batch_size, input_dim).cuda()
    y = torch.randint(0, 10, (batch_size,)).cuda()
    
    steps_list = [10, 20, 50, 100, 200]
    
    results_eqprop = []
    results_bptt = []
    
    print(f"{'Steps':<10} | {'EqProp (MB)':<15} | {'BPTT (MB)':<15} | {'Ratio'}")
    print("-" * 60)
    
    for steps in steps_list:
        # --- EqProp ---
        model_ep = ModernEqProp(input_dim, hidden_dim, output_dim, use_spectral_norm=True).cuda()
        # Use LocalHebbianUpdate for O(1)
        # Note: We need a fresh object each time or reset hooks
        update_strat = LocalHebbianUpdate(beta=0.22) 
        optimizer_ep = optim.Adam(model_ep.parameters(), lr=0.001)
        trainer = EqPropTrainer(model_ep, optimizer_ep, max_steps=steps, update_strategy=update_strat)
        
        def train_ep():
            trainer.step(x, y)
            
        mem_ep = measure_peak_memory(train_ep)
        results_eqprop.append(mem_ep)
        
        # --- BPTT ---
        # Note: ModernEqProp with standard loss.backward() does BPTT if we don't detach?
        # But ModernEqProp.forward usually iterates in-place? 
        # Standard RNN with autograd will store history.
        model_rnn = SimpleRNN(input_dim, hidden_dim, output_dim).cuda()
        optimizer_rnn = optim.Adam(model_rnn.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        def train_bptt():
            optimizer_rnn.zero_grad()
            out = model_rnn(x, steps)
            loss = criterion(out, y)
            loss.backward()
            optimizer_rnn.step()
            
        mem_bptt = measure_peak_memory(train_bptt)
        results_bptt.append(mem_bptt)
        
        ratio = mem_bptt / mem_ep
        print(f"{steps:<10} | {mem_ep:<15.2f} | {mem_bptt:<15.2f} | {ratio:.2f}x")

    return steps_list, results_eqprop, results_bptt

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA required.")
        sys.exit(1)
    run_test()
