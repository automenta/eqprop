
import torch
import torch.nn as nn
import numpy as np
import time
from typing import Optional, Dict, Any

from .types import TrainingState

# Import models
# Assuming models are available in the path as set by entry point
from models import (
    LoopedMLP, DeepHebbianChain, DirectFeedbackAlignmentEqProp,
    ContrastiveHebbianLearning
)

class SimpleLM(nn.Module):
    """Minimal transformer for language modeling (Backprop baseline)."""
    def __init__(self, vocab_size, hidden_dim=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos = nn.Parameter(torch.randn(1, 64, hidden_dim) * 0.02)
        self.tf = nn.TransformerEncoderLayer(hidden_dim, 4, hidden_dim*2, batch_first=True, dropout=0.1)
        self.head = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        h = self.embed(x) + self.pos[:, :x.size(1)]
        mask = torch.triu(torch.ones(x.size(1), x.size(1)), diagonal=1).bool().to(x.device)
        h = self.tf(h, src_mask=mask)
        return self.head(h)[:, -1, :]  # Last token only

class AlgorithmWrapper:
    """Unified wrapper for all algorithms."""
    
    def __init__(self, name: str, vocab_size: int, hidden_dim: int = 128, num_layers: int = 20, device: str = 'cpu'):
        self.name = name
        self.device = device
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Hyperparameters
        self.beta = 0.22 # Default nudge strength
        self.steps = 30 # Default equilibrium steps
        
        # Create model
        self.model = self._create_model()
        
        # Determine strict param count
        self.param_count = sum(p.numel() for p in self.model.parameters())
        if self.has_embed:
             self.param_count += sum(p.numel() for p in self.embed.parameters())
        
        # Optimizer
        params = list(self.model.parameters())
        if self.has_embed:
            params += list(self.embed.parameters())
        self.opt = torch.optim.Adam(params, lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        
    def _create_model(self):
        """Factory method for model creation."""
        self.has_embed = True # Default for bio-models
        
        if self.name == 'Backprop':
            self.has_embed = False
            return SimpleLM(self.vocab_size, self.hidden_dim).to(self.device)
            
        elif self.name == 'EqProp':
            self.embed = nn.Embedding(self.vocab_size, self.hidden_dim).to(self.device)
            return LoopedMLP(self.hidden_dim, self.hidden_dim, self.vocab_size, use_spectral_norm=True).to(self.device)
            
        elif self.name == 'DFA':
            self.embed = nn.Embedding(self.vocab_size, self.hidden_dim).to(self.device)
            return DirectFeedbackAlignmentEqProp(self.hidden_dim, self.hidden_dim, self.vocab_size, min(self.num_layers, 20), use_spectral_norm=True).to(self.device)
            
        elif self.name == 'CHL':
            self.embed = nn.Embedding(self.vocab_size, self.hidden_dim).to(self.device)
            return ContrastiveHebbianLearning(self.hidden_dim, self.hidden_dim, self.vocab_size, min(self.num_layers, 20), use_spectral_norm=True).to(self.device)
            
        elif self.name == 'Deep Hebbian':
            self.embed = nn.Embedding(self.vocab_size, self.hidden_dim).to(self.device)
            return DeepHebbianChain(self.hidden_dim, self.hidden_dim, self.vocab_size, self.num_layers, use_spectral_norm=True).to(self.device)
            
        else:
            raise ValueError(f"Unknown algorithm: {self.name}")

    def update_hyperparams(self, lr: float = None, beta: float = None, steps: int = None):
        if lr is not None:
            for g in self.opt.param_groups:
                g['lr'] = lr
        if beta is not None:
            self.beta = beta
        if steps is not None:
             self.steps = steps

    def train_step(self, x, y, step_num) -> TrainingState:
        """Single training iteration."""
        t0 = time.time()
        
        self.model.train()
        self.opt.zero_grad()
        
        try:
            if self.has_embed:
                h = self.embed(x).mean(dim=1)  # Pool
                if self.name == 'EqProp':
                     # Pass specific hyperparameters for EqProp
                     # Note: assuming call signature based on previous code
                     logits = self.model(h, steps=self.steps) 
                else:
                     logits = self.model(h)
            else:
                logits = self.model(x)
            
            loss = self.criterion(logits, y)
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
            
            # Metrics
            acc = (logits.argmax(1) == y).float().mean().item()
            iter_time = time.time() - t0
            
            # VRAM estimate
            if torch.cuda.is_available() and self.device == 'cuda':
                vram = torch.cuda.memory_allocated(self.device) / 1e9
            else:
                # CPU estimate (rough)
                vram = (self.param_count * 4) / 1e9 
            
            # Signal norms (for visualizing deep signal propagation)
            norms = []
            if hasattr(self.model, 'layers'):
                for layer in self.model.layers:
                    if hasattr(layer, 'weight'):
                        norms.append(layer.weight.norm().item())
            if not norms:
                 # Fallback for models without standard layer structure
                norms = [1.0] * 10
            
            return TrainingState(
                loss=loss.item(),
                accuracy=acc,
                perplexity=np.exp(min(loss.item(), 10)),
                iter_time=iter_time,
                vram_gb=vram,
                signal_norms=norms[:30],
                step=step_num
            )
        
        except Exception as e:
            print(f"Error in {self.name} train_step: {e}")
            # print stack trace for debugging
            import traceback
            traceback.print_exc()
            return TrainingState(loss=10.0, iter_time=0.01, step=step_num)
    
    def generate(self, seed_idx, i2c, length=50):
        """Generate text sample."""
        self.model.eval()
        curr = seed_idx.clone().to(self.device)
        result = ""
        
        with torch.no_grad():
            try:
                for _ in range(length):
                    if self.has_embed:
                        h = self.embed(curr[-64:].unsqueeze(0)).mean(dim=1)
                        logits = self.model(h) # Use default steps for inference
                    else:
                        ctx = curr[-64:].unsqueeze(0)
                        logits = self.model(ctx)
                    
                    next_tok = logits.argmax(-1).item()
                    result += i2c.get(next_tok, '?')
                    curr = torch.cat([curr, torch.tensor([next_tok], device=self.device)])
            except:
                pass
        
        return result if result else "..."
