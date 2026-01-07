"""
Experiment Runner

Executes hyperparameter optimization trials and collects metrics.
"""

import torch
import torch.nn as nn
import time
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path
import sys

# Add parent to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from gui.algorithms import AlgorithmWrapper, get_model_spec
from gui.utils import load_shakespeare
from .storage import HyperoptStorage
from .metrics import TrialMetrics


class TrialRunner:
    """Runs individual hyperparameter optimization trials."""
    
    def __init__(
        self,
        storage: HyperoptStorage = None,
        device: str = 'auto',
        task: str = 'shakespeare',
        quick_mode: bool = True
    ):
        self.storage = storage or HyperoptStorage()
        self.device = 'cuda' if (device == 'auto' and torch.cuda.is_available()) else device
        self.task = task
        self.quick_mode = quick_mode
        
        # Training config
        if quick_mode:
            self.epochs = 5
            self.batches_per_epoch = 100
            self.eval_batches = 20
        else:
            self.epochs = 20
            self.batches_per_epoch = 200
            self.eval_batches = 50
        
        self.batch_size = 32
        self.seq_len = 64
        
        # Load data based on task
        print(f"Loading {task} dataset...")
        if task == 'shakespeare':
            self.data, self.c2i, self.i2c = load_shakespeare()
            self.vocab_size = len(self.c2i)
        else:
            # Placeholder for other tasks
            raise NotImplementedError(f"Task '{task}' not yet implemented. Only 'shakespeare' is supported.")
        
        # Split train/val
        n = int(0.9 * len(self.data))
        self.data_train = self.data[:n]
        self.data_val = self.data[n:]
        
        print(f"Dataset ready: {len(self.data_train)} train, {len(self.data_val)} val tokens")
    
    def get_batch(self, data, device):
        """Get a random batch."""
        idx = torch.randint(0, len(data) - self.seq_len - 1, (self.batch_size,))
        x =torch.stack([data[i:i+self.seq_len] for i in idx]).to(device)
        y = torch.stack([data[i+self.seq_len] for i in idx]).to(device)
        return x, y
    
    def run_trial(self, trial_id: int) -> bool:
        """Run a single trial and record results.
        
        Returns:
            True if successful, False if failed.
        """
        # Get trial
        trial = self.storage.get_trial(trial_id)
        if not trial:
            print(f"Trial {trial_id} not found")
            return False
        
        print(f"\n{'='*60}")
        print(f"Trial {trial_id}: {trial.model_name}")
        print(f"Config: {trial.config}")
        print(f"{'='*60}\n")
        
        # Update status
        self.storage.update_trial(trial_id, status='running')
        
        try:
            # Create model
            spec = get_model_spec(trial.model_name)
            
            # Apply config to spec (override defaults)
            config = trial.config
            hidden_dim = config.get('hidden_dim', 128)
            num_layers = config.get('num_layers', 20)
            
            algo = AlgorithmWrapper(
                spec,
                self.vocab_size,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                device=self.device
            )
            
            # Apply hyperparameters
            lr = config.get('lr', spec.default_lr)
            beta = config.get('beta', spec.default_beta) if spec.has_beta else None
            steps = config.get('steps', spec.default_steps) if spec.has_steps else None
            algo.update_hyperparams(lr=lr, beta=beta, steps=steps)
            
            # Training loop
            epoch_times = []
            
            for epoch in range(self.epochs):
                epoch_start = time.time()
                
                # Training
                algo.model.train()
                train_losses = []
                
                for _ in range(self.batches_per_epoch):
                    x, y = self.get_batch(self.data_train, self.device)
                    state = algo.train_step(x, y, epoch * self.batches_per_epoch + _)
                    train_losses.append(state.loss)
                
                # Validation
                algo.model.eval()
                val_losses = []
                val_accs = []
                
                with torch.no_grad():
                    for _ in range(self.eval_batches):
                        x, y = self.get_batch(self.data_val, self.device)
                        
                        if algo.has_embed:
                            h = algo.embed(x).mean(dim=1)
                            logits =algo.model(h, steps=algo.steps) if hasattr(algo.model, 'W_rec') else algo.model(h)
                        else:
                            logits = algo.model(x, steps=algo.steps) if hasattr(algo.model, 'eq_steps') else algo.model(x)
                            if logits.dim() == 3:
                                logits = logits[:, -1, :]
                        
                        loss = algo.criterion(logits, y)
                        acc = (logits.argmax(1) == y).float().mean()
                        
                        val_losses.append(loss.item())
                        val_accs.append(acc.item())
                
                epoch_time = time.time() - epoch_start
                epoch_times.append(epoch_time)
                
                avg_val_loss = np.mean(val_losses)
                avg_val_acc = np.mean(val_accs)
                avg_val_ppl = np.exp(min(avg_val_loss, 10))
                
                # Log epoch
                self.storage.log_epoch(
                    trial_id, epoch,
                    avg_val_loss, avg_val_acc, avg_val_ppl, epoch_time
                )
                
                print(f"Epoch {epoch+1}/{self.epochs}: "
                      f"loss={avg_val_loss:.4f}, acc={avg_val_acc:.4f}, "
                      f"ppl={avg_val_ppl:.2f}, time={epoch_time:.1f}s")
            
            # Final metrics
            final_loss = np.mean(val_losses)
            final_acc = np.mean(val_accs)
            final_ppl = np.exp(min(final_loss, 10))
            avg_epoch_time = np.mean(epoch_times)
            avg_iter_time = avg_epoch_time / self.batches_per_epoch
            param_count_millions = algo.param_count / 1e6
            
            # Update trial
            self.storage.update_trial(
                trial_id,
                status='completed',
                epochs_completed=self.epochs,
                final_loss=final_loss,
                accuracy=final_acc,
                perplexity=final_ppl,
                iteration_time=avg_iter_time,
                param_count=param_count_millions
            )
            
            print(f"\n✅ Trial {trial_id} completed successfully!")
            print(f"   Final Accuracy: {final_acc:.4f}")
            print(f"   Final Perplexity: {final_ppl:.2f}")
            print(f"   Avg Iter Time: {avg_iter_time*1000:.1f}ms")
            print(f"   Param Count: {param_count_millions:.2f}M\n")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Trial {trial_id} failed: {e}")
            import traceback
            traceback.print_exc()
            
            self.storage.update_trial(trial_id, status='failed')
            return False
