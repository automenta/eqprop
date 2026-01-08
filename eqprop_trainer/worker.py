"""
Training Worker Thread for EqProp Trainer

Runs training in background thread with rich, real-time progress updates.
"""

import torch
from PyQt6.QtCore import QThread, pyqtSignal
from typing import Optional, Dict, Any
import traceback
import time


class TrainingWorker(QThread):
    """Background worker for model training with real-time updates."""
    
    # Signals
    progress = pyqtSignal(dict)  # Emit training metrics frequently
    finished = pyqtSignal(dict)  # Emit final results
    error = pyqtSignal(str)      # Emit error message
    generation = pyqtSignal(str) # Emit generated text
    weights_updated = pyqtSignal(dict)  # Emit weight snapshots for visualization
    
    def __init__(
        self,
        model,
        train_loader,
        epochs: int = 10,
        lr: float = 0.001,
        use_compile: bool = True,
        generate_interval: int = 5,  # Generate text every N epochs
        prompts: list = None,
        hyperparams: dict = None,  # Model-specific hyperparameters
        parent=None,
    ):
        super().__init__(parent)
        self.model = model
        self.train_loader = train_loader
        self.epochs = epochs
        self.lr = lr
        self.use_compile = use_compile
        self.generate_interval = generate_interval
        self.prompts = prompts or ["ROMEO:"]
        self.hyperparams = hyperparams or {}
        
        self._stop_requested = False
        
    def stop(self):
        """Request training stop."""
        self._stop_requested = True
    
    def run(self):
        """Run training loop with rich real-time feedback."""
        try:
            from eqprop_torch import EqPropTrainer
            
            trainer = EqPropTrainer(
                self.model,
                lr=self.lr, 
                use_compile=self.use_compile,
            )
            
            # Apply dynamic hyperparameters
            if self.hyperparams:
                for name, value in self.hyperparams.items():
                    # Apply to trainer if applicable (e.g. beta, nudge_steps)
                    if hasattr(trainer, name):
                        setattr(trainer, name, value)
                    # Apply to model config (Research Algorithms)
                    elif hasattr(self.model, 'config') and hasattr(self.model.config, name):
                        setattr(self.model.config, name, value)
                    # Apply to model directly (Legacy/Simple models)
                    elif hasattr(self.model, name):
                        setattr(self.model, name, value)
            
            num_batches = len(self.train_loader)
            total_start = time.time()
            
            for epoch in range(self.epochs):
                if self._stop_requested:
                    break
                
                epoch_start = time.time()
                
                # Training epoch
                epoch_loss = 0.0
                epoch_correct = 0
                epoch_total = 0
                
                for batch_idx, (x, y) in enumerate(self.train_loader):
                    if self._stop_requested:
                        break
                    
                    batch_start = time.time()
                    
                    x = x.to(trainer.device)
                    y = y.to(trainer.device)
                    
                    # Handle input conversions
                    if x.dim() == 4 and hasattr(self.model, 'input_dim'):
                        # Vision: flatten images
                        x = x.view(x.size(0), -1)
                    elif x.dtype == torch.long:
                        # LM tokens: convert to float for bioplausible models
                        if hasattr(self.model, 'config'):
                            vocab_size = self.model.config.input_dim
                            # Flatten sequence tokens and one-hot encode
                            x_flat = x.reshape(-1)
                            x = torch.nn.functional.one_hot(x_flat, num_classes=vocab_size).float()
                            if y.dim() > 1:
                                y = y.reshape(-1)

                    
                    trainer.optimizer.zero_grad()
                    
                    # Forward pass
                    output = self.model(x)
                    
                    # Compute loss (handle both vision and LM cases)
                    if output.dim() == 3:
                        # Language modeling: output is [batch, seq_len, vocab_size]
                        batch_size, seq_len, vocab_size = output.shape
                        output_flat = output.reshape(batch_size * seq_len, vocab_size)
                        y_flat = y.reshape(batch_size * seq_len)
                        loss = torch.nn.functional.cross_entropy(output_flat, y_flat)
                        
                        pred = output.argmax(dim=-1)
                        batch_correct = (pred == y).sum().item()
                        batch_total = batch_size * seq_len
                    else:
                        # Vision: output is [batch, num_classes]
                        loss = torch.nn.functional.cross_entropy(output, y)
                        
                        pred = output.argmax(dim=1)
                        batch_correct = (pred == y).sum().item()
                        batch_total = x.size(0)
                    
                    # Backward and optimize
                    loss.backward()
                    trainer.optimizer.step()
                    
                    batch_time = time.time() - batch_start
                    
                    # Accumulate metrics
                    epoch_loss += loss.item() * x.size(0)
                    epoch_correct += batch_correct
                    epoch_total += batch_total
                    
                    # Emit batch-level progress every 10 batches or on last batch
                    if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
                        current_loss = epoch_loss / max(epoch_total, 1)
                        current_acc = epoch_correct / max(epoch_total, 1)
                        
                        # Time estimates
                        elapsed_total = time.time() - total_start
                        batches_done_total = epoch * num_batches + batch_idx + 1
                        batches_remaining = self.epochs * num_batches - batches_done_total
                        avg_batch_time = elapsed_total / batches_done_total
                        eta_seconds = batches_remaining * avg_batch_time
                        
                        # Throughput
                        samples_per_sec = x.size(0) / max(batch_time, 0.001)
                        
                        # Emit rich progress
                        metrics = {
                            'epoch': epoch + 1,
                            'total_epochs': self.epochs,
                            'batch': batch_idx + 1,
                            'total_batches': num_batches,
                            'loss': current_loss,
                            'batch_loss': loss.item(),
                            'accuracy': current_acc,
                            'batch_accuracy': batch_correct / max(batch_total, 1),
                            'lipschitz': 0.0,  # Placeholder, computed below
                            'samples_per_sec': samples_per_sec,
                            'eta_seconds': eta_seconds,
                            'progress': (batches_done_total / (self.epochs * num_batches)) * 100,
                        }
                        self.progress.emit(metrics)
                        
                        # Emit weight snapshots for visualization
                        if (batch_idx + 1) % 10 == 0:
                            try:
                                from .viz_utils import extract_weights
                                weights = extract_weights(self.model)
                                if weights:
                                    self.weights_updated.emit(weights)
                            except Exception:
                                pass  # Ignore visualization errors
                
                # End of epoch: compute Lipschitz
                lipschitz = trainer.compute_lipschitz() or 0.0
                
                # Final epoch metrics
                final_metrics = {
                    'epoch': epoch + 1,
                    'total_epochs': self.epochs,
                    'batch': num_batches,
                    'total_batches': num_batches,
                    'loss': epoch_loss / max(epoch_total, 1),
                    'accuracy': epoch_correct / max(epoch_total, 1),
                    'lipschitz': lipschitz,
                    'samples_per_sec': 0.0,
                    'eta_seconds': 0.0,
                    'progress': ((epoch + 1) / self.epochs) * 100,
                }
                self.progress.emit(final_metrics)
                
                # Generate text periodically
                if (epoch + 1) % self.generate_interval == 0 and hasattr(self.model, 'generate'):
                    try:
                        for prompt in self.prompts:
                            text = self.model.generate(prompt, max_new_tokens=100)
                            self.generation.emit(text)
                    except Exception:
                        pass  # Ignore generation errors
            
            self.finished.emit({'success': True, 'epochs_completed': epoch + 1})
            
        except Exception as e:
            self.error.emit(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


class GenerationWorker(QThread):
    """Background worker for text generation."""
    
    result = pyqtSignal(str)
    error = pyqtSignal(str)
    
    def __init__(self, model, prompt: str, max_tokens: int = 100, temperature: float = 1.0, parent=None):
        super().__init__(parent)
        self.model = model
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
    
    def run(self):
        try:
            if hasattr(self.model, 'generate'):
                text = self.model.generate(
                    self.prompt,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                self.result.emit(text)
            else:
                self.error.emit("Model does not support generation")
        except Exception as e:
            self.error.emit(str(e))
