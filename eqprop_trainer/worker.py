"""
Training Worker Thread for EqProp Trainer

Runs training in background thread to keep UI responsive.
"""

import torch
from PyQt6.QtCore import QThread, pyqtSignal
from typing import Optional, Dict, Any
import traceback


class TrainingWorker(QThread):
    """Background worker for model training."""
    
    # Signals
    progress = pyqtSignal(dict)  # Emit training metrics each step
    finished = pyqtSignal(dict)  # Emit final results
    error = pyqtSignal(str)      # Emit error message
    generation = pyqtSignal(str) # Emit generated text
    
    def __init__(
        self,
        model,
        train_loader,
        epochs: int = 10,
        lr: float = 0.001,
        use_compile: bool = True,
        generate_interval: int = 5,  # Generate text every N epochs
        prompts: list = None,
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
        
        self._stop_requested = False
        
    def stop(self):
        """Request training stop."""
        self._stop_requested = True
    
    def run(self):
        """Run training loop."""
        try:
            from eqprop_torch import EqPropTrainer
            
            trainer = EqPropTrainer(
                self.model,
                lr=self.lr, 
                use_compile=self.use_compile,
            )
            
            for epoch in range(self.epochs):
                if self._stop_requested:
                    break
                
                # Training epoch
                epoch_loss = 0.0
                epoch_correct = 0
                epoch_total = 0
                
                for batch_idx, (x, y) in enumerate(self.train_loader):
                    if self._stop_requested:
                        break
                    
                    x = x.to(trainer.device)
                    y = y.to(trainer.device)
                    
                    # Flatten if needed
                    if x.dim() == 4 and hasattr(self.model, 'input_dim'):
                        x = x.view(x.size(0), -1)
                    
                    trainer.optimizer.zero_grad()
                    output = self.model(x)
                    loss = torch.nn.functional.cross_entropy(output, y)
                    loss.backward()
                    trainer.optimizer.step()
                    
                    epoch_loss += loss.item() * x.size(0)
                    _, predicted = output.max(1)
                    epoch_correct += predicted.eq(y).sum().item()
                    epoch_total += x.size(0)
                
                # Compute Lipschitz
                lipschitz = trainer.compute_lipschitz() or 0.0
                
                # Emit progress
                metrics = {
                    'epoch': epoch + 1,
                    'total_epochs': self.epochs,
                    'loss': epoch_loss / max(epoch_total, 1),
                    'accuracy': epoch_correct / max(epoch_total, 1),
                    'lipschitz': lipschitz,
                    'progress': (epoch + 1) / self.epochs * 100,
                }
                self.progress.emit(metrics)
                
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
