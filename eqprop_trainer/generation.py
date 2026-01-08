"""
Universal Text Generation for EqProp Trainer

Enables text generation for ANY model, including Vision models without native
generate() methods. Uses autoregressive next-token prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Optional


class SimpleCharTokenizer:
    """Simple character-level tokenizer for universal generation."""
    
    def __init__(self, chars: str = None):
        """Initialize with character vocabulary."""
        if chars is None:
            # Default: printable ASCII
            chars = ''.join(chr(i) for i in range(32, 127))
        
        self.chars = chars
        self.vocab_size = len(chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
    
    def encode(self, text: str) -> list:
        """Convert text to list of indices."""
        return [self.char_to_idx.get(ch, 0) for ch in text]
    
    def decode(self, indices: list) -> str:
        """Convert indices to text."""
        return ''.join(self.idx_to_char.get(idx, '?') for idx in indices)


class UniversalGenerator:
    """
    Universal text generator that works with ANY PyTorch model.
    
    For models without .generate() method, uses autoregressive prediction.
    Works with Vision models, LM models, or any classifier.
    """
    
    def __init__(
        self, 
        model: nn.Module,
        vocab_size: int = 95,  # Printable ASCII
        tokenizer: Optional[SimpleCharTokenizer] = None,
        device: str = 'cpu'
    ):
        """
        Args:
            model: Any PyTorch model with forward() method
            vocab_size: Size of output vocabulary
            tokenizer: Optional tokenizer (creates default if None)
            device: Device for generation
        """
        self.model = model
        self.vocab_size = vocab_size
        self.device = device
        
        if tokenizer is None:
            tokenizer = SimpleCharTokenizer()
        self.tokenizer = tokenizer
        
        # Check if model has native generation
        self.has_native_generate = hasattr(model, 'generate') and callable(model.generate)
    
    def generate(
        self,
        prompt: str = "",
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Starting text
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling (0 = disabled)
            
        Returns:
            Generated text
        """
        # Use native generation if available
        if self.has_native_generate:
            try:
                # Encode prompt to tokens
                if isinstance(prompt, str):
                    prompt_tokens = torch.tensor(
                        self.tokenizer.encode(prompt), 
                        dtype=torch.long,
                        device=self.device
                    ).unsqueeze(0)
                else:
                    prompt_tokens = prompt
                
                # Call native generate
                output_tokens = self.model.generate(
                    prompt_tokens,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature
                )
                
                # Decode back to text
                if isinstance(output_tokens, torch.Tensor):
                    output_tokens = output_tokens[0].tolist()
                return self.tokenizer.decode(output_tokens)
            except Exception as e:
                print(f"Native generation failed: {e}, falling back to autoregressive")
        
        # Fallback: autoregressive generation
        return self._autoregressive_generate(prompt, max_new_tokens, temperature, top_k)
    
    def _autoregressive_generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
    ) -> str:
        """Autoregressive generation for models without native generate()."""
        self.model.eval()
        
        # Encode prompt
        if not prompt:
            prompt = " "  # Start with space if empty
        
        tokens = self.tokenizer.encode(prompt)
        generated_tokens = tokens.copy()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Prepare input - different strategies for different model types
                if hasattr(self.model, 'input_dim'):
                    # Vision model: use last token as flattened input
                    input_tensor = torch.zeros(1, self.model.input_dim, device=self.device)
                    # One-hot encode last token
                    if len(generated_tokens) > 0:
                        last_token = generated_tokens[-1] % self.vocab_size
                        if last_token < self.model.input_dim:
                            input_tensor[0, last_token] = 1.0
                elif hasattr(self.model, 'token_emb'):
                    # LM model: use tokens directly
                    input_tensor = torch.tensor(
                        generated_tokens[-min(len(generated_tokens), 128):],  # Last 128 tokens
                        dtype=torch.long,
                        device=self.device
                    ).unsqueeze(0)
                else:
                    # Generic: one-hot encode last token
                    input_tensor = torch.zeros(1, self.vocab_size, device=self.device)
                    if len(generated_tokens) > 0:
                        last_token = generated_tokens[-1] % self.vocab_size
                        input_tensor[0, last_token] = 1.0
                
                # Forward pass
                try:
                    output = self.model(input_tensor)
                except Exception as e:
                    print(f"Generation forward pass failed: {e}")
                    break
                
                # Get logits for next token
                if output.dim() == 3:
                    # LM output: [batch, seq, vocab]
                    logits = output[0, -1, :]
                elif output.dim() == 2:
                    # Vision output: [batch, classes]
                    logits = output[0, :]
                else:
                    print(f"Unexpected output shape: {output.shape}")
                    break
                
                # Ensure logits are correct size
                if logits.shape[0] != self.vocab_size:
                    # Pad or trim logits to vocab size
                    if logits.shape[0] < self.vocab_size:
                        padding = torch.zeros(
                            self.vocab_size - logits.shape[0],
                            device=logits.device
                        )
                        logits = torch.cat([logits, padding])
                    else:
                        logits = logits[:self.vocab_size]
                
                # Apply temperature
                logits = logits / max(temperature, 0.01)
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, min(top_k, logits.shape[0]))[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                
                generated_tokens.append(next_token)
        
        # Decode to text
        return self.tokenizer.decode(generated_tokens)


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_parameter_count(count: int) -> str:
    """Format parameter count in human-readable form."""
    if count >= 1_000_000:
        return f"{count / 1_000_000:.2f}M"
    elif count >= 1_000:
        return f"{count / 1_000:.1f}K"
    else:
        return str(count)
