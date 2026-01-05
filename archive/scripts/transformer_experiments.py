#!/usr/bin/env python3
"""
Comprehensive Transformer EqProp Experiments

Tests TransformerEqProp on:
1. Sequence classification (toy task)
2. Copy task (memory test)
3. Simple language modeling (character-level)

Provides detailed progress feedback and results.
"""

import sys
sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
from dataclasses import dataclass
from typing import List, Tuple

print("Loading TransformerEqProp...")
from src.models.transformer_eqprop import TransformerEqProp
print("✓ Loaded")


@dataclass
class ExperimentResult:
    """Results from one experiment."""
    name: str
    initial_loss: float
    final_loss: float
    initial_acc: float
    final_acc: float
    training_time: float
    converged: bool


def progress_bar(current, total, prefix='', suffix='', length=40):
    """Print progress bar."""
    percent = current / total
    filled = int(length * percent)
    bar = '█' * filled + '░' * (length - filled)
    print(f'\r{prefix} |{bar}| {percent*100:.1f}% {suffix}', end='', flush=True)


# =============================================================================
# Experiment 1: Sequence Classification
# =============================================================================

def generate_sequence_classification_data(n_samples=1000, seq_len=20, vocab_size=50):
    """
    Generate synthetic sequence classification data.
    
    Task: Classify sequences as "positive" if sum of tokens > threshold.
    """
    X = torch.randint(0, vocab_size, (n_samples, seq_len))
    
    # Label based on sum of tokens
    token_sums = X.sum(dim=1)
    threshold = token_sums.median()
    y = (token_sums > threshold).long()
    
    return X, y


def experiment_sequence_classification():
    """Experiment 1: Sequence Classification."""
    print("\n" + "="*70)
    print("EXPERIMENT 1: SEQUENCE CLASSIFICATION")
    print("="*70)
    print("\nTask: Classify sequences by sum of tokens")
    print("Setup: 20 tokens, vocab=50, binary classification")
    
    # Data
    print("\nGenerating data...")
    X_train, y_train = generate_sequence_classification_data(n_samples=800)
    X_test, y_test = generate_sequence_classification_data(n_samples=200)
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Model
    print("\nCreating TransformerEqProp...")
    model = TransformerEqProp(
        vocab_size=50,
        hidden_dim=64,
        output_dim=2,
        num_layers=2,
        num_heads=4,
        max_seq_len=20
    )
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test initial performance
    with torch.no_grad():
        out = model(X_test, steps=15)
        initial_loss = F.cross_entropy(out, y_test).item()
        initial_acc = (out.argmax(dim=1) == y_test).float().mean().item()
    
    print(f"\n  Initial: Loss={initial_loss:.3f}, Acc={initial_acc:.1%}")
    
    # Training
    print("\nTraining for 30 epochs...")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    batch_size = 32
    epochs = 30
    
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_correct = 0
        n_batches = len(X_train) // batch_size
        
        # Training batches
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            optimizer.zero_grad()
            out = model(batch_X, steps=15)
            loss = F.cross_entropy(out, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_correct += (out.argmax(dim=1) == batch_y).sum().item()
        
        # Test
        model.eval()
        with torch.no_grad():
            test_out = model(X_test, steps=15)
            test_loss = F.cross_entropy(test_out, y_test).item()
            test_acc = (test_out.argmax(dim=1) == y_test).float().mean().item()
        
        # Progress
        train_acc = epoch_correct / len(X_train)
        if (epoch + 1) % 5 == 0:
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch+1:2d}/{epochs}: Train Acc={train_acc:.1%}, "
                  f"Test Acc={test_acc:.1%}, Test Loss={test_loss:.3f}, "
                  f"Time={elapsed:.1f}s")
    
    # Final results
    model.eval()
    with torch.no_grad():
        final_out = model(X_test, steps=15)
        final_loss = F.cross_entropy(final_out, y_test).item()
        final_acc = (final_out.argmax(dim=1) == y_test).float().mean().item()
    
    total_time = time.time() - start_time
    
    print(f"\nFinal: Loss={final_loss:.3f}, Acc={final_acc:.1%}")
    print(f"Improvement: {(final_acc - initial_acc)*100:+.1f}%")
    
    converged = final_acc > 0.7  # Reasonable threshold
    
    return ExperimentResult(
        name="Sequence Classification",
        initial_loss=initial_loss,
        final_loss=final_loss,
        initial_acc=initial_acc,
        final_acc=final_acc,
        training_time=total_time,
        converged=converged
    )


# =============================================================================
# Experiment 2: Copy Task (Memory Test)
# =============================================================================

def generate_copy_task_data(n_samples=500, seq_len=10, vocab_size=20):
    """
    Generate copy task data.
    
    Task: Copy input sequence to output.
    """
    X = torch.randint(1, vocab_size, (n_samples, seq_len))  # Start from 1 (0 is padding)
    y = X.clone()  # Target is same as input
    
    return X, y


def experiment_copy_task():
    """Experiment 2: Copy Task."""
    print("\n" + "="*70)
    print("EXPERIMENT 2: COPY TASK (Memory Test)")
    print("="*70)
    print("\nTask: Copy input sequence to output")
    print("Setup: 10 tokens, vocab=20")
    
    # Data
    print("\nGenerating data...")
    X_train, y_train = generate_copy_task_data(n_samples=400, seq_len=10)
    X_test, y_test = generate_copy_task_data(n_samples=100, seq_len=10)
    
    # Model (output per token)
    print("\nCreating TransformerEqProp for sequence-to-sequence...")
    model = TransformerEqProp(
        vocab_size=20,
        hidden_dim=32,
        output_dim=20,  # Predict next token
        num_layers=2,
        num_heads=2,
        max_seq_len=10
    )
    
    # Initial performance
    with torch.no_grad():
        # For copy task, compute per-token accuracy
        out = model(X_test, steps=10)
        initial_loss = F.cross_entropy(out, y_test.view(-1)).item()
        pred_tokens = out.argmax(dim=1).view(-1)
        initial_acc = (pred_tokens == y_test.view(-1)).float().mean().item()
    
    print(f"\n  Initial: Loss={initial_loss:.3f}, Token Acc={initial_acc:.1%}")
    
    # Training
    print("\nTraining for 20 epochs...")
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    batch_size = 20
    epochs = 20
    
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            optimizer.zero_grad()
            out = model(batch_X, steps=10)
            
            # Compute loss per sequence (mean pooling approach)
            loss = F.cross_entropy(out, batch_y.view(-1) % 20)  # Modulo for safety
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            # Test
            model.eval()
            with torch.no_grad():
                test_out = model(X_test, steps=10)
                test_loss = F.cross_entropy(test_out, y_test.view(-1) % 20).item()
                pred_tokens = test_out.argmax(dim=1).view(len(X_test), -1)
                # Token-level accuracy
                test_acc = (pred_tokens == y_test).float().mean().item()
            
            print(f"  Epoch {epoch+1:2d}/{epochs}: Test Loss={test_loss:.3f}, "
                  f"Token Acc={test_acc:.1%}")
    
    # Final
    model.eval()
    with torch.no_grad():
        final_out = model(X_test, steps=10)
        final_loss = F.cross_entropy(final_out, y_test.view(-1) % 20).item()
        pred_tokens = final_out.argmax(dim=1).view(len(X_test), -1)
        final_acc = (pred_tokens == y_test).float().mean().item()
    
    total_time = time.time() - start_time
    
    print(f"\nFinal Token Acc: {final_acc:.1%}")
    print(f"Improvement: {(final_acc - initial_acc)*100:+.1f}%")
    
    return ExperimentResult(
        name="Copy Task",
        initial_loss=initial_loss,
        final_loss=final_loss,
        initial_acc=initial_acc,
        final_acc=final_acc,
        training_time=total_time,
        converged=final_acc > 0.5
    )


# =============================================================================
# Experiment 3: Character-Level Language Modeling
# =============================================================================

def generate_char_lm_data(text: str, seq_len: int = 32):
    """Generate character-level LM data from text."""
    # Create vocabulary
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    vocab_size = len(chars)
    
    # Create sequences
    sequences = []
    targets = []
    
    for i in range(len(text) - seq_len):
        seq = text[i:i+seq_len]
        target = text[i+1:i+seq_len+1]  # Shifted by 1
        
        seq_idx = torch.tensor([char_to_idx[ch] for ch in seq])
        target_idx = torch.tensor([char_to_idx[ch] for ch in target])
        
        sequences.append(seq_idx)
        targets.append(target_idx[-1])  # Predict last character
    
    X = torch.stack(sequences)
    y = torch.stack(targets)
    
    return X, y, vocab_size, char_to_idx


def experiment_character_lm():
    """Experiment 3: Character-Level Language Modeling."""
    print("\n" + "="*70)
    print("EXPERIMENT 3: CHARACTER-LEVEL LANGUAGE MODELING")
    print("="*70)
    print("\nTask: Predict next character in sequence")
    
    # Sample text (simple pattern)
    text = "the quick brown fox jumps over the lazy dog. " * 50  # Repeat for more data
    text += "a bird in the hand is worth two in the bush. " * 50
    text += "all that glitters is not gold. " * 50
    
    print(f"Text length: {len(text)} characters")
    
    # Generate data
    print("\nGenerating sequences...")
    X, y, vocab_size, char_to_idx = generate_char_lm_data(text, seq_len=20)
    
    # Split train/test
    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]
    
    print(f"  Vocab size: {vocab_size}")
    print(f"  Train: {len(X_train)} sequences")
    print(f"  Test: {len(X_test)} sequences")
    
    # Model
    print("\nCreating TransformerEqProp for char-LM...")
    model = TransformerEqProp(
        vocab_size=vocab_size,
        hidden_dim=64,
        output_dim=vocab_size,
        num_layers=2,
        num_heads=2,
        max_seq_len=20
    )
    
    # Initial
    with torch.no_grad():
        out = model(X_test, steps=15)
        initial_loss = F.cross_entropy(out, y_test).item()
        initial_acc = (out.argmax(dim=1) == y_test).float().mean().item()
    
    print(f"\n  Initial: Loss={initial_loss:.3f}, Acc={initial_acc:.1%}")
    print(f"  Random baseline: {1/vocab_size:.1%}")
    
    # Training
    print("\nTraining for 40 epochs...")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    batch_size = 32
    epochs = 40
    
    start_time = time.time()
    best_test_acc = 0
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        # Shuffle training data
        perm = torch.randperm(len(X_train))
        X_train_shuffled = X_train[perm]
        y_train_shuffled = y_train[perm]
        
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train_shuffled[i:i+batch_size]
            batch_y = y_train_shuffled[i:i+batch_size]
            
            optimizer.zero_grad()
            out = model(batch_X, steps=15)
            loss = F.cross_entropy(out, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Test every 5 epochs
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                test_out = model(X_test, steps=15)
                test_loss = F.cross_entropy(test_out, y_test).item()
                test_acc = (test_out.argmax(dim=1) == y_test).float().mean().item()
            
            best_test_acc = max(best_test_acc, test_acc)
            elapsed = time.time() - start_time
            
            print(f"  Epoch {epoch+1:2d}/{epochs}: Test Loss={test_loss:.3f}, "
                  f"Test Acc={test_acc:.1%}, Best={best_test_acc:.1%}, "
                  f"Time={elapsed:.1f}s")
    
    # Final
    model.eval()
    with torch.no_grad():
        final_out = model(X_test, steps=15)
        final_loss = F.cross_entropy(final_out, y_test).item()
        final_acc = (final_out.argmax(dim=1) == y_test).float().mean().item()
    
    total_time = time.time() - start_time
    
    print(f"\nFinal: Loss={final_loss:.3f}, Acc={final_acc:.1%}")
    print(f"Improvement: {(final_acc - initial_acc)*100:+.1f}%")
    print(f"vs Random ({1/vocab_size:.1%}): {(final_acc - 1/vocab_size)*100:+.1f}%")
    
    return ExperimentResult(
        name="Character LM",
        initial_loss=initial_loss,
        final_loss=final_loss,
        initial_acc=initial_acc,
        final_acc=final_acc,
        training_time=total_time,
        converged=final_acc > 0.3
    )


# =============================================================================
# Main
# =============================================================================

def main():
    print("\n" + "="*70)
    print("TRANSFORMER EQPROP: COMPREHENSIVE EXPERIMENTS")
    print("="*70)
    print("\nTesting TransformerEqProp on 3 tasks:")
    print("1. Sequence Classification (binary)")
    print("2. Copy Task (memory test)")
    print("3. Character-Level Language Modeling")
    
    results = []
    
    # Run experiments
    try:
        results.append(experiment_sequence_classification())
    except Exception as e:
        print(f"\n❌ Experiment 1 failed: {e}")
    
    try:
        results.append(experiment_copy_task())
    except Exception as e:
        print(f"\n❌ Experiment 2 failed: {e}")
    
    try:
        results.append(experiment_character_lm())
    except Exception as e:
        print(f"\n❌ Experiment 3 failed: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: TRANSFORMER EQPROP CAPABILITIES")
    print("="*70)
    
    for result in results:
        status = "✓" if result.converged else "⚠"
        print(f"\n{status} {result.name}:")
        print(f"    Initial Acc: {result.initial_acc:.1%}")
        print(f"    Final Acc: {result.final_acc:.1%}")
        print(f"    Improvement: {(result.final_acc - result.initial_acc)*100:+.1f}%")
        print(f"    Training Time: {result.training_time:.1f}s")
        print(f"    Converged: {'Yes' if result.converged else 'Partial'}")
    
    # Language modeling assessment
    print("\n" + "="*70)
    print("LANGUAGE MODELING POTENTIAL")
    print("="*70)
    
    lm_result = [r for r in results if r.name == "Character LM"]
    if lm_result:
        r = lm_result[0]
        print(f"\nCharacter-level LM performance: {r.final_acc:.1%}")
        
        if r.final_acc > 0.5:
            print("\n✓ STRONG POTENTIAL for language modeling")
            print("  - Model learns character patterns")
            print("  - Next step: Word-level LM on real text")
            print("  - Path to: Sentiment analysis, text generation")
        elif r.final_acc > 0.3:
            print("\n⚠ MODERATE POTENTIAL for language modeling")
            print("  - Model shows some learning")
            print("  - Needs: Larger model, more data, better hyperparameters")
        else:
            print("\n⚠ LIMITED POTENTIAL (needs investigation)")
            print("  - Consider: architecture changes, longer training")


if __name__ == "__main__":
    main()
