# Performance Optimization Report - TEP Experiment System

## Profiling Results (40.2s total for 2 trials)

### Bottleneck Analysis

| Component | Time (s) | % | Location |
|-----------|----------|---|----------|
| Backward pass | 12.8 | 32% | PyTorch autograd |
| Equilibrium solver | 17.6 | 44% | `src/solver.py` |
| Attention forward | 10.4 | 26% | `src/attention.py` |
| Linear layers | 4.6 | 12% | `torch.nn.linear` |
| Einsum | 2.7 | 7% | `torch.einsum` |

### Recommended Optimizations (Safe & Portable)

#### 1. **Enable Compiled Mode** (PyTorch 2.0+)
```python
# In runner.py _create_model()
model = torch.compile(model, mode='reduce-overhead')
```
Expected: 10-30% speedup

#### 2. **Set GPU Optimizations**
```python
# In runner.py __init__()
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
```
Expected: 5-15% speedup on GPU

#### 3. **Gradient Accumulation Optimization**
```python
# In objectives.py train loops
optimizer.zero_grad(set_to_none=True)  # ✅ Already done
```

#### 4. **DataLoader Optimizations**
```python
# In tasks.py
DataLoader(
    ...,
    num_workers=2,  # Prefetch in parallel
    pin_memory=True if cuda else False,
    persistent_workers=True,
)
```
Expected: 10-20% speedup

#### 5. **Reduce Equilibrium Solver Calls**
Currently: 1260 calls taking 17.6s
- Cache equilibrium states when possible
- Use early stopping more aggressively

#### 6. **Einsum Optimization**
Replace `torch.einsum` with optimized operations:
```python
# Instead of einsum for attention
# Use matmul + reshape which is faster
```

### Implementation Priority

1. ✅ **DataLoader optimizations** - Easy, 10-20% gain
2. ✅ **GPU settings** - One-liner, 5-15% gain
3. ⚠️ **torch.compile** - PyTorch 2.0+ only, 10-30% gain
4. 🔧 **Solver caching** - Requires careful testing
5. 🔧 **Einsum replacement** - Needs validation

### Estimated Total Speedup

Conservative: **25-40% faster**
With all optimizations: **50-70% faster**

This would reduce Phase 1 (300 trials) from ~8 hours to **4-5 hours**.
