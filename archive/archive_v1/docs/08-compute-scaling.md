# Adaptive Compute Scaling

TorEqProp is designed to **automatically scale** from commodity hardware to datacenter resources. The research plan adapts based on detected compute tier.

---

## Hardware Tier Detection

```python
import torch

def detect_compute_tier() -> str:
    """Auto-detect compute tier based on available GPU resources."""
    if not torch.cuda.is_available():
        return "CPU_ONLY"
    
    gpu_count = torch.cuda.device_count()
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    gpu_name = torch.cuda.get_device_properties(0).name.lower()
    
    # Tier classification
    if gpu_count >= 8 or "h100" in gpu_name or "a100" in gpu_name and gpu_count >= 4:
        return "TIER_4_DATACENTER"
    elif "a100" in gpu_name or "a6000" in gpu_name or gpu_mem_gb >= 40:
        return "TIER_3_HIGH_END"
    elif gpu_mem_gb >= 16 or "3090" in gpu_name or "4090" in gpu_name:
        return "TIER_2_PROSUMER"
    elif gpu_mem_gb >= 6:
        return "TIER_1_COMMODITY"
    else:
        return "CPU_ONLY"

# Usage: CONFIG = TIER_CONFIGS[detect_compute_tier()]
```

---

## Tier Configurations

### Tier 0: CPU Only (Laptop/Debugging)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| d_model | 32 | Minimal viable |
| n_heads | 2 | Reduce computation |
| batch_size | 8 | Memory constraint |
| max_iters | 20 | Fast iteration |
| Dataset | MNIST subset (1k) | Quick validation |

**Research scope**: Gradient verification only. ~2 hours per experiment.

---

### Tier 1: Commodity GPU (6-12GB VRAM)

*Examples: RTX 3060, RTX 4060, GTX 1080 Ti*

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| d_model | 64 | Fits in VRAM |
| n_heads | 4 | Standard ratio |
| batch_size | 32 | Balance speed/memory |
| max_iters | 50 | Full convergence |
| grad_accum_steps | 4 | Simulate larger batch |
| mixed_precision | ✅ fp16 | Essential |
| Dataset | MNIST full | Proof of concept |

**Research scope**: Experiments 1-2 (gradient verification + MNIST training).

**Timeline adjustment**: 
- Week 1-4: Foundation + MNIST
- Scaling experiments deferred to Tier 2+

**Estimated cost**: $0 (local hardware) or ~$50 cloud (spot instances)

---

### Tier 2: Prosumer GPU (16-24GB VRAM)

*Examples: RTX 3090, RTX 4090, A5000*

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| d_model | 128-256 | Primary validation size |
| n_heads | 4-8 | Flexibility |
| batch_size | 64-128 | Efficient |
| max_iters | 50-100 | Full convergence |
| mixed_precision | ✅ fp16/bf16 | Standard |
| checkpointing | Optional | For larger models |
| Dataset | MNIST, CIFAR-10 | Full validation |

**Research scope**: Experiments 1-3 (gradient verification + training + scaling).

**Timeline**: Full 8-week plan achievable.

**Estimated cost**: $0 (local) or ~$200 cloud

---

### Tier 3: High-End Workstation (40-80GB VRAM)

*Examples: A100-40GB, A100-80GB, A6000*

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| d_model | 256-512 | Near-publication scale |
| n_heads | 8-16 | Full expressiveness |
| batch_size | 128-256 | Fast iteration |
| max_iters | 100 | High precision |
| parallel_relaxation | ✅ | Batch-parallelized solver |
| Dataset | MNIST, CIFAR-10, SST-2 | Complete benchmark suite |

**Research scope**: All experiments (1-4) + scaling analysis.

**Additional capabilities**:
- Hyperparameter sweeps (Optuna/Ray Tune)
- Multiple random seeds for statistical significance
- Ablation matrix

**Estimated cost**: $400-600 cloud (1-2 weeks A100)

---

### Tier 4: Datacenter / Multi-GPU

*Examples: 4-8× A100/H100, DGX systems*

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| d_model | 512-1024 | Publication-scale |
| n_heads | 16-32 | Maximum expressiveness |
| batch_size | 512-2048 | Data parallelism |
| max_iters | 100-200 | Precision at scale |
| distributed | ✅ FSDP/DDP | Multi-GPU training |
| Dataset | + WikiText-103, ImageNet-1k | Extended benchmarks |

**Research scope**: Full publication + extension experiments.

**Additional capabilities**:
- Language modeling experiments
- ImageNet classification
- Scaling law analysis (d_model vs. iterations)
- Wall-clock competitive with BP

**Estimated cost**: $1000-3000 cloud

---

## Configuration Presets

```python
TIER_CONFIGS = {
    "CPU_ONLY": {
        "d_model": 32, "n_heads": 2, "d_ff": 128,
        "batch_size": 8, "max_iters": 20, "damping": 0.9,
        "mixed_precision": False, "compile": False,
        "dataset": "mnist_subset", "experiments": [1]
    },
    "TIER_1_COMMODITY": {
        "d_model": 64, "n_heads": 4, "d_ff": 256,
        "batch_size": 32, "max_iters": 50, "damping": 0.9,
        "mixed_precision": True, "compile": True,
        "grad_accum": 4,
        "dataset": "mnist", "experiments": [1, 2]
    },
    "TIER_2_PROSUMER": {
        "d_model": 128, "n_heads": 4, "d_ff": 512,
        "batch_size": 64, "max_iters": 50, "damping": 0.9,
        "mixed_precision": True, "compile": True,
        "dataset": "cifar10", "experiments": [1, 2, 3]
    },
    "TIER_3_HIGH_END": {
        "d_model": 256, "n_heads": 8, "d_ff": 1024,
        "batch_size": 128, "max_iters": 100, "damping": 0.9,
        "mixed_precision": True, "compile": True,
        "parallel_solver": True,
        "dataset": "sst2", "experiments": [1, 2, 3, 4]
    },
    "TIER_4_DATACENTER": {
        "d_model": 512, "n_heads": 16, "d_ff": 2048,
        "batch_size": 512, "max_iters": 100, "damping": 0.9,
        "mixed_precision": True, "compile": True,
        "distributed": True, "parallel_solver": True,
        "dataset": "wikitext103", "experiments": [1, 2, 3, 4, "scaling_laws"]
    }
}
```

---

## Adaptive Training Script

```python
def main():
    tier = detect_compute_tier()
    config = TIER_CONFIGS[tier]
    
    print(f"🔧 Detected compute tier: {tier}")
    print(f"📊 Model size: d={config['d_model']}, heads={config['n_heads']}")
    print(f"🎯 Experiments enabled: {config['experiments']}")
    
    model = LoopedTransformerBlock(
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        d_ff=config["d_ff"]
    )
    
    if config.get("mixed_precision"):
        scaler = torch.cuda.amp.GradScaler()
    
    if config.get("compile") and hasattr(torch, "compile"):
        model = torch.compile(model)
    
    if config.get("distributed"):
        model = torch.nn.parallel.DistributedDataParallel(model)
    
    # Run applicable experiments
    for exp_id in config["experiments"]:
        run_experiment(exp_id, model, config)
```

---

## Progressive Research Path

```
┌─────────────────────────────────────────────────────────────────────┐
│                     PROGRESSIVE RESEARCH PATH                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Tier 0 (CPU)     Tier 1          Tier 2          Tier 3    Tier 4 │
│  ─────────────────────────────────────────────────────────────────  │
│                                                                     │
│  Gradient ────────► MNIST ─────────► CIFAR ────────► SST-2 ──────► │
│  Verify             Training         Scaling         Text    LM    │
│                                                             WikiText│
│  [proof of         [MVP paper]      [+ablations]   [full    [scale]│
│   concept]                                          paper]         │
│                                                                     │
│  Deliverable:      Deliverable:     Deliverable:   Deliverable:    │
│  Blog post /       Workshop paper   Conference     Top venue       │
│  Tech report                        submission     submission      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Memory Optimization Strategies

| Strategy | Memory Saving | Speed Cost | When to Use |
|----------|---------------|------------|-------------|
| Mixed precision (fp16) | 40-50% | ~0% | Always on GPU |
| Gradient checkpointing | 60-70% | 20-30% | Large d_model |
| Micro-batching | Proportional | Linear | Tier 1 |
| torch.compile | Varies | -10% (faster) | PyTorch 2.0+ |
| Activation offload | 80%+ | 50-100% | Last resort |

---

## Scaling Law Experiments (Tier 4)

With datacenter resources, investigate:

1. **Iteration scaling**: How does $T_{converge}$ scale with $d_{model}$?
   - Hypothesis: $T \propto \log(d)$ under proper normalization

2. **β-efficiency scaling**: Optimal β as function of model size
   - Smaller models may tolerate larger β

3. **Memory advantage scaling**: At what $d_{model}$ does O(1) memory dominate?
   - Profile crossover point vs. BP

4. **Wall-clock parity**: When does TorEqProp match BP throughput?
   - Critical for practical adoption

---

## Software Dependencies

```
torch >= 2.0
einops
wandb
scipy (for Anderson acceleration)
optuna (optional, hyperparameter search)
```

### Personnel

| Tier | Researcher Time | Notes |
|------|-----------------|-------|
| Tier 0-1 | Part-time (evenings/weekends) | Hobby project viable |
| Tier 2 | 4 weeks full-time | MVP paper |
| Tier 3-4 | 8 weeks full-time + advisor | Full publication |
