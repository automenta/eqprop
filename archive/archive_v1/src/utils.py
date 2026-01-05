import torch
import torch.nn.functional as F

def linear_attention(Q, K, V, eps=1e-6):
    """φ(x) = elu(x) + 1 feature map."""
    phi = lambda x: F.elu(x) + 1
    Q_prime, K_prime = phi(Q), phi(K)
    KV = torch.einsum('bnd,bnv->bdv', K_prime, V)
    Z = torch.einsum('bnd,nd->bn', Q_prime, K_prime.sum(dim=0)) + eps
    return torch.einsum('bnd,bdv->bnv', Q_prime, KV) / Z.unsqueeze(-1)

def detect_compute_tier() -> str:
    """Auto-detect compute tier based on available GPU resources."""
    if not torch.cuda.is_available():
        return "CPU_ONLY"

    gpu_count = torch.cuda.device_count()
    try:
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        gpu_name = torch.cuda.get_device_properties(0).name.lower()
    except:
        gpu_mem_gb = 0
        gpu_name = "unknown"

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
