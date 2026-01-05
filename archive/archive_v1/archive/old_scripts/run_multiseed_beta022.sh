#!/bin/bash
# Multi-seed validation for β=0.22
# Run 5 independent training runs with different random seeds

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}======================================================================${NC}"
echo -e "${GREEN}Multi-Seed Validation: β=0.22${NC}"
echo -e "${GREEN}======================================================================${NC}"
echo ""
echo "Running 5 independent training runs with seeds 1-5"
echo "Configuration: β=0.22, d_model=256, 30 epochs each"
echo ""
echo -e "${YELLOW}Estimated total time: ~5 hours${NC}"
echo ""

mkdir -p logs/multiseed_beta022

for seed in 1 2 3 4 5; do
    echo -e "${GREEN}[Seed $seed/5]${NC} Starting training..."
    
    python train.py \
        --d-model 256 \
        --n-heads 8 \
        --d-ff 1024 \
        --beta 0.22 \
        --damping 0.8 \
        --lr 0.002 \
        --epochs 30 \
        --dropout 0.1 \
        --seed $seed \
        --compile \
        2>&1 | tee logs/multiseed_beta022/seed_${seed}.log
    
    echo -e "${GREEN}[Seed $seed/5]${NC} Complete!"
    echo ""
done

echo -e "${GREEN}======================================================================${NC}"
echo -e "${GREEN}Multi-Seed Validation Complete!${NC}"
echo -e "${GREEN}======================================================================${NC}"
echo ""
echo "Computing statistics..."

# Extract final test accuracies
python - <<EOF
import re
from pathlib import Path

accuracies = []
for seed in range(1, 6):
    log_file = Path(f"logs/multiseed_beta022/seed_{seed}.log")
    with open(log_file) as f:
        content = f.read()
        # Find best test accuracy
        match = re.search(r'Best test accuracy: ([\d.]+)', content)
        if match:
            acc = float(match.group(1))
            accuracies.append(acc)
            print(f"Seed {seed}: {acc:.4f} ({acc*100:.2f}%)")

if accuracies:
    import statistics
    mean = statistics.mean(accuracies)
    stdev = statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0
    print(f"\nStatistics:")
    print(f"  Mean: {mean:.4f} ({mean*100:.2f}%)")
    print(f"  Std Dev: {stdev:.4f} ({stdev*100:.2f}%)")
    print(f"  Min: {min(accuracies):.4f}")
    print(f"  Max: {max(accuracies):.4f}")
EOF

echo ""
