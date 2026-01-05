#!/bin/bash
# Learning rate sweep for β=0.22
# Test lr ∈ {0.001, 0.0015, 0.002, 0.0025, 0.003}

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

GREEN='\033[0;32m'
NC='\033[0m'

echo -e "${GREEN}======================================================================${NC}"
echo -e "${GREEN}Learning Rate Sweep for β=0.22${NC}"
echo -e "${GREEN}======================================================================${NC}"

mkdir -p logs/lr_sweep_beta022

for lr in 0.001 0.0015 0.002 0.0025 0.003; do
    echo -e "${GREEN}[LR=$lr]${NC} Starting training..."
    
    python train.py \
        --d-model 256 \
        --n-heads 8 \
        --d-ff 1024 \
        --beta 0.22 \
        --damping 0.8 \
        --lr $lr \
        --epochs 20 \
        --dropout 0.1 \
        --compile \
        2>&1 | tee logs/lr_sweep_beta022/lr_${lr}.log
    
    echo -e "${GREEN}[LR=$lr]${NC} Complete!"
    echo ""
done

echo -e "${GREEN}All LR experiments complete!${NC}"
echo ""
echo "Results summary:"
grep "Best test accuracy" logs/lr_sweep_beta022/*.log
