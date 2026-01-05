#!/bin/bash
# Architecture scaling experiments
# Test larger models with optimal β=0.22

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}======================================================================${NC}"
echo -e "${GREEN}Architecture Scaling Experiments${NC}"
echo -e "${GREEN}======================================================================${NC}"
echo ""

mkdir -p logs/architecture_scaling

# Experiment 1: d_model=512 (2x capacity)
echo -e "${GREEN}[1/2] Training d_model=512${NC}"
echo "  Configuration: d=512, n_heads=16, d_ff=2048"
echo "  Epochs: 30"
echo -e "  ${YELLOW}Estimated time: ~2 hours${NC}"
echo ""

python train.py \
    --d-model 512 \
    --n-heads 16 \
    --d-ff 2048 \
    --beta 0.22 \
    --damping 0.8 \
    --lr 0.002 \
    --epochs 30 \
    --dropout 0.1 \
    --compile \
    2>&1 | tee logs/architecture_scaling/d512_h16_ff2048.log

echo -e "${GREEN}[1/2] Complete!${NC}"
echo ""

# Experiment 2: Larger FFN only (d_model=256, d_ff=2048)
echo -e "${GREEN}[2/2] Training with larger FFN (d_ff=2048)${NC}"
echo "  Configuration: d=256, n_heads=8, d_ff=2048"
echo "  Epochs: 30"
echo -e "  ${YELLOW}Estimated time: ~2 hours${NC}"
echo ""

python train.py \
    --d-model 256 \
    --n-heads 8 \
    --d-ff 2048 \
    --beta 0.22 \
    --damping 0.8 \
    --lr 0.002 \
    --epochs 30 \
    --dropout 0.1 \
    --compile \
    2>&1 | tee logs/architecture_scaling/d256_h8_ff2048.log

echo -e "${GREEN}[2/2] Complete!${NC}"
echo ""

echo -e "${GREEN}======================================================================${NC}"
echo -e "${GREEN}Architecture Scaling Complete!${NC}"
echo -e "${GREEN}======================================================================${NC}"
echo ""
echo "Results summary:"
echo ""
echo "Baseline (d=256, ff=1024): 92.37%"
echo ""
grep "Best test accuracy" logs/architecture_scaling/*.log
echo ""
