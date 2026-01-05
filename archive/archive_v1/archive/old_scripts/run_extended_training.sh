#!/bin/bash
# Extended Training with Optimal β=0.22
# Goal: Push accuracy from 92.37% (15 epochs) to 94%+ (50 epochs)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}======================================================================${NC}"
echo -e "${GREEN}Extended Training: β=0.22 for 50 Epochs${NC}"
echo -e "${GREEN}======================================================================${NC}"
echo ""
echo "Configuration:"
echo "  β: 0.22 (fixed - no annealing!)"
echo "  d_model: 256"
echo "  Epochs: 50 (vs 15 baseline)"
echo "  Target: 94%+ accuracy"
echo ""
echo -e "${YELLOW}Estimated time: ~3.5 hours${NC}"
echo ""

read -p "Press Enter to start training..."

python train.py \
    --d-model 256 \
    --n-heads 8 \
    --d-ff 1024 \
    --beta 0.22 \
    --damping 0.8 \
    --lr 0.002 \
    --epochs 50 \
    --dropout 0.1 \
    --compile \
    2>&1 | tee logs/extended_beta022_50ep.log

echo ""
echo -e "${GREEN}======================================================================${NC}"
echo -e "${GREEN}Training Complete!${NC}"
echo -e "${GREEN}======================================================================${NC}"
echo ""
echo "Log file: logs/extended_beta022_50ep.log"
echo "Checkpoint: checkpoints/best_mnist.pt"
echo ""
