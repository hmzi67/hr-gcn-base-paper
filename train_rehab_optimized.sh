#!/bin/bash
# Optimized training script for GCADA rehabilitation pipeline
# Fixes: (1) Remove 11-epoch warmup, (2) Progressive loss weighting, (3) Better defaults
#
# RECOMMENDED APPROACH: From-scratch training on UI-PRMD
# Rationale: H3WB -> Vicon mocap is severe domain shift (different sensors, projections)

set -e

echo "=========================================="
echo "GCADA Rehab Training - Optimized"
echo "=========================================="
echo ""

# Configuration
EPOCHS=50
BATCH_SIZE=64
LR=5e-4
WARMUP_EPOCHS=3
CHECKPOINT_DIR="checkpoint_rehab_v3"
CONFIG="w32_adam_lr1e-3.yaml"

echo "Settings:"
echo "  Training approach: FROM-SCRATCH (no pretrained H3WB model)"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE (was 256 - smaller for domain adaptation)"
echo "  Learning rate: $LR (was 1e-4)"
echo "  Warmup epochs: $WARMUP_EPOCHS (was 11)"
echo "  Progressive loss weighting: ENABLED (0.1x -> 1.0x)"
echo "  Checkpoint dir: $CHECKPOINT_DIR"
echo ""

# Activate environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "Virtual environment activated"
else
    echo "WARNING: No .venv found, proceeding anyway"
fi

# Run training
python train_rehab.py \
    --cfg "$CONFIG" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --backbone_lr_factor 1.0 \
    --warmup_epochs $WARMUP_EPOCHS \
    --progressive_weights \
    --lambda_angle 0.01 \
    --lambda_constraint 0.01 \
    --checkpoint "$CHECKPOINT_DIR" \
    --from_scratch

echo ""
echo "=========================================="
echo "Training complete!"
echo "Best checkpoint: $CHECKPOINT_DIR/ckpt_best_rehab.pth.tar"
echo "=========================================="
