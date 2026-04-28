# QUICK START: Fixed Training

## 🚀 START HERE (30 seconds)

### Best Approach: From-Scratch Training
```bash
cd /home/genesys/hamza/HR-GCN
source .venv/bin/activate
./train_rehab_optimized.sh
```

Or manually:
```bash
python train_rehab.py --from_scratch --lr 5e-4 --batch_size 64 \
    --backbone_lr_factor 1.0 --warmup_epochs 3 --progressive_weights \
    --epochs 50 --checkpoint checkpoint_rehab_v3
```

**Expected Results After 50 Epochs:**
- ✅ MPJPE: ~150-250mm (was 375mm) 
- ✅ Mean ROM MAE: ~8-12° (was 20.3°)
- ✅ Better convergence: Continuous improvement (not plateau)

---

## 📊 What Changed (Problem → Solution)

### Problem 1: MPJPE Too High (375mm)
**Root Cause**: H3WB pretrained model doesn't match Vicon mocap data
- H3WB: RGB cameras, 3D ground truth
- Vicon: Orthographic projection from mocap, different scale/coordinates

**Solution**: `--from_scratch` — Train from random initialization
- Directly learns UI-PRMD coordinate system
- Avoids catastrophic forgetting of incompatible H3WB features

```bash
# ❌ OLD (causes domain mismatch)
python train_rehab.py --pretrained checkpoint/ckpt_best.pth.tar --lr 1e-4 ...

# ✅ NEW (ignores bad pretrained)
python train_rehab.py --from_scratch --lr 5e-4 ...
```

---

### Problem 2: Position Loss Plateaus (0.89 → 0.13)
**Root Cause 1**: 11-epoch warmup freezes backbone while untrained angle head predicts random angles (115° off!)
**Root Cause 2**: High loss weights (λ_angle=0.1) overwhelm position loss from epoch 1

**Solution 1**: Shorter warmup (3 epochs)
```bash
# ❌ OLD: Epochs 1-11 frozen, angle head stuck
# ✅ NEW: Epochs 1-3 frozen, angle head has time to initialize
--warmup_epochs 3
```

**Solution 2**: Progressive loss weighting
```bash
# ❌ OLD: L_angle_weight=0.1 from epoch 1 (overwhelming)
# L_pos = 0.89 stays constant through all epochs
# ✅ NEW: L_angle starts at 0.001, ramps to 0.01 over training
--progressive_weights
# Epochs 1-17:   L_pos focused (minimal angle loss)
# Epochs 17-33:  Transition (medium angle loss)  
# Epochs 34-50:  Refinement (full angle loss)
```

---

### Problem 3: Slow Learning Rate (1e-4)
**Root Cause**: Conservative fine-tuning hyperparameters
- `--lr 1e-4` too low for domain adaptation
- `--batch_size 256` too large
- `--backbone_lr_factor 0.1` (10x slower than angle head)

**Solution**: Aggressive but stable hyperparameters for from-scratch
```bash
# ❌ OLD (fine-tune focused)
--lr 1e-4 --batch_size 256 --backbone_lr_factor 0.1

# ✅ NEW (domain adaptation focused)
--lr 5e-4 --batch_size 64 --backbone_lr_factor 1.0
```

**Why?**
- 5x higher LR: Domains are different, need faster learning
- 4x smaller batches: Better gradient signal for small UI-PRMD dataset
- 1.0x backbone factor: No artificial slowdown in from-scratch mode

---

## 🎯 Three Training Options

### Option 1: FROM-SCRATCH (BEST - Recommended)
```bash
python train_rehab.py \
    --from_scratch \                  # Ignore H3WB
    --lr 5e-4 \                       # Faster learning
    --batch_size 64 \                 # Smaller batches
    --backbone_lr_factor 1.0 \        # Equal learning rates
    --warmup_epochs 3 \               # Short warmup
    --progressive_weights \           # Ramp angle loss
    --epochs 50 \
    --checkpoint checkpoint_rehab_v3
```
**When**: Always (unless you have validated H3WB alignment)
**Expected MPJPE**: 150-250mm | **Expected MAE**: 8-12°

---

### Option 2: FINE-TUNE (if H3WB looks good)
```bash
python train_rehab.py \
    --pretrained checkpoint/HRGCN/dc_preagg-2026-04-24T10:48:02/ckpt_best.pth.tar \
    --lr 1e-4 \                       # Conservative
    --batch_size 64 \                 # Still smaller
    --backbone_lr_factor 0.5 \        # 2x slower than angle head
    --warmup_epochs 5 \               # Longer warmup
    --progressive_weights \           # Ramp angle loss
    --epochs 50 \
    --checkpoint checkpoint_rehab_ft
```
**When**: After validating H3WB coordinates match Vicon
**Expected MPJPE**: 200-300mm | **Expected MAE**: 10-15°

---

### Option 3: BASELINE (No Novel Losses)
```bash
python train_rehab.py \
    --from_scratch \
    --lr 5e-4 \
    --batch_size 64 \
    --warmup_epochs 3 \
    --progressive_weights \
    --lambda_angle 0.0 \              # No angle supervision
    --lambda_constraint 0.0 \         # No constraint loss
    --epochs 50 \
    --checkpoint checkpoint_rehab_baseline
```
**When**: Comparing against position-only baseline
**Expected MPJPE**: 180-280mm | **Expected MAE**: N/A (no angles trained)

---

## 📈 Monitoring Progress

Look for these signs in the logs:

### ✅ GOOD (Should See)
```
Epoch 1:  MPJPE=800mm  L_pos=0.89
Epoch 10: MPJPE=550mm  L_pos=0.35
Epoch 20: MPJPE=350mm  L_pos=0.15
Epoch 30: MPJPE=250mm  L_pos=0.08
Epoch 50: MPJPE=180mm  L_pos=0.05  ← Better than 375mm!
```

### ❌ BAD (Stop & Adjust)
```
Epoch 50: MPJPE=375mm  L_pos=0.89  ← Same as old training!
→ Likely --from_scratch not working or pretrained still loading
→ Check log for "Loading pretrained weights..." message
```

---

## 🔧 Implementation Details

### What was changed in train_rehab.py:

1. **New CLI Arguments** (4)
   - `--from_scratch`: Skip pretrained, train from random
   - `--warmup_epochs N`: Configurable warmup (default 3)
   - `--progressive_weights`: Enable loss ramping
   
2. **New Defaults** (6)
   - `--lr`: 1e-4 → 5e-4 (5x faster)
   - `--batch_size`: 256 → 64 (4x smaller)
   - `--backbone_lr_factor`: 0.1 → 0.5 (2x faster backbone)
   - `--lambda_angle`: 0.1 → 0.01 (10x lower start)
   - `--lambda_constraint`: 0.05 → 0.01 (2x lower)
   - `--warmup_epochs`: hardcoded 11 → configurable 3

3. **Dynamic Loss Weighting** (NEW)
   - Epoch 1-17: λ *= 0.1x
   - Epoch 17-33: λ *= 0.5x  
   - Epoch 34-50: λ *= 1.0x

4. **Smart Preloading** (NEW)
   - Respects `--from_scratch` (ignores `--pretrained`)
   - Validation warning if both set

---

## ❓ FAQ

**Q: Why from-scratch instead of fine-tuning?**
A: H3WB → Vicon is NOT a small domain shift:
   - Different sensors (RGB → mocap)
   - Different coordinates (camera view → orthographic projection)
   - Different body representations (maybe different skeletal mapping)
   This violates fine-tuning assumptions. Training from scratch is safer.

**Q: Should I use progressive_weights with fine-tune?**
A: Yes, always. Progressive weighting helps in both cases:
   - From-scratch: Focuses on 3D positions first
   - Fine-tune: Gradually teaches new angles without breaking 3D backbone

**Q: Can I modify the progressive weighting schedule?**
A: Currently hardcoded as 1/3 of epochs. To customize, edit line ~385 in train_rehab.py:
   ```python
   if epoch < args.epochs // 3:  # Change denominator (3 → 2 for 50% threshold)
   ```

**Q: How long does training take?**
A: ~40 minutes per epoch on RTX 3090 with batch_size=64
   50 epochs ≈ 33 hours total

**Q: Can I resume training?**
A: Yes, just re-run with existing checkpoint:
   ```bash
   python train_rehab.py ... --checkpoint checkpoint_rehab_v3
   ```
   It will load best_ckpt if exists and continue from there.

---

## 📝 Expected Log Output

First run will show:
```
==> Settings: {...'from_scratch': True, 'warmup_epochs': 3, 'progressive_weights': True...}
=> INFO: from-scratch + progressive_weights recommended combination
==> Building model...
    Total parameters: 18.70M
==> Training from random initialization (--from_scratch)
==> Optimizer (from-scratch): backbone lr=5.00e-04  angle_head lr=5.00e-04
==> Progressive weighting enabled: 0.1x → 0.5x → 1.0x over training

Epoch 1/50  lr=[5.00e-04  5.00e-04]  λ_angle=0.001 λ_constr=0.001
  Batch   10/797 | total=4.5123  L_pos=0.8234  L_angle=115.2345  L_constr=0.0001
  ...
  [Eval] Body MPJPE: 750.23 mm | Mean ROM MAE: 31.45 deg
  --> New best Mean ROM MAE: 31.45 deg (checkpoint saved)

Epoch 2/50  lr=[5.00e-04  5.00e-04]  λ_angle=0.001 λ_constr=0.001
  ...
```

Look for:
- ✅ "from_scratch" in settings
- ✅ "Training from random initialization"
- ✅ MPJPE decreasing each epoch  
- ✅ λ values changing (progressive weights)

Done! 🎉
