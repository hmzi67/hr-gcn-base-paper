# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements **HR-GCN variants for 3D whole-body pose estimation** from 2D keypoints using the H3WB dataset, extended with a **GCADA clinical rehabilitation pipeline** for UI-PRMD exercise assessment.

- **Input**: 2D pose keypoints (133 joints)
- **Output**: 3D pose coordinates (133 joints) + 8 clinical ROM angles
- **Task**: Estimate full 3D body, face, hand poses from 2D monocular input; regress rehabilitation joint angles
- **Architecture**: Multi-resolution Graph Convolutional Networks built on PyTorch

For detailed architecture and data flow, see [CODEBASE_ARCHITECTURE_GUIDE.md](CODEBASE_ARCHITECTURE_GUIDE.md).

## Environment Setup

**Python Version**: 3.8+, PyTorch 1.13.1+, CUDA-capable GPU (tested on RTX 3090)

**Install dependencies**:
```bash
pip install -r requirements.txt
pip install cdflib scipy mmengine opencv-python
```

**Optional**: Build NMS (non-maximum suppression) native extension for HRNet 2D code:
```bash
cd lib && make && cd ..
```

## Common Commands

### Training (H3WB — original)
```bash
# Standard variant (Decoupled Pre-Agg GCN with HR-Net multi-branch model)
python HRNet_GCN_WB.py --gcn dc_preagg --model 1

# Other GCN variants: dc_vanilla, semantic, dc_postagg, convst, nosharing, modulated
python HRNet_GCN_WB.py --gcn <type> --model <1-4>
```

### Evaluation (from checkpoint)
```bash
python HRNet_GCN_WB.py --model 1 --gcn dc_preagg --evaluate checkpoint/ckpt_best.pth.tar -cfg checkpoint/w32_adam_lr1e-3.yaml
```

### Inference (H3WB)
```bash
python infer.py --evaluate checkpoint/ckpt_best.pth.tar -cfg checkpoint/w32_adam_lr1e-3.yaml
```

For more options, run:
```bash
python HRNet_GCN_WB.py --help
```

See [README.md](README.md) for additional training variants and documentation.

## Architecture at a Glance

**Models** (selected via `--model` flag, IDs 1–4):
1. **HRGCN** (`graph_hrnet_multi_branch.py`): Multi-resolution with part-specific branches
2. **GraphResNet** (`graph_resnet.py`): ResNet-style stacked blocks
3. **HRGCN*** (`graph_hrnet.py`): Single-head HR-Net backbone
4. **GraphSH** (`graph_sh.py`): Stack-Hourglass with pool/unpool

**Graph Convolutions** (selected via `--gcn` flag):
- `dc_vanilla`, `dc_preagg`, `dc_postagg`: Decoupled variants
- `semantic`: Channel-wise semantic aggregation
- `convst`: Convolution-style aggregation
- `nosharing`: No weight sharing across layers
- `modulated`: Modulated graph convolution

**Data Pipeline**:
- 2D keypoints (scaled and root-centered)
- Graph adjacency from skeleton structure
- Per-part loss: MSE + λ·L1 (body, face, left hand, right hand)
- Evaluation metrics: MPJPE, P-MPJPE, symmetric penalties

See [CODEBASE_ARCHITECTURE_GUIDE.md](CODEBASE_ARCHITECTURE_GUIDE.md) for full data flow, dataset schema, and folder structure.

## Key Implementation Notes

**Critical Constraints**:
- **GPU hardcoded to `cuda:0`** in HRNet_GCN_WB.py, infer.py, train_rehab.py, and infer_rehab.py; CPU-only environments will fail without modification
- **Mutually exclusive flags**: `--resume` and `--evaluate` cannot be set together
- **Model IDs**: Code supports `1–4`; README example references model `5` which is not wired in the current code

**Known Issues (resolved)**:
- `infer.py` previously imported missing `models.graph_hrnet_multi_branch_58`; **fixed** — now imports `models.graph_hrnet_multi_branch`

**Dataset**:
- Requires `data/h3wb_train.npz` and `data/h3wb_test.npz`
- Train subjects: S1, S5, S6, S7
- Test subjects: S8
- Configuration files: `w32_adam_lr1e-3.yaml` (graph models), `res50_adam_lr1e-3.yaml` (HRNet/ResNet 2D)

---

## Rehabilitation Extension (GCADA)

Extends HR-GCN for clinical rehabilitation assessment on the UI-PRMD dataset.

### New Files

| File | Purpose |
|------|---------|
| `utils/prepare_data_uiprmd.py` | UI-PRMD dataset preprocessor (txt→NPZ) |
| `common/anatomical_constraints.py` | Joint ROM limit enforcement — Novel #1 |
| `common/clinical_loss.py` | Clinical angle supervision loss — Novel #2 |
| `common/one_euro_filter.py` | Temporal smoothing for real-time inference |
| `common/visualize.py` | Skeleton + ROM angle overlay on video frames |
| `models/clinical_angle_head.py` | ROM angle regression MLP head |
| `train_rehab.py` | Fine-tuning pipeline on UI-PRMD |
| `infer_rehab.py` | Real-time end-to-end inference pipeline |

### New Commands

```bash
# Step 1: preprocess UI-PRMD (data/UI-PRMD/raw/ must contain Vicon .txt files)
python utils/prepare_data_uiprmd.py \
  --data_dir data/UI-PRMD/raw \
  --output_dir data/

# Step 2: fine-tune on UI-PRMD (full novel loss)
python train_rehab.py \
  --pretrained checkpoint/ckpt_best.pth.tar \
  --cfg w32_adam_lr1e-3.yaml \
  --epochs 50 \
  --lambda_angle 0.1 \
  --lambda_constraint 0.05

# Baseline (position loss only, no novel contributions):
python train_rehab.py \
  --pretrained checkpoint/ckpt_best.pth.tar \
  --cfg w32_adam_lr1e-3.yaml \
  --epochs 50 \
  --lambda_angle 0.0 \
  --lambda_constraint 0.0

# Step 3: real-time inference with RTMPose 2D detector
python infer_rehab.py \
  --source 0 \
  --checkpoint checkpoint_rehab/ckpt_best_rehab.pth.tar \
  --cfg w32_adam_lr1e-3.yaml \
  --rtmpose_config <path_to_rtmpose_config> \
  --rtmpose_checkpoint <path_to_rtmpose_checkpoint>

# Real-time inference (skip 2D detector — for testing pipeline without RTMPose)
python infer_rehab.py \
  --source video.mp4 \
  --checkpoint checkpoint_rehab/ckpt_best_rehab.pth.tar \
  --cfg w32_adam_lr1e-3.yaml \
  --skip_pose_detector
```

### UI-PRMD Dataset Format

- Location: `data/UI-PRMD/raw/Movements/Vicon/Positions/` — plain `.txt` files
- Naming: `m{exercise:02d}_s{subject:02d}_positions.txt`
- Vicon positions: `(T, 39, 3)` in mm; Vicon angles: `(T, 39, 3)` in degrees
- 10 exercises (m01–m10), 10 subjects (s01–s10); subject 3 exercise 3 missing
- **Subject-level split**: train s01–s08, test s09–s10

Preprocessed NPZ schema:
```
poses_2d:     (N, 133, 2)   — normalized 2D keypoints (orthographic projection)
poses_3d:     (N, 133, 3)   — 3D in meters, hip-centred; face+hands zero-padded
rom_angles:   (N, 8)        — geometric ROM angles in degrees
subject_ids:  (N,)          — 0-indexed subject (0–9)
exercise_ids: (N,)          — 0-indexed exercise (0–9)
frame_ids:    (N,)          — frame index within each file
```

### Novel Contributions (Thesis)

| ID | File | Description |
|----|------|-------------|
| **N1** | `common/anatomical_constraints.py` | `AnatomicalConstraintLoss`: soft quadratic penalty when predicted angles violate clinical ROM limits. Hard clamp at inference via `clamp_angles_to_valid_range`. No prior whole-body pose method enforces rehab-specific limits in the loss. |
| **N2** | `common/clinical_loss.py` + `models/clinical_angle_head.py` | `ClinicalPoseLoss`: L_total = L_MPJPE + λ_angle × L_ROM + λ_constraint × L_anatomical. `ClinicalAngleHead`: lightweight MLP (69→128→64→8) regressing ROM degrees directly. HR-GCN baseline uses only L_MPJPE. |

**Thesis experiment**:
- Run baseline (`--lambda_angle 0.0 --lambda_constraint 0.0`) → record Mean ROM MAE
- Run full GCADA (`--lambda_angle 0.1 --lambda_constraint 0.05`) → record Mean ROM MAE
- Improvement in Mean ROM MAE (degrees) is the primary thesis result

### Rehab Checkpoint Format

```python
torch.load('checkpoint_rehab/ckpt_best_rehab.pth.tar') == {
    'epoch':                 int,
    'state_dict':            dict,   # HR-GCN backbone weights
    'angle_head_state_dict': dict,   # ClinicalAngleHead weights
    'optimizer':             dict,
    'best_mae':              float,  # Mean ROM MAE (degrees)
    'args':                  dict,
}
```

## Change Guidance for Agents

**Minimal & Local Edits**:
- Keep changes focused; avoid broad refactors unless explicitly requested
- Preserve checkpoint compatibility when modifying training loops (state_dict structure, naming conventions)
- All new rehab code must stay behind new CLI flags — never break H3WB training

**When Adding or Renaming Models/GCN Options**:
- Update CLI argument parsing in `HRNet_GCN_WB.py` (model branching and GCN selection)
- Mirror changes in `train_rehab.py` and `infer_rehab.py` (both use the same `build_backbone` pattern)
- Update documentation: [README.md](README.md), this file, and [AGENTS.md](AGENTS.md) with command examples

**Testing & Validation**:
- No dedicated test suite; validate changes with focused smoke runs
- H3WB smoke: `python HRNet_GCN_WB.py --gcn dc_preagg --model 1 --epochs 1`
- Rehab smoke: `python train_rehab.py --pretrained <ckpt> --cfg w32_adam_lr1e-3.yaml --epochs 1 --batch_size 16 --freeze_backbone`

**File Organization**:
- Core models: `models/graph_*.py`
- GCN layers: `models/gconv/*.py` (one file per variant)
- Training pipeline: `HRNet_GCN_WB.py`, `common/data_utils.py`, `common/generators.py`
- Rehab training: `train_rehab.py`, `common/clinical_loss.py`, `models/clinical_angle_head.py`
- Data loading: `utils/prepare_data_h3wb.py`, `utils/prepare_data_uiprmd.py`
- Config system: `lib/config/default.py` (YACS), `.yaml` files

## References

- **Detailed Architecture**: [CODEBASE_ARCHITECTURE_GUIDE.md](CODEBASE_ARCHITECTURE_GUIDE.md)
- **Quick Start**: [AGENTS.md](AGENTS.md)
- **Full Usage & Setup**: [README.md](README.md)
- **Dataset (H3WB)**: https://github.com/wholebody3d/wholebody3d
- **Dataset (UI-PRMD)**: https://www.webpages.uidaho.edu/ui-prmd/
- **Upstream Projects**: VideoPose3D, Semantic GCN, Modulated-GCN, GraphSH (see README acknowledgments)
