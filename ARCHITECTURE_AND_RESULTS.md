# HR-GCN / GCADA: Architecture and Experimental Results

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Input / Output Specification](#2-input--output-specification)
3. [Architecture: H3WB Backbone (HR-GCN)](#3-architecture-h3wb-backbone-hr-gcn)
4. [Architecture: GCADA Rehabilitation Extension](#4-architecture-gcada-rehabilitation-extension)
5. [Loss Functions](#5-loss-functions)
6. [Data Pipeline](#6-data-pipeline)
7. [Dataset Details](#7-dataset-details)
8. [Experimental Results](#8-experimental-results)
9. [Comparison with Prior Work](#9-comparison-with-prior-work)
10. [Data Analysis Findings](#10-data-analysis-findings)

---

## 1. System Overview

The system is a two-stage clinical rehabilitation assessment pipeline:

```
2D keypoints (133 joints)
        │
        ▼
┌───────────────────┐
│   HR-GCN Backbone │  ← Pre-trained on H3WB; fine-tuned on UI-PRMD
│  (graph_hrnet_    │
│  multi_branch.py) │
└────────┬──────────┘
         │  3D pose  (N, 133, 3)
         ▼
┌───────────────────┐
│ ClinicalAngleHead │  ← Trained from scratch on UI-PRMD
│  69-dim features  │
│  → 12 ROM angles  │
└────────┬──────────┘
         │
         ▼
  ROM angles (N, 12) + 3D body pose
```

**Stage 1 — H3WB backbone**: Lifts 133-joint 2D keypoints to 3D coordinates using multi-resolution graph convolutions. Pre-trained on the H3WB dataset (RGB-camera ground truth, 4 subjects).

**Stage 2 — ClinicalAngleHead**: Takes a 69-dimensional geometric feature vector derived from the predicted 3D joints and regresses 12 clinical ROM angles simultaneously. Supervised with `ClinicalPoseLoss` = MPJPE + λ_angle × ROM_MAE + λ_constraint × AnatomicalConstraintLoss.

---

## 2. Input / Output Specification

| Item | Shape | Description |
|------|-------|-------------|
| Input 2D | `(N, 133, 2)` | Normalized keypoints; root-centered at hip midpoint |
| Output 3D | `(N, 133, 3)` | 3D coordinates in metres, hip-centred |
| ROM angles | `(N, 12)` | Geometric angles in degrees |

**Joint partition (133 total):**

| Body part | Joint indices | Count |
|-----------|---------------|-------|
| Body | 0–22 | 23 |
| Face | 23–90 | 68 |
| Left hand | 91–111 | 21 |
| Right hand | 112–132 | 21 |

**12 ROM angles (ordered):**

| Index | Joint | Convention |
|-------|-------|-----------|
| 0 | Cervical Pitch | 0° = neutral |
| 1 | Trunk Flexion | 0° = upright |
| 2 | L Shoulder Flex | 0° = arm at side |
| 3 | R Shoulder Flex | 0° = arm at side |
| 4 | L Shoulder Abd | 0° = arm at side |
| 5 | R Shoulder Abd | 0° = arm at side |
| 6 | L Hip | geometric angle |
| 7 | R Hip | geometric angle |
| 8 | L Knee | geometric angle |
| 9 | R Knee | geometric angle |
| 10 | L Ankle | 90° = neutral; >90° = dorsiflexion |
| 11 | R Ankle | 90° = neutral; >90° = dorsiflexion |

---

## 3. Architecture: H3WB Backbone (HR-GCN)

### 3.1 Model Variants

Four backbones are available via `--model 1..4`:

| ID | Class | File | Description |
|----|-------|------|-------------|
| 1 | `HRGCN` | `graph_hrnet_multi_branch.py` | **Default.** Multi-resolution HR-Net with 4 part-specific output branches (body/face/L-hand/R-hand). Used in all rehab experiments. |
| 2 | `GraphResNet` | `graph_resnet.py` | ResNet-style stacked bottleneck graph blocks. |
| 3 | `HRGCN*` | `graph_hrnet.py` | Single-head HR-Net backbone, then split. |
| 4 | `GraphSH` | `graph_sh.py` | Stack-Hourglass with graph pool/unpool. |

### 3.2 Graph Convolution Variants

Selectable via `--gcn`. All rehab experiments use `dc_preagg`.

| Flag | File | Mechanism |
|------|------|-----------|
| `dc_vanilla` | `vanilla_graph_conv.py` | Standard decoupled GCN |
| `dc_preagg` | `pre_agg_graph_conv.py` | **Pre-aggregation decoupled** — features aggregated before weight multiplication |
| `dc_postagg` | `post_agg_graph_conv.py` | Post-aggregation decoupled |
| `semantic` | `sem_graph_conv.py` | Channel-wise semantic attention on adjacency |
| `convst` | `conv_style_graph_conv.py` | Convolution-style spatial aggregation |
| `nosharing` | `no_sharing_graph_conv.py` | No weight sharing across adjacency components |
| `modulated` | `modulated_gcn_conv.py` | Learned modulation masks on graph edges |

### 3.3 HRGCN Multi-Branch (Model 1) — Internal Structure

```
Input: (N, 133, 2)  →  embed to (N, 133, C)
         │
         ▼
  ┌─── Stage 2 ────────────────────────────────────┐
  │  Branch 1: full-res  (N, 133, C₁)              │
  │  Branch 2: ½-res     (N,  67, C₂)              │
  │  Exchange features between branches             │
  └─────────────────────────────────────────────────┘
         │
         ▼
  ┌─── Stage 3 ────────────────────────────────────┐
  │  Branch 1: full-res  (N, 133, C₁)              │
  │  Branch 2: ½-res     (N,  67, C₂)              │
  │  Branch 3: ¼-res     (N,  34, C₃)              │
  │  Exchange features across all branches          │
  └─────────────────────────────────────────────────┘
         │  fuse & upsample → (N, 133, C_out)
         ▼
  ┌─── Output Head (4 branches) ───────────────────┐
  │  Body:       Conv1d(C_out, 3) → (N, 23, 3)     │
  │  Face:       Conv1d(C_out, 3) → (N, 68, 3)     │
  │  Left hand:  Conv1d(C_out, 3) → (N, 21, 3)     │
  │  Right hand: Conv1d(C_out, 3) → (N, 21, 3)     │
  └─────────────────────────────────────────────────┘
```

Channel widths are set by `w32_adam_lr1e-3.yaml` (STAGE2, STAGE3, OUTPUT sections).

### 3.4 Graph Adjacency

Built from the H3WB skeleton topology by `common/graph_utils.py::adj_mx_from_skeleton`. A separate adjacency matrix is used per resolution branch (133 / 67 / 34 nodes), constructed by pooling adjacent joints hierarchically.

---

## 4. Architecture: GCADA Rehabilitation Extension

### 4.1 ClinicalAngleHead (`models/clinical_angle_head.py`)

A lightweight MLP that maps geometric body features to 12 ROM angles:

```
3D joints (N, 133, 3)
      │
      │  extract 23 body joints
      ▼
geometric_features()         ← 23 joint vectors → 69 derived scalars
      │                         (segment lengths, dot products, cross products)
      ▼
Linear(69 → 128) + ReLU + Dropout
      ▼
Linear(128 → 64) + ReLU + Dropout
      ▼
Linear(64 → 12)
      ▼
ROM angles (N, 12) in degrees
```

The 69-dimensional input encodes segment orientations, inter-joint angles, and cross-product components — all computed analytically from the 3D skeleton, making the head dataset-agnostic.

An optional `QualityScoreHead` (enabled via `--train_quality_head`) adds a parallel branch that regresses a per-frame exercise quality scalar from the same 69-dimensional features.

### 4.2 AnatomicalConstraintLoss (`common/anatomical_constraints.py`) — Novel Contribution #1

Soft quadratic penalty applied during training when a predicted angle exceeds clinical ROM limits:

```
L_constraint = Σ_j [ max(0, angle_j - max_j)² + max(0, min_j - angle_j)² ]
```

Clinical limits (degrees) per joint:

| Joint | Min | Max |
|-------|-----|-----|
| Cervical Pitch | −45 | 45 |
| Trunk Flex | 0 | 60 |
| Shoulder Flex | 0 | 180 |
| Shoulder Abd | 0 | 180 |
| Hip | 0 | 120 |
| Knee | 0 | 140 |
| Ankle | 60 | 120 |

At inference, `clamp_angles_to_valid_range` applies a hard clamp to the predicted angles before display.

### 4.3 ClinicalPoseLoss (`common/clinical_loss.py`) — Novel Contribution #2

$$L_{total} = L_{MPJPE} + \lambda_{angle} \cdot L_{ROM} + \lambda_{constraint} \cdot L_{anatomical}$$

Where:
- **L_MPJPE**: mean per-joint position error on all 133 joints
- **L_ROM**: MSE between predicted and ground-truth ROM angles (12 values)
- **L_anatomical**: quadratic constraint penalty from N1

**Baseline** (no novel contributions): λ_angle = 0, λ_constraint = 0 → L_total = L_MPJPE only

### 4.4 Optional Temporal Smoothing (`common/one_euro_filter.py`)

One Euro Filter applied per-joint at inference time in `infer_rehab.py` to suppress high-frequency jitter in real-time angle streams.

---

## 5. Loss Functions

### H3WB Training Loss

Per-part weighted loss:

```
L_part = (1 − λ) × MSE(pred, gt) + λ × L1(pred, gt)
L_total = L_body + L_face + L_left_hand + L_right_hand
```

### Rehabilitation Training Loss

```
L_total = MPJPE + λ_angle × MSE(pred_angles, gt_angles) + λ_constraint × L_anatomical
```

**Progressive weighting** (`--progressive_weights`): λ_angle and λ_constraint are scaled by 0.1× in the first third of epochs, 0.5× in the second third, and 1.0× thereafter. This ensures the position loss converges before angle supervision is fully applied.

---

## 6. Data Pipeline

### H3WB Pipeline

```
h3wb_train.npz / h3wb_test.npz
        │
        ▼  Human3WBDataset (utils/prepare_data_h3wb.py)
        │
        ▼  read_3d_data()       → root-center at hip midpoint; mm → metres
        │  create_2d_data()     → normalize to camera-normalized range; root-center
        │
        ▼  adj_mx_from_skeleton()  → adjacency matrix
        │
        ▼  PoseGenerator / DataLoader
        │
        ▼  HRGCN forward → 4 branches → per-part loss
```

### UI-PRMD Pipeline

```
Vicon .txt files  (T, 39, 3) mm
        │
        ▼  prepare_data_uiprmd.py
        │   - orthographic projection → 2D keypoints (133-joint mapped)
        │   - hip-centred 3D in metres (face+hands zero-padded)
        │   - geometric ROM angles computed from 3D skeleton
        │
        ▼  uiprmd_train.npz / uiprmd_test.npz
        │
        ▼  TensorDataset + DataLoader
        │
        ▼  HRGCN forward → 3D pose
        │
        ▼  ClinicalAngleHead → 12 ROM angles
        │
        ▼  ClinicalPoseLoss
```

**Subject-level split**: train = S01–S08 (204,194 frames), test = S09–S10 (49,475 frames).

---

## 7. Dataset Details

### H3WB

| Property | Value |
|----------|-------|
| Joints | 133 (body + face + hands) |
| Train subjects | S1, S5, S6, S7 |
| Test subjects | S8 |
| Cameras | 4 (IDs: 54138969, 55011271, 58860488, 60457274) |
| Format | NPZ with `global_3d (T,133,3)` and `pose_2d (T,133,2)` per action |
| Config | `w32_adam_lr1e-3.yaml` |

### UI-PRMD

| Property | Value |
|----------|-------|
| Exercises | 10 (m01–m10) |
| Subjects | 10 (s01–s10); s03/m03 missing |
| Train frames | 204,194 (S01–S08) |
| Test frames | 49,475 (S09–S10) |
| Sensor | Vicon optical motion capture |
| Sampling | Variable; ~100 Hz |
| Missing joint data | Face, hands zero-padded (Vicon has 39 markers vs 133) |

**Frame distribution per exercise (training split):**

| Exercise | Name | Frames | Category |
|----------|------|--------|----------|
| E1 | Deep Squat | 22,657 | Lower/Trunk |
| E2 | Hurdle Step | 19,677 | Lower/Trunk |
| E3 | Inline Lunge | 19,106 | Lower/Trunk |
| E4 | Side Lunge | 23,403 | Lower/Trunk |
| E5 | Sit to Stand | 24,588 | Lower/Trunk |
| E6 | Str. Leg Raise | 18,076 | Lower/Trunk |
| E7 | Sho. Abduction | 21,211 | Shoulder |
| E8 | Sho. Extension | 18,576 | Shoulder |
| E9 | Sho. Int. Rot. | 19,504 | Shoulder |
| E10 | Sho. Scaption | 17,396 | Shoulder |
| — | **Total** | **204,194** | — |

Lower/trunk: 62.4% of frames. Shoulder: 37.6%.

---

## 8. Experimental Results

### 8.1 Summary Table

| Experiment | Config | Mean ROM MAE (°) | Body MPJPE (mm) | Best Epoch |
|-----------|--------|-----------------|-----------------|-----------|
| Baseline — from scratch | λ=0, no pretrained | 9.21 | 183.86 | 77 |
| Clinical Loss v1 — from scratch | λ_angle=0.01, λ_constraint=0.01 | 5.51 | 152.88 | — |
| Clinical Loss v2 — from scratch | λ_angle=0.01, λ_constraint=0.01 (v2) | 6.09 | 153.56 | — |
| **Baseline — fine-tuned** (H3WB) | λ=0, pretrained | **4.59** | **67.65** | 50 |
| **GCADA — fine-tuned** (H3WB) | λ_angle=0.1, λ_constraint=0.05, pretrained | **4.22** | 85.98 | — |

**Key thesis result**: Fine-tuned GCADA (ClinicalPoseLoss + AnatomicalConstraintLoss) reduces Mean ROM MAE from **4.59° → 4.22°** (−8.1%) relative to the fine-tuned baseline.  
Trade-off: MPJPE increases from 67.65 mm → 85.98 mm (+27%) because the angle supervision partially redirects the model from minimising joint position error.

### 8.2 Baseline (Fine-Tuned, No Clinical Loss) — Full Results

**Config**: pretrained from H3WB (`checkpoint_rehab_baseline_v8`), λ_angle=0, λ_constraint=0, lr=5e-4, batch_size=64, backbone_lr_factor=0.5, epochs=100, best at epoch 50.

**Training convergence:**

| Epoch | MPJPE (mm) | Mean ROM MAE (°) |
|-------|-----------|-----------------|
| 1 | 814.4 | 10.42 |
| 10 | 73.4 | 7.17 |
| 20 | 67.6 | 5.30 |
| 30 | 69.2 | 5.06 |
| 40 | 68.5 | 4.78 |
| **50** | **67.7** | **4.59** ← best |

**Per-exercise ROM MAE:**

| Exercise | Name | ROM MAE (°) | MPJPE (mm) |
|----------|------|------------|-----------|
| E1 | Deep Squat | 4.21 | 56.86 |
| E2 | Hurdle Step | 5.16 | 55.51 |
| E3 | Inline Lunge | 5.73 | 46.88 |
| E4 | Side Lunge | 4.97 | 72.30 |
| E5 | Sit to Stand | 5.11 | 50.81 |
| E6 | Str. Leg Raise | 4.22 | 76.85 |
| E7 | Sho. Abduction | 3.35 | 76.17 |
| E8 | Sho. Extension | **6.47** | **109.11** |
| E9 | Sho. Int-Ext Rot | 2.97 | 73.90 |
| E10 | Sho. Scaption | 3.94 | 67.43 |
| **Average** | — | **4.59** | **67.65** |

E8 (Shoulder Extension) is the hardest exercise: highest ROM MAE and MPJPE.

**Per-joint ROM MAE:**

| Joint | MAE (°) | Clinical threshold (<5°) |
|-------|---------|--------------------------|
| Cervical Pitch | 3.13 | PASS |
| Trunk Flexion | 1.68 | PASS |
| L Shoulder Flex | 4.84 | PASS |
| R Shoulder Flex | 7.33 | **FAIL** |
| L Shoulder Abd | 4.75 | PASS |
| R Shoulder Abd | 9.34 | **FAIL** |
| L Hip | 2.58 | PASS |
| R Hip | 3.13 | PASS |
| L Knee | 3.81 | PASS |
| R Knee | 3.77 | PASS |
| L Ankle | 4.36 | PASS |
| R Ankle | 6.35 | **FAIL** |
| **Mean** | **4.59** | **9/12 pass** |

Right shoulder joints systematically exceed the clinical threshold due to large R–L asymmetry in the dataset (R Sho Abd mean = 104.2° vs L = 56.3°).

### 8.3 GCADA (Fine-Tuned with Clinical Loss) — Full Results

**Config**: pretrained from H3WB, λ_angle=0.1, λ_constraint=0.05, lr=5e-4, batch_size=64, backbone_lr_factor=0.5, epochs=50.

**Per-exercise ROM MAE:**

| Exercise | Name | ROM MAE (°) | MPJPE (mm) |
|----------|------|------------|-----------|
| E1 | Deep Squat | 3.74 | 65.98 |
| E2 | Hurdle Step | 4.32 | 70.79 |
| E3 | Inline Lunge | 5.29 | 72.83 |
| E4 | Side Lunge | 5.73 | 88.84 |
| E5 | Sit to Stand | 3.69 | 67.30 |
| E6 | Str. Leg Raise | 3.91 | 99.20 |
| E7 | Sho. Abduction | 3.87 | 87.18 |
| E8 | Sho. Extension | **4.46** | **164.09** |
| E9 | Sho. Int-Ext Rot | 2.99 | 81.78 |
| E10 | Sho. Scaption | 4.17 | 77.35 |
| **Average** | — | **4.22** | **85.98** |

**Per-joint ROM MAE:**

| Joint | MAE (°) | Clinical threshold (<5°) |
|-------|---------|--------------------------|
| Cervical Pitch | 4.06 | PASS |
| Trunk Flexion | 2.24 | PASS |
| L Shoulder Flex | 4.11 | PASS |
| R Shoulder Flex | 4.79 | PASS ← was FAIL |
| L Shoulder Abd | 3.85 | PASS |
| R Shoulder Abd | 5.89 | **FAIL** |
| L Hip | 2.87 | PASS |
| R Hip | 3.84 | PASS |
| L Knee | 2.94 | PASS |
| R Knee | 4.08 | PASS |
| L Ankle | 4.71 | PASS |
| R Ankle | 7.25 | **FAIL** |
| **Mean** | **4.22** | **10/12 pass** |

Clinical loss recovers R Shoulder Flex (7.33° → 4.79°), bringing it under the clinical threshold. R Shoulder Abd and R Ankle remain above 5°.

### 8.4 From-Scratch Experiments

Running from random initialisation (no H3WB pretrained weights) to isolate the effect of the clinical loss on a clean baseline.

**From-Scratch Baseline** (λ=0, `checkpoint_rehab_baseline_scratch`, best epoch 77):

| Metric | Value |
|--------|-------|
| Mean ROM MAE | 9.21° |
| Body MPJPE | 183.86 mm |
| Normalized MAD | 0.0804 |

**From-Scratch Clinical Loss v1** (λ_angle=0.01, λ_constraint=0.01):

| Metric | Value | vs baseline |
|--------|-------|-------------|
| Mean ROM MAE | 5.51° | −40.2% |
| Body MPJPE | 152.88 mm | −16.9% |
| Normalized MAD | 0.0521 | −35.2% |

Clinical supervision alone reduces ROM MAE by 40% even without domain-adapted pretraining.

### 8.5 Normalized MAD — Per-Exercise (Clinical Loss v1 vs Baseline, from scratch)

Normalized MAD = mean(|pred − gt| / clinical_ROM_range), dimensionless (0–1).

| Exercise | Baseline Norm MAD | Clinical v1 Norm MAD | Improvement |
|----------|-------------------|----------------------|-------------|
| E1 | 0.0788 | 0.0511 | −35% |
| E2 | 0.0813 | 0.0525 | −35% |
| E3 | 0.1041 | 0.0571 | −45% |
| E4 | 0.0961 | 0.0772 | −20% |
| E5 | 0.0924 | 0.0516 | −44% |
| E6 | 0.0709 | 0.0447 | −37% |
| E7 | 0.0533 | 0.0400 | −25% |
| E8 | 0.1064 | 0.0626 | −41% |
| E9 | 0.0591 | 0.0365 | −38% |
| E10 | 0.0578 | 0.0458 | −21% |
| **Overall** | **0.0804** | **0.0521** | **−35%** |

---

## 9. Comparison with Prior Work

### Physio2.2M (Rode et al., Sci. Reports 2025)

Physio2.2M is the closest published work: a large-scale RGB-video clinical pose estimation system.

| Method | MPJPE (mm) | Knee MAE (°) | Mean ROM MAE (°) |
|--------|-----------|-------------|-----------------|
| Physio2.2M Best | 72 | 9.3 | — |
| Physio2.2M Worst | 122 | 21.9 | — |
| **GCADA Baseline (ours)** | **67.65** ★ | **3.79** ★ | **4.59** |
| **GCADA Full (ours)** | 85.98 | **3.51** ★ | **4.22** |

★ = GCADA outperforms both Physio2.2M variants on this metric.

**Key findings**:
- GCADA Baseline achieves lower MPJPE than Physio2.2M Best (67.65 vs 72 mm) despite operating on Vicon mocap data, not RGB video.
- Knee MAE (3.79°/3.51°) is 2.5–6× better than Physio2.2M (9.3°/21.9°).
- GCADA Full trades MPJPE (+19 mm) for a modest ROM improvement (−0.37°) when adding clinical supervision.

### Metric Alignment Note

Direct comparison to Kourbane et al. (2025) Normalized MAD = 0.009 is **invalid**:
- Kourbane predicts a scalar quality score (0–1); their MAD is error on that scalar.
- Our Normalized MAD = mean(angle_error / clinical_ROM_range) across 12 joints.
- These measure fundamentally different things. Our 0.0521 ≠ 5.8× worse than 0.009.

---

## 10. Data Analysis Findings

These findings were obtained by running the diagnostic scripts (`check_*.py`, `results/`) on the training and test splits.

### 10.1 Right Shoulder Bias

Right shoulder consistently produces higher errors than left across all experiments:

| Joint | Train Mean | Test Mean | Train→Test Shift |
|-------|-----------|----------|-----------------|
| R Sho Abd | 104.2° | 96.9° | **7.3°** |
| L Sho Abd | 56.3° | 59.2° | 2.9° |
| R Sho Flex | 99.4° | 92.4° | 7.0° |
| L Sho Flex | 61.9° | 65.1° | −3.2° |

The large R–L asymmetry (e.g. E7 Shoulder Abduction: R mean=101.8°, L mean=20.1°, diff=81.7°) means the right shoulder operates over nearly double the angular range, making it harder to regress. Test subjects S09–S10 also show a distribution shift of 7.3° for R Sho Abd vs 2.9° for L, amplifying generalisation difficulty.

### 10.2 Exercise Imbalance

Shoulder exercises represent only 37.6% of training frames vs 62.4% for lower/trunk. This contributes to higher shoulder angle errors and is one root cause of E8 being the hardest exercise.

### 10.3 E8 Difficulty (Shoulder Extension)

E8 consistently produces the highest MPJPE (109–164 mm depending on run) and ROM MAE across all experiments. Causes:
1. Arm moves behind the body → significant self-occlusion in 2D keypoints
2. R shoulder 2D keypoint x-std (0.104) > L (0.094), suggesting more occlusion on the right
3. Small shoulder extension ROM range in the dataset means even moderate errors dominate MAE

### 10.4 Clinical Normalization Ranges

The dataset-observed ROM for some joints is narrower than the full anatomical range:
- Cervical Pitch: observed ≈ 45°, clinical ≈ 120° → using clinical range reduces Normalized MAD by 19%
- Ankle: observed ≈ 60°, full dorsi/plantarflexion ≈ 90°

Recommendation: report raw degree MAE rather than Normalized MAD to avoid range-choice sensitivity.

---

*Document generated from experimental logs and CSV outputs in `results/`, `results_comparison/`, and `results_comparison_novel/`.*
