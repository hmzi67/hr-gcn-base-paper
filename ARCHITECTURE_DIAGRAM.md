# HR-GCN: System Architecture Diagram

## 1. High-Level System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         HR-GCN Pose Estimation System                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌─────────────────┐    ┌──────────────┐               │
│  │   Input: 2D  │    │ HR-GCN Backbone │    │  Output: 3D  │               │
│  │  Keypoints   │───▶│  (Graph Models)  │───▶│ Coordinates  │               │
│  │ (133 joints) │    │  + ROM Angles    │    │ + ROM Angles │               │
│  └──────────────┘    └─────────────────┘    └──────────────┘               │
│       (T,133,2)          GCN Layers           (T,133,3)+(T,12)              │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │           Two Main Pipelines: H3WB (original) + Rehab (GCADA)      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Complete Data & Control Flow

### 2.1 H3WB Training Pipeline (HRNet_GCN_WB.py)

```
User Command: python HRNet_GCN_WB.py --model 1 --gcn dc_preagg --epochs 40

    ↓

┌─────────────────────────────────────────────────────────────────┐
│                      parse_args()                               │
│  Flags: --model, --gcn, --epochs, --batch_size, --lr, etc.    │
└──────────────────┬──────────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────────────────┐
│               Load Configuration                                │
│  cfg.merge_from_file('w32_adam_lr1e-3.yaml')                   │
│  Sets: channels, stages, layers, etc.                          │
└──────────────────┬──────────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────────────────┐
│              Load H3WB Dataset                                   │
│  data/h3wb_train.npz  (S1, S5, S6, S7)                          │
│  data/h3wb_test.npz   (S8)                                      │
│  Human3WBDataset(dataset_name, data_path, subjects)            │
└──────────────────┬──────────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────────────────┐
│          Data Preprocessing (common/data_utils.py)              │
│                                                                 │
│  ┌─ 3D Data ─────────────────────┐                             │
│  │ read_3d_data():                │                             │
│  │ • Root-center by hip (avg)     │                             │
│  │ • Convert mm → meters          │                             │
│  │ Output: (N, 133, 3)            │                             │
│  └────────────────────────────────┘                             │
│                                                                 │
│  ┌─ 2D Data ─────────────────────┐                             │
│  │ create_2d_data():              │                             │
│  │ • Normalize img coords         │                             │
│  │ • Root-center at hip center    │                             │
│  │ Output: (N, 133, 2)            │                             │
│  └────────────────────────────────┘                             │
│                                                                 │
│  ┌─ Graph Adjacency ──────────────┐                             │
│  │ adj_mx_from_skeleton():         │                             │
│  │ • Build skeleton edges          │                             │
│  │ Output: (133, 133) adj matrix   │                             │
│  └────────────────────────────────┘                             │
└──────────────────┬──────────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────────────────┐
│      Build Train/Val DataLoaders (common/generators.py)         │
│                                                                 │
│  fetch() → TensorDataset → PoseGenerator → DataLoader           │
│  Batch size: 256 (default)                                      │
│  Workers: 24 (default)                                          │
└──────────────────┬──────────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────────────────┐
│      Instantiate Model (models/*)                               │
│                                                                 │
│  --model 1: GraphHRNetMultiBranch  (models/graph_hrnet_mb.py)  │
│  --model 2: GraphResNet             (models/graph_resnet.py)   │
│  --model 3: GraphHRNet              (models/graph_hrnet.py)    │
│  --model 4: GraphSH                 (models/graph_sh.py)       │
│                                                                 │
│  All models:                                                    │
│  Input: (B, 133, 2)                                            │
│  Output: (B, 133, 3) → split into:                            │
│    • body:  [:23]      (23 joints)                             │
│    • face:  [23:91]    (68 joints)                             │
│    • l_hand:[91:112]   (21 joints)                             │
│    • r_hand:[112:]     (21 joints)                             │
└──────────────────┬──────────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────────────────┐
│      Training Loop (Epochs)                                     │
│                                                                 │
│  For each epoch:                                               │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ for batch in train_loader:                                │ │
│  │    pred_3d = model(batch_2d)  # (B, 133, 3)             │ │
│  │                                                           │ │
│  │    # Compute loss per part                              │ │
│  │    loss = 0                                              │ │
│  │    for part in [body, face, l_hand, r_hand]:            │ │
│  │        pred = pred_3d[part_slice]                       │ │
│  │        gt = gt_3d[part_slice]                           │ │
│  │        loss += (1-λ)*MSE(pred,gt) + λ*L1(pred,gt)      │ │
│  │                                                           │ │
│  │    optimizer.zero_grad()                                │ │
│  │    loss.backward()                                       │ │
│  │    nn.utils.clip_grad_norm_(model.parameters(), 1.0)   │ │
│  │    optimizer.step()                                      │ │
│  │                                                           │ │
│  │    # Log batch metrics                                   │ │
│  │    batch_mpjpe = mpjpe(pred_3d, gt_3d)                  │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌─ Evaluation (every epoch) ─────────────────────────────────┐ │
│  │ eval_loss, metrics = evaluate(model, val_loader)        │ │
│  │ • MPJPE (Mean Per Joint Position Error)                 │ │
│  │ • P-MPJPE (Procrustes-aligned MPJPE)                    │ │
│  │ • Sym penalty (left-right symmetry)                     │ │
│  │                                                           │ │
│  │ if eval_loss < best_loss:                               │ │
│  │     best_loss = eval_loss                               │ │
│  │     save_checkpoint('ckpt_best.pth.tar')                │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌─ Learning Rate Decay ──────────────────────────────────────┐ │
│  │ LR *= gamma every lr_decay steps                         │ │
│  │ (default: gamma=0.9, lr_decay=500)                      │ │
│  └───────────────────────────────────────────────────────────┘ │
└──────────────────┬──────────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────────────────┐
│           Checkpoints Saved (checkpoint/)                       │
│                                                                 │
│  checkpoint_<timestamp>/                                        │
│  ├── ckpt_best.pth.tar              (best MPJPE)              │
│  ├── ckpt_<epoch>.pth.tar           (every snapshot)          │
│  ├── w32_adam_lr1e-3.yaml           (config copy)             │
│  ├── train_rehab.log                (metrics)                 │
│  └── opts.json                      (args)                    │
└─────────────────────────────────────────────────────────────────┘
```

---

### 2.2 Inference Pipeline (infer.py & infer_rehab.py)

#### Standard Inference (infer.py)
```
python infer.py --evaluate checkpoint/ckpt_best.pth.tar -cfg config.yaml

    ↓

Load dataset + preprocess (same as training)
    ↓
Load checkpoint weights into model
    ↓
For test sample (hardcoded: S1/Posing[3]):
    • input_2d: (1, 133, 2)
    • output_3d = model(input_2d)  →  (1, 133, 3)
    • Save to 3d_keypoints.pt
    ↓
Output: 3d_keypoints.pt (test predictions)
```

#### Real-Time Rehab Inference (infer_rehab.py)
```
python infer_rehab.py --source 0 --checkpoint checkpoint_rehab/ckpt_best_rehab.pth.tar

    ↓

┌─────────────────────────────────────────────────────────────────┐
│  Input Source (video file, camera, or synthetic)               │
│  • --source 0           (webcam)                               │
│  • --source video.mp4   (video file)                           │
│  • --skip_pose_detector (synthetic zeros for testing)          │
└──────────────────┬──────────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────────────────┐
│  RTMPose 2D Detector (optional, mmpose-based)                   │
│  RGB frame → 2D keypoints (133 joints)                         │
│  Output: (133, 2) per frame                                    │
└──────────────────┬──────────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────────────────┐
│  Normalize & Root-Center                                        │
│  2D keypoints → normalized + hip-centered                      │
└──────────────────┬──────────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────────────────┐
│  HR-GCN 3D Prediction                                           │
│  Input: (1, 133, 2) normalized 2D                              │
│  model(input_2d) → output_3d (1, 133, 3)                       │
└──────────────────┬──────────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────────────────┐
│  One-Euro Filter (Temporal Smoothing)                           │
│  SkeletonFilter().smooth(joints_3d, body_visibility)           │
│  Reduces jitter while preserving fast movements                │
│  Output: smoothed_joints_3d (1, 133, 3)                        │
└──────────────────┬──────────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────────────────┐
│  Clinical ROM Angle Regression (ClinicalAngleHead)             │
│  Input: body_joints_3d (1, 23, 3)                              │
│  Model: Linear(69) → ReLU → Linear(128) → ReLU → Linear(12)   │
│  Output: rom_angles (1, 12) in degrees                         │
│                                                                 │
│  ROM order:                                                     │
│  [0:cerv_pitch, 1:trunk_flex,                                  │
│   2:l_sho_flex, 3:r_sho_flex, 4:l_sho_abd, 5:r_sho_abd,      │
│   6:l_hip, 7:r_hip, 8:l_knee, 9:r_knee, 10:l_ankle, 11:r_ank]│
└──────────────────┬──────────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────────────────┐
│  Anatomical Constraint Clamp                                    │
│  clamp_angles_to_valid_range(rom_angles)                       │
│  Enforce clinical joint limits (hard bounds)                   │
│  Output: clamped_angles (1, 12)                                │
└──────────────────┬──────────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────────────────┐
│  Visualization & Output                                         │
│  • draw_skeleton_on_frame(frame, joints_3d, angles)           │
│  • Overlay skeleton edges on video frame                       │
│  • Display ROM angle dashboard (text overlay)                  │
│  • Save to output video (--save_video) or display (--display) │
└─────────────────────────────────────────────────────────────────┘
```

---

### 2.3 Rehabilitation Training Pipeline (train_rehab.py)

```
python train_rehab.py \
    --pretrained checkpoint/ckpt_best.pth.tar \
    --cfg w32_adam_lr1e-3.yaml \
    --epochs 50 \
    --lambda_angle 0.1 \
    --lambda_constraint 0.05

    ↓

┌─────────────────────────────────────────────────────────────────┐
│  Load Pretrained H3WB Checkpoint (optional)                     │
│  model.load_state_dict(pretrained_ckpt)  (backbone only)       │
│  angle_head: fresh random init                                 │
└──────────────────┬──────────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────────────────┐
│  Load UI-PRMD Dataset                                           │
│  data/ui_prmd_train.npz (S1-S8)                                │
│  data/ui_prmd_test.npz  (S9-S10)                               │
│                                                                 │
│  Schema:                                                        │
│  • poses_2d:     (N, 133, 2)      normalized 2D keypoints      │
│  • poses_3d:     (N, 133, 3)      3D in meters (hip-centered)  │
│  • rom_angles:   (N, 12)          ground truth clinical angles │
│  • subject_ids:  (N,)             0-indexed subject            │
│  • exercise_ids: (N,)             0-indexed exercise          │
│  • frame_ids:    (N,)             frame within file            │
└──────────────────┬──────────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────────────────┐
│  Instantiate ClinicalAngleHead                                  │
│  ClinicalAngleHead(in_features=69, hidden=128)                 │
│  Output: 12 ROM angles                                         │
└──────────────────┬──────────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────────────────┐
│  Setup Losses & Optimizers                                      │
│                                                                 │
│  ┌─ Loss Components ───────────────────────────────────────┐   │
│  │ 1. L_MPJPE (pose MSE+L1)   (standard HR-GCN loss)      │   │
│  │ 2. L_ROM (angle MSE)       (ROM ground truth)          │   │
│  │ 3. L_anatomical (constraint penalty)                   │   │
│  │                                                         │   │
│  │ L_total = L_MPJPE                                      │   │
│  │         + λ_angle    × L_ROM                          │   │
│  │         + λ_constraint × L_anatomical                 │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─ Optimizer ─────────────────────────────────────────────┐   │
│  │ Adam(model.parameters() + angle_head.parameters())      │   │
│  │ lr: 1e-4 (default for fine-tuning)                     │   │
│  │ wd: 1e-4 (weight decay)                                │   │
│  └─────────────────────────────────────────────────────────┘   │
└──────────────────┬──────────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────────────────┐
│  Training Loop (Epochs)                                         │
│                                                                 │
│  For each epoch:                                               │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ for batch in train_loader:                               │ │
│  │    pred_3d = model(batch_2d)              # (B, 133, 3) │ │
│  │    pred_angles = angle_head(pred_3d[:23]) # (B, 12)    │ │
│  │                                                           │ │
│  │    # Loss computation                                    │ │
│  │    loss_pose = mpjpe_loss(pred_3d, gt_3d)              │ │
│  │    loss_rom = F.mse_loss(pred_angles, gt_rom_angles)   │ │
│  │    loss_constraint = AnatomicalConstraintLoss()         │ │
│  │                                                           │ │
│  │    loss = loss_pose                                     │ │
│  │          + lambda_angle * loss_rom                      │ │
│  │          + lambda_constraint * loss_constraint          │ │
│  │                                                           │ │
│  │    optimizer.zero_grad()                                │ │
│  │    loss.backward()                                       │ │
│  │    nn.utils.clip_grad_norm_()                           │ │
│  │    optimizer.step()                                      │ │
│  │                                                           │ │
│  │    # Log metrics                                         │ │
│  │    batch_mae = mae(pred_angles, gt_rom_angles)          │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌─ Validation (every epoch) ─────────────────────────────────┐ │
│  │ val_loss, val_mae = evaluate(model, angle_head, val_ld) │ │
│  │                                                           │ │
│  │ if val_mae < best_mae:                                  │ │
│  │     best_mae = val_mae                                  │ │
│  │     save_checkpoint({                                   │ │
│  │         'state_dict': model.state_dict(),              │ │
│  │         'angle_head_state_dict': angle_head.sd(),      │ │
│  │         'best_mae': val_mae,                           │ │
│  │         'epoch': epoch                                 │ │
│  │     })                                                  │ │
│  └───────────────────────────────────────────────────────────┘ │
└──────────────────┬──────────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────────────────┐
│  Checkpoint Format: checkpoint_rehab_<variant>/               │
│                                                                 │
│  checkpoint_rehab_baseline_v5/   (λ_angle=0, λ_constraint=0)  │
│  ├── ckpt_best.pth.tar                                        │
│  ├── opts.json                                                │
│  ├── log.txt                                                  │
│  └── metrics.npz                                              │
│                                                                 │
│  checkpoint_rehab_gcada_v2/      (λ_angle=0.1, λ_constraint=0)│
│  ├── ckpt_best.pth.tar                                        │
│  └── ...                                                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Model Architecture Details

### 3.1 Backbone Models (Input: 2D Keypoints → Output: 3D Coordinates)

#### Model 1: GraphHRNetMultiBranch (graph_hrnet_multi_branch.py)
```
Input: (B, 133, 2)  [2D keypoints]
    ↓
┌─────────────────────────────────────────────────────────────┐
│  GCN Stem (Multiple ResBlocks)                              │
│  • Processes input features through graph convolutions      │
│  • Maintains multi-resolution structure                    │
│  • GCN type determined by --gcn flag                       │
│  Output: (B, 133, hidden_dim)                              │
└──────────────────┬──────────────────────────────────────────┘
                   ↓
        ┌──────────┴──────────┬──────────────┬─────────────────┐
        ↓                     ↓              ↓                 ↓
  ┌──────────┐         ┌──────────┐  ┌──────────┐       ┌──────────┐
  │  Body    │         │  Face    │  │L-Hand    │       │R-Hand    │
  │ Branch   │         │ Branch   │  │ Branch   │       │ Branch   │
  │ (23 pts) │         │(68 pts)  │  │(21 pts)  │       │(21 pts)  │
  │   ↓      │         │   ↓      │  │   ↓      │       │   ↓      │
  │ ResBlks  │         │ ResBlks  │  │ ResBlks  │       │ ResBlks  │
  │   ↓      │         │   ↓      │  │   ↓      │       │   ↓      │
  │Conv1d→3  │         │Conv1d→3  │  │Conv1d→3  │       │Conv1d→3  │
  └────┬─────┘         └────┬─────┘  └────┬─────┘       └────┬─────┘
       ↓                    ↓             ↓                  ↓
     (23,3)              (68,3)         (21,3)             (21,3)
       └────────────────────┬──────────────┬────────────────┘
                            ↓
                    Output: (B, 133, 3)  [3D coordinates]
```

#### Model 2: GraphResNet (graph_resnet.py)
```
Input: (B, 133, 2)
    ↓
┌────────────────────────────────────┐
│  GCN Stem                          │
│  Input projection                  │
└──────────────┬─────────────────────┘
               ↓
┌────────────────────────────────────┐
│  ResNet-style Blocks (stacked)     │
│  • Basic blocks or bottleneck      │
│  • Each: GConv → BN → ReLU        │
│  • Skip connections                │
└──────────────┬─────────────────────┘
               ↓
┌────────────────────────────────────┐
│  Final projection Conv1d           │
│  Output channels: 3 (x, y, z)      │
└──────────────┬─────────────────────┘
               ↓
        Output: (B, 133, 3)
```

#### Model 3: GraphHRNet (graph_hrnet.py)
```
Input: (B, 133, 2)
    ↓
┌────────────────────────────────────┐
│  Multi-resolution stem             │
│  Different resolution branches     │
└──────────────┬─────────────────────┘
               ↓
┌────────────────────────────────────┐
│  HR transitions                    │
│  Fuse information across resols    │
└──────────────┬─────────────────────┘
               ↓
┌────────────────────────────────────┐
│  Final layer                       │
│  Output projection to 3D           │
└──────────────┬─────────────────────┘
               ↓
        Output: (B, 133, 3)
```

#### Model 4: GraphSH (graph_sh.py)
```
Input: (B, 133, 2)
    ↓
┌────────────────────────────────────┐
│  Stacked Hourglass Modules         │
│  • Encoder (downsampling)          │
│  • Decoder (upsampling)            │
│  • Skip connections                │
│  • Multiple hourglasses in series  │
└──────────────┬─────────────────────┘
               ↓
┌────────────────────────────────────┐
│  Output head                       │
│  Conv1d → 3D coordinates           │
└──────────────┬─────────────────────┘
               ↓
        Output: (B, 133, 3)
```

---

### 3.2 Clinical Angle Head (models/clinical_angle_head.py)

```
Input: Body joints 3D (B, 23, 3)  [Body part from HR-GCN output]
    ↓
┌──────────────────────────────────────┐
│  Reshape: (B, 23, 3) → (B, 69)      │
│  Flatten body joints to feature vec  │
└────────────────┬─────────────────────┘
                 ↓
        ┌────────────────────────┐
        │  Linear(69 → 128)      │
        │  ReLU activation       │
        │  Dropout(0.1)          │
        └────────────┬───────────┘
                     ↓
        ┌────────────────────────┐
        │  Linear(128 → 64)      │
        │  ReLU activation       │
        └────────────┬───────────┘
                     ↓
        ┌────────────────────────┐
        │  Linear(64 → 12)       │
        │  (No activation)        │
        └────────────┬───────────┘
                     ↓
Output: ROM Angles (B, 12)  [in degrees]

ROM Order (JOINT_LIMIT_TENSOR_ORDER):
[0: Cervical Pitch,      1: Trunk Flexion,
 2: L-Shoulder Flexion,  3: R-Shoulder Flexion,
 4: L-Shoulder Abduction,5: R-Shoulder Abduction,
 6: L-Hip,               7: R-Hip,
 8: L-Knee,              9: R-Knee,
10: L-Ankle,            11: R-Ankle]
```

---

### 3.3 Graph Convolution Variants (models/gconv/)

```
All GCN types receive:
    Input: (B, N_joints, in_dim)
    Graph: adjacency matrix (N_joints, N_joints)
    ↓

┌────────────────────────────────────────────────────┐
│  Graph Convolution Operator (--gcn flag)           │
├────────────────────────────────────────────────────┤
│                                                     │
│  dc_vanilla      Decoupled Vanilla GConv           │
│  dc_preagg   ✓✓  Decoupled Pre-Aggregation (BEST)  │
│  dc_postagg      Decoupled Post-Aggregation        │
│  semantic        Channel-wise Semantic GConv       │
│  convst          Conv-style Aggregation            │
│  nosharing       No Weight Sharing across layers   │
│  modulated       Modulated Graph Convolution       │
│                                                     │
└────────────────────────────────────────────────────┘

All output: (B, N_joints, out_dim)

Example: dc_preagg
    (B, N, in_dim)
        ↓
    Aggregate neighbors FIRST:
    agg = A @ X          where A = normalized adjacency
        ↓
    Transform aggregated feature:
    out = W @ agg + b
        ↓
    BatchNorm → ReLU → Dropout
        ↓
    (B, N, out_dim)
```

---

## 4. Data Structure & Semantics

### 4.1 H3WB Dataset Schema

```
data/h3wb_train.npz
├── metadata (dict)
│   ├── action names
│   ├── subject IDs
│   └── camera IDs
│
└── train_data (dict)
    ├── S1 (subject 1)
    │   ├── Posing (action)
    │   │   ├── camera_54138969
    │   │   │   ├── global_3d: (T1, 133, 3)      [mm]
    │   │   │   ├── pose_2d:   (T1, 133, 2)      [pixels]
    │   │   │   ├── camera_3d: (T1, 133, 3)      [mm, camera frame]
    │   │   │   └── frame_id:  (T1,)
    │   │   └── [other cameras...]
    │   │
    │   ├── [other actions...]
    │
    ├── S5, S6, S7 (other train subjects)
    └── [more actions]

data/h3wb_test.npz
└── data (dict)
    ├── S8 (test subject)
    │   └── [same structure]
```

### 4.2 UI-PRMD Dataset Schema

```
data/ui_prmd_train.npz
├── poses_2d:     (N, 133, 2)      Normalized 2D keypoints
├── poses_3d:     (N, 133, 3)      3D coordinates (meters, hip-centered)
├── rom_angles:   (N, 12)          Clinical joint angles (degrees)
├── subject_ids:  (N,)             Subject index [0-7]  (Train: S1-S8)
├── exercise_ids: (N,)             Exercise index [0-9] (m01-m10)
└── frame_ids:    (N,)             Frame index within each video

data/ui_prmd_test.npz
└── [same structure for S9-S10]

Joint limits (anatomical constraints):
ROM_LIMITS = {
    'cerv_pitch':    [-25, 25],      # degrees
    'trunk_flex':    [-20, 60],
    'l_sho_flex':    [0, 170],
    'r_sho_flex':    [0, 170],
    'l_sho_abd':     [-30, 160],
    'r_sho_abd':     [-30, 160],
    'l_hip':         [-20, 120],
    'r_hip':         [-20, 120],
    'l_knee':        [0, 140],
    'r_knee':        [0, 140],
    'l_ankle':       [60, 120],      # 90°=neutral
    'r_ankle':       [60, 120],
}
```

### 4.3 Joint Partition (133 total)

```
┌─────────────────────────────────────────────────────┐
│  Joint Indices & Anatomy                            │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Body:     [0:23]   (23 joints)                    │
│    • Torso, hips, knees, ankles, arms, wrists    │
│                                                     │
│  Face:     [23:91]  (68 joints)                    │
│    • Eyes, nose, mouth, face contours             │
│                                                     │
│  Left Hand:  [91:112] (21 joints)                  │
│    • Wrist + 5 fingers × 4 joints each            │
│                                                     │
│  Right Hand: [112:133](21 joints)                  │
│    • Wrist + 5 fingers × 4 joints each            │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## 5. Loss Functions & Metrics

### 5.1 H3WB Training Losses

```
For each batch:

    ┌─────────────────────────────────────────────────┐
    │  Per-Part Loss (4 parts)                        │
    ├─────────────────────────────────────────────────┤
    │                                                 │
    │  For part in [body, face, l_hand, r_hand]:    │
    │                                                 │
    │  pred_part = pred_3d[part_slice]               │
    │  gt_part = gt_3d[part_slice]                   │
    │                                                 │
    │  mse = MSE(pred_part, gt_part)                 │
    │  l1  = L1(pred_part, gt_part)                  │
    │                                                 │
    │  loss_part = (1 - λ) * mse + λ * l1            │
    │              (λ = 0.2 by default)              │
    │                                                 │
    │  total_loss += loss_part                       │
    │                                                 │
    └─────────────────────────────────────────────────┘

    Output: loss = loss_body + loss_face
                  + loss_l_hand + loss_r_hand
```

### 5.2 Rehabilitation Training Losses

```
┌─────────────────────────────────────────────────────┐
│  L_total = L_pose + λ_angle × L_rom                │
│                    + λ_constraint × L_anatomical   │
│                                                     │
│  --lambda_angle:     0.0 - 1.0 (default: 0.1)    │
│  --lambda_constraint: 0.0 - 1.0 (default: 0.05)  │
├─────────────────────────────────────────────────────┤
│                                                     │
│  L_pose (common.loss):                              │
│    Standard MPJPE loss                             │
│                                                     │
│  L_rom (common.clinical_loss.ClinicalPoseLoss):    │
│    MSE(pred_angles, gt_angles)                     │
│    Dimension: (B, 12) → scalar                     │
│                                                     │
│  L_anatomical (common.anatomical_constraints):     │
│    Soft penalty when angles exceed clinical limits │
│    ∑ max(0, angle - limit_upper)²                  │
│    ∑ max(0, limit_lower - angle)²                  │
│    Dimension: (B, 12) → scalar                     │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### 5.3 Evaluation Metrics

```
┌──────────────────────────────────────────────────┐
│  Common Metrics (common/loss.py)                  │
├──────────────────────────────────────────────────┤
│                                                  │
│  MPJPE (Mean Per-Joint Position Error)           │
│    = mean(||pred_i - gt_i||) over all joints    │
│    Unit: millimeters                            │
│                                                  │
│  P-MPJPE (Procrustes-Aligned MPJPE)             │
│    = MPJPE after optimal rigid alignment        │
│    (scale + rotation + translation)             │
│                                                  │
│  Symmetry Penalty                               │
│    = ||left_joint - mirror(right_joint)||       │
│    Encourages anatomical symmetry               │
│                                                  │
│  Part-wise MPJPE                                │
│    = MPJPE for each body part separately        │
│                                                  │
├──────────────────────────────────────────────────┤
│  ROM Metrics (for rehabilitation)                │
│                                                  │
│  MAE (Mean Absolute Error)                      │
│    = mean(|pred_angle - gt_angle|)              │
│    Unit: degrees                                │
│                                                  │
│  Per-joint ROM MAE                              │
│    = MAE for each of 12 ROM joints              │
│                                                  │
└──────────────────────────────────────────────────┘
```

---

## 6. File Organization & Module Dependencies

### 6.1 Complete Directory Tree

```
/home/genesys/hamza/HR-GCN/
│
├─ HRNet_GCN_WB.py              # Main training script (H3WB)
├─ infer.py                     # Inference on H3WB test set
├─ train_rehab.py               # Rehabilitation fine-tuning
├─ infer_rehab.py               # Real-time rehabilitation inference
├─ infer_live.py                # [Optional live inference variant]
│
├─ CLAUDE.md                    # Project instructions
├─ CODEBASE_ARCHITECTURE_GUIDE.md  # Detailed architecture
├─ ARCHITECTURE_DIAGRAM.md      # This file
├─ README.md                    # Quick start guide
├─ CHECKPOINT_GUIDE.md          # Checkpoint formats
│
├─ w32_adam_lr1e-3.yaml         # Main graph model config
├─ res50_adam_lr1e-3.yaml       # HRNet 2D config
│
├─ data/
│   ├─ h3wb_train.npz              # H3WB training data
│   ├─ h3wb_test.npz               # H3WB test data
│   ├─ prepare_data_h3wb.py        # Not used in current flow
│   ├─ prepare_data_h36m.py        # Legacy
│   └─ prepare_data_2d_h36m_sh.py  # Legacy
│
├─ utils/
│   ├─ prepare_data_h3wb.py        # H3WB dataset class + metadata
│   └─ prepare_data_uiprmd.py      # UI-PRMD preprocessing
│
├─ models/
│   ├─ __init__.py                  # Empty
│   ├─ graph_hrnet_multi_branch.py # Model 1 (best)
│   ├─ graph_resnet.py              # Model 2
│   ├─ graph_hrnet.py               # Model 3
│   ├─ graph_sh.py                  # Model 4
│   ├─ graph_non_local.py           # Non-local attention utility
│   ├─ clinical_angle_head.py       # ROM angle regression head (NEW)
│   │
│   └─ gconv/                       # Graph convolution operators
│       ├─ __init__.py
│       ├─ vanilla_graph_conv.py    # Base GConv
│       ├─ pre_agg_graph_conv.py    # Pre-aggregation (dc_preagg) ✓
│       ├─ post_agg_graph_conv.py   # Post-aggregation
│       ├─ conv_style_graph_conv.py # Conv-style
│       ├─ no_sharing_graph_conv.py # No weight sharing
│       ├─ modulated_gcn_conv.py    # Modulated variant
│       └─ sem_graph_conv.py        # Semantic
│
├─ common/
│   ├─ arguments.py                 # CLI argument parsing utils
│   ├─ data_utils.py                # 2D/3D preprocessing, fetch()
│   ├─ generators.py                # PoseGenerator dataset wrapper
│   ├─ generators_video.py          # VideoPose3D-style generators
│   ├─ graph_utils.py               # Adjacency builders
│   ├─ loss.py                      # MPJPE, P-MPJPE, metrics
│   ├─ camera.py                    # Projection utilities
│   ├─ log.py                       # Logger, checkpointing
│   ├─ utils.py                     # LR decay, grad clipping
│   ├─ skeleton.py                  # Skeleton joint definitions
│   ├─ visualization.py             # Plotting utilities
│   ├─ visualize.py                 # Frame visualization (NEW)
│   ├─ one_euro_filter.py           # Temporal smoothing (NEW)
│   ├─ clinical_loss.py             # Clinical loss functions (NEW)
│   ├─ anatomical_constraints.py    # ROM limits & clamping (NEW)
│   ├─ h36m_dataset.py              # Legacy H36M loader
│   ├─ mocap_dataset.py             # Legacy mocap utils
│   └─ quaternion.py                # Rotation utilities
│
├─ lib/
│   ├─ config/
│   │   ├─ default.py               # YACS config system
│   │   └─ models.py                # Model-specific configs
│   ├─ core/
│   ├─ dataset/
│   ├─ models/
│   ├─ nms/
│   ├─ utils/
│   └─ Makefile                     # Builds NMS extension
│
├─ progress/
│   ├─ __init__.py
│   ├─ bar.py                       # CLI progress bar
│   ├─ counter.py
│   ├─ spinner.py
│   └─ helpers.py
│
├─ checkpoint/                      # H3WB checkpoints
│   ├─ ckpt_best.pth.tar
│   ├─ log.txt
│   └─ opts.json
│
├─ checkpoint_rehab_baseline_v5/    # Rehab baseline (no loss)
├─ checkpoint_rehab_gcada_v2/       # Rehab with GCADA losses
│
├─ experiments/
│   └─ [HRNet 2D COCO/MPII configs] (legacy)
│
└─ .claude/
    └─ [Claude Code settings]
```

### 6.2 Core Import Dependencies

```
HRNet_GCN_WB.py (Training)
├─ models.graph_hrnet_multi_branch (Model 1)
├─ models.graph_resnet             (Model 2)
├─ models.graph_hrnet              (Model 3)
├─ models.graph_sh                 (Model 4)
├─ common.data_utils               (read_3d_data, create_2d_data)
├─ common.generators               (PoseGenerator)
├─ common.graph_utils              (adj_mx_from_skeleton)
├─ common.loss                     (mpjpe, p_mpjpe)
├─ utils.prepare_data_h3wb         (Human3WBDataset)
└─ lib.config                      (cfg)

train_rehab.py (Rehabilitation)
├─ models.graph_hrnet_multi_branch
├─ models.graph_resnet
├─ models.graph_hrnet
├─ models.graph_sh
├─ models.clinical_angle_head      (ROM regression)
├─ common.clinical_loss            (ClinicalPoseLoss)
├─ common.anatomical_constraints   (AnatomicalConstraintLoss)
├─ common.data_utils
├─ common.loss
└─ lib.config

infer_rehab.py (Real-time)
├─ models.clinical_angle_head
├─ common.one_euro_filter          (SkeletonFilter)
├─ common.anatomical_constraints   (clamp_angles_to_valid_range)
├─ common.visualize                (draw_skeleton_on_frame)
└─ mmpose (optional)               (RTMPose detector)
```

---

## 7. Configuration System

### 7.1 YACS Config Hierarchy

```
lib/config/default.py (base defaults)
    ↓
    Merged with --cfg w32_adam_lr1e-3.yaml
    ↓
    ┌─────────────────────────────────────┐
    │  Final Config Object (cfg)          │
    ├─────────────────────────────────────┤
    │                                     │
    │  Network Architecture:              │
    │  • STAGE2_CHANNEL    = 32           │
    │  • STAGE3_CHANNEL    = 32           │
    │  • STAGE4_CHANNEL    = 32           │
    │  • OUTPUT_CHANNEL    = 128          │
    │  • NUM_BLOCKS        = 2            │
    │                                     │
    │  Training:                          │
    │  • LR                 = 0.01        │
    │  • BATCH_SIZE         = 256         │
    │  • EPOCHS             = 40          │
    │  • OPTIMIZER          = adam        │
    │                                     │
    │  Loss:                              │
    │  • WEIGHT_L1          = 0.2         │
    │  • LAMBDA_ANGLE       = 0.0 (rehab) │
    │  • LAMBDA_CONSTRAINT  = 0.0 (rehab) │
    │                                     │
    └─────────────────────────────────────┘
```

---

## 8. Execution Flow Summary

### 8.1 Training (Single Epoch)

```
main(args)
  ├─ Load dataset
  ├─ Build model
  ├─ Setup losses & optimizer
  │
  ├─ FOR epoch in epochs:
  │   ├─ FOR batch in train_loader:
  │   │   ├─ Forward: pred = model(batch_2d)
  │   │   ├─ Compute loss per part
  │   │   ├─ Backward: loss.backward()
  │   │   └─ Update: optimizer.step()
  │   │
  │   ├─ Evaluate on val_loader
  │   │
  │   ├─ Log metrics
  │   │
  │   └─ IF best_loss: save_checkpoint()
  │
  └─ FINAL: save_checkpoint('ckpt_last')
```

### 8.2 Inference (Single Frame)

```
main(args)
  ├─ Load dataset & preprocess
  ├─ Load checkpoint
  ├─ Load model
  │
  ├─ Sample = dataset[idx]
  ├─ pred_3d = model(sample_2d)
  ├─ Save pred_3d
  │
  └─ END
```

### 8.3 Real-Time Inference (Per Frame)

```
main(args)
  ├─ Load model + checkpoint
  ├─ Open video source
  │
  ├─ WHILE True:
  │   ├─ Read frame
  │   ├─ Detect 2D keypoints (RTMPose)
  │   ├─ Normalize 2D
  │   ├─ Predict 3D: pred_3d = model(normalized_2d)
  │   ├─ Smooth: smoothed = filter.smooth(pred_3d)
  │   ├─ ROM angles: angles = angle_head(smoothed[:23])
  │   ├─ Clamp: angles = clamp_angles(angles)
  │   ├─ Draw skeleton on frame
  │   ├─ Display / save frame
  │   │
  │   └─ IF escape key: break
  │
  └─ Release video resources
```

---

## 9. Key Architectural Decisions

### 9.1 Design Patterns Used

| Pattern | Location | Purpose |
|---------|----------|---------|
| **Model Registry** | `HRNet_GCN_WB.py:300-320` | Select model by --model flag |
| **GCN Dispatcher** | `models/graph_hrnet_multi_branch.py:20-46` | Select GCN type by --gcn flag |
| **Dataset Adapter** | `common/generators.py` | PoseGenerator wraps raw tensors |
| **Checkpoint Manager** | `common/log.py` | Save/load model + optimizer state |
| **Config Merger** | `lib/config/default.py` | YACS config inheritance |

### 9.2 Critical Hardcoding

| Item | Location | Issue | Fix |
|------|----------|-------|-----|
| GPU Device | `HRNet_GCN_WB.py:150` | `device = torch.device('cuda:0')` | Modify to support CPU or multi-GPU |
| Skeleton Adjacency | `common/graph_utils.py` | Hard-coded H3WB skeleton edges | Same for H36M/COCO via abstraction |
| Joint Limits | `common/anatomical_constraints.py:10-25` | Clinical limits (hard-coded) | Load from config file |

### 9.3 Extension Points

| Feature | File | Entry |
|---------|------|-------|
| **New Model** | `models/graph_*.py` | Inherit from `nn.Module`, register in HRNet_GCN_WB.py:310 |
| **New GCN Type** | `models/gconv/*.py` | Implement `forward(x)`, register in graph_hrnet_multi_branch.py:25 |
| **New Loss** | `common/clinical_loss.py` | Inherit from `nn.Module`, add to `train_rehab.py:loss_total` |
| **New Dataset** | `utils/prepare_data_*.py` | Create dataset class with `__getitem__`, integrate in main script |

---

## 10. Performance Characteristics

```
┌────────────────────────────────────────────────────────┐
│  Typical Training Performance (H3WB, GPU RTX 3090)     │
├────────────────────────────────────────────────────────┤
│                                                        │
│  Batch Time:          ~0.1-0.2 seconds                │
│  Epoch Time:          ~2-3 minutes (256-batch)         │
│  Checkpoint Save:     ~30 seconds                      │
│  Total Training:      ~40 epochs × 3 min = 2 hours   │
│                                                        │
│  Inference (single):  ~10-50 ms per frame              │
│  Real-time (30 FPS):  ~33 ms budget per frame          │
│                                                        │
│  Memory (Training):   ~8-12 GB GPU memory              │
│  Memory (Inference):  ~2-3 GB GPU memory               │
│                                                        │
└────────────────────────────────────────────────────────┘
```

---

## 11. Future Architecture Extensions

### Possible Enhancements

1. **Multi-GPU Support**: Distributed training via `nn.DataParallel` or `DistributedDataParallel`
2. **Model Export**: ONNX / TorchScript for deployment
3. **Lightweight Models**: Pruning / quantization for mobile inference
4. **Temporal Modeling**: 3D CNN or Transformer for sequence processing
5. **Uncertainty Estimation**: Bayesian networks for confidence intervals
6. **Multi-task Learning**: Joint pose + action recognition + ROM prediction

---

**Document Version**: 2026-05-04  
**Last Updated**: 2026-05-04  
**Author**: Claude Code Architecture Analysis
