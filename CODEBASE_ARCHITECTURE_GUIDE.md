# HR-GCN Codebase: Full Data Flow, Architecture, and Folder Structure

This document explains how data moves through the repository, how models are wired, and what every major folder/file is responsible for.

---

## 1) Project Purpose

This repository trains and evaluates **3D whole-body pose estimation** models (133 joints) from **2D keypoints** on the **H3WB dataset**.

- **Input:** 2D keypoints per frame (`133 x 2`)
- **Output:** 3D keypoints per frame (`133 x 3`)
- **Main script:** `HRNet_GCN_WB.py`
- **Inference script:** `infer.py`

---

## 2) End-to-End Data Flow

## 2.1 Training / Evaluation Flow (`HRNet_GCN_WB.py`)

1. Parse CLI args (`parse_args`).
2. Load dataset files:
   - `data/h3wb_train.npz`
   - `data/h3wb_test.npz`
3. Build dataset object: `Human3WBDataset(...)` from `utils/prepare_data_h3wb.py`.
4. Preprocess 3D (`common/data_utils.py::read_3d_data`):
   - Root-center using hips (joint 11 and 12 average)
   - Convert mm to meters
5. Preprocess 2D (`common/data_utils.py::create_2d_data`):
   - Normalize image coordinates to camera-normalized range
   - Root-center with same hip center
6. Build adjacency from skeleton (`common/graph_utils.py::adj_mx_from_skeleton`).
7. Load architecture config (`cfg.merge_from_file(...)`, usually `w32_adam_lr1e-3.yaml`).
8. Instantiate selected model:
   - `--model 1`: `models/graph_hrnet_multi_branch.py`
   - `--model 2`: `models/graph_resnet.py`
   - `--model 3`: `models/graph_hrnet.py`
   - `--model 4`: `models/graph_sh.py`
9. Build train/val tensors with `fetch(...)` then `PoseGenerator(...)` + `DataLoader`.
10. Forward pass returns 4 branches:
    - body `[:23]`, face `[23:91]`, left hand `[91:112]`, right hand `[112:]`
11. Loss = weighted sum of MSE + L1 on each part.
12. Save checkpoints + log metrics (MPJPE family) each epoch.

## 2.2 Inference Flow (`infer.py`)

1. Loads same dataset and preprocessing path.
2. Builds model and loads checkpoint.
3. Runs one sample currently hardcoded as:
   - `keypoints['S1']['Posing'][3]`
4. Saves output tensor to `3d_keypoints.pt`.

> Note: `infer.py` imports `models.graph_hrnet_multi_branch_58`, but this source file is not present in `models/` (only cached `.pyc` exists).

---

## 3) Dataset Structure and Semantics

## 3.1 NPZ Schema

- `data/h3wb_train.npz`:
  - `metadata` (dict)
  - `train_data` (dict)
- `data/h3wb_test.npz`:
  - `data` (dict)

Typical per-action fields:
- `global_3d` -> `(T, 133, 3)`
- `frame_id`
- Camera IDs: `54138969`, `55011271`, `58860488`, `60457274`
  - each camera contains:
    - `camera_3d` -> `(T, 133, 3)`
    - `pose_2d` -> `(T, 133, 2)`
    - `sample_id`

## 3.2 Subject Splits

Defined in `utils/prepare_data_h3wb.py`:
- Train: `S1, S5, S6, S7`
- Test: `S8`

## 3.3 Joint Partition (133 total)

Used consistently in training/eval/model outputs:
- Body: `0:23` (23 joints)
- Face: `23:91` (68 joints)
- Left hand: `91:112` (21 joints)
- Right hand: `112:133` (21 joints)

---

## 4) Model Architecture Organization

All high-level models share a pattern:
- Graph-conv stem from 2D features
- Multi-block graph feature extraction
- Output projection to 3D coordinates via `Conv1d(..., out_channels=3)`
- Return split outputs (body/face/left hand/right hand)

## 4.1 Model Files

- `models/graph_hrnet_multi_branch.py`
  - HR-style multi-resolution graph network
  - Separate part-specific output refinement branches
- `models/graph_hrnet.py`
  - HR-style graph backbone, single output head then split
- `models/graph_resnet.py`
  - Graph-ResNet-style stacked bottlenecks/basic blocks
- `models/graph_sh.py`
  - Graph Stack-Hourglass style with pool/unpool
- `models/graph_non_local.py`
  - Non-local attention block utility (used by GraphSH helper module)

## 4.2 Graph Convolution Variants (`models/gconv/`)

Selectable via `--gcn`:
- `vanilla_graph_conv.py` (`vanilla`, `dc_vanilla`)
- `pre_agg_graph_conv.py` (`preagg`, `dc_preagg`)
- `post_agg_graph_conv.py` (`postagg`, `dc_postagg`)
- `conv_style_graph_conv.py` (`convst`)
- `no_sharing_graph_conv.py` (`nosharing`)
- `modulated_gcn_conv.py` (`modulated`)
- `sem_graph_conv.py` (`semantic`)
- `sem_ch_graph_conv.py` (channel-wise semantic variant, present but not wired in model switch logic)

---

## 5) Training, Losses, and Metrics

## 5.1 Losses (`HRNet_GCN_WB.py`)

Per-part loss:
- `(1 - lambda) * MSE + lambda * L1`

Total loss:
- `loss_body + loss_face + loss_left_hand + loss_right_hand`

Utilities:
- `common/loss.py::mpjpe`, `p_mpjpe`, `sym_penalty`, etc.

## 5.2 Evaluation Metrics

`evaluate(...)` reports:
- Overall MPJPE
- Body MPJPE
- Face MPJPE
- Hand MPJPE
- Face-aligned MPJPE
- Hand-aligned MPJPE

---

## 6) Configuration System

- Base config object: `lib/config/default.py` (`yacs`)
- Model extras templates: `lib/config/models.py`
- Main runtime file typically used for graph models:
  - `w32_adam_lr1e-3.yaml`
    - Controls stage structure (`STAGE2`, `STAGE3`, `OUTPUT` channels)
- `res50_adam_lr1e-3.yaml` and `experiments/*` are mostly from 2D pose HRNet/ResNet ecosystem.

---

## 7) Folder Structure (Practical Map)

```text
.
├── HRNet_GCN_WB.py                 # Main train/eval entrypoint (3D whole-body)
├── infer.py                        # Inference entrypoint
├── README.md
├── requirements.txt
├── w32_adam_lr1e-3.yaml            # Main graph model architecture config
├── res50_adam_lr1e-3.yaml          # HRNet/ResNet 2D-style config
│
├── data/
│   ├── h3wb_train.npz              # Training split data
│   ├── h3wb_test.npz               # Test split data
│   ├── data_preparation.py         # Video->frame helpers
│   ├── prepare_data_h36m.py        # H36M conversion utility
│   └── prepare_data_2d_h36m_sh.py  # Stacked-hourglass 2D conversion utility
│
├── utils/
│   ├── prepare_data_h3wb.py        # H3WB dataset class + camera/skeleton metadata
│   ├── camera.py / skeleton.py / mocap_dataset.py / utils.py
│
├── common/
│   ├── data_utils.py               # 2D/3D preprocessing + fetch
│   ├── generators.py               # PoseGenerator dataset wrapper
│   ├── generators_video.py         # Sequence/chunk generators (VideoPose3D style)
│   ├── graph_utils.py              # Adjacency builders
│   ├── camera.py                   # Projections + coordinate transforms
│   ├── loss.py                     # MPJPE and related errors
│   ├── log.py / utils.py           # Logging/checkpoint/lr helpers
│   └── h36m_dataset.py etc.        # Legacy/auxiliary dataset code
│
├── models/
│   ├── graph_hrnet_multi_branch.py # Model 1
│   ├── graph_resnet.py             # Model 2
│   ├── graph_hrnet.py              # Model 3
│   ├── graph_sh.py                 # Model 4
│   ├── graph_non_local.py
│   └── gconv/                      # Graph convolution operator families
│
├── lib/                            # Upstream HRNet 2D pose codebase pieces
│   ├── config/ core/ dataset/ models/ nms/ utils/
│   └── Makefile                    # Builds native NMS extension
│
├── experiments/                    # HRNet/ResNet experiment YAMLs (COCO/MPII)
├── progress/                       # CLI progress bar helpers
└── checkpoint/                     # Training outputs (weights/logs/params)
```

---

## 8) Important Implementation Notes / Pitfalls

1. GPU is hardcoded as `cuda:0` in main scripts.
2. `--resume` and `--evaluate` are mutually exclusive.
3. Model IDs supported in code are `1..4`.
4. Checkpoint path logic is inconsistent for model 3/4 (not rooted under `--checkpoint`).
5. `infer.py` depends on missing `models/graph_hrnet_multi_branch_58.py`.

---

## 9) What Is Core vs Legacy

## Core path for this project

- `HRNet_GCN_WB.py`
- `infer.py`
- `utils/prepare_data_h3wb.py`
- `common/data_utils.py`, `common/generators.py`
- `models/`, `models/gconv/`
- `w32_adam_lr1e-3.yaml`

## Mostly inherited / auxiliary ecosystem

- `lib/` (2D HRNet stack)
- `experiments/` (COCO/MPII 2D configs)
- `data/prepare_data_h36m.py`, `data/prepare_data_2d_h36m_sh.py`

---

## 10) Quick Start Commands

```bash
# Train
python HRNet_GCN_WB.py --gcn dc_preagg --model 1

# Evaluate
python HRNet_GCN_WB.py --model 1 --gcn dc_preagg --evaluate checkpoint/ckpt_best.pth.tar -cfg checkpoint/w32_adam_lr1e-3.yaml

# Infer (depends on model file availability noted above)
python infer.py --evaluate checkpoint_58.4/ckpt_best.pth.tar -cfg checkpoint_58.4/w32_adam_lr1e-3.yaml
```






(venv) genesys@genesys-MS-7B05:~/hamza/HR-GCN % python train_rehab.py --pretrained checkpoint/HRGCN/dc_preagg-2026-04-24T10:48:02/ckpt_best.pth.tar --cfg w32_adam_lr1e-3.yaml --epochs 50 --lambda_angle 0.0 --lambda_constraint 0.0 --checkpoint checkpoint_rehab_baseline_v4
==> Log file: checkpoint_rehab_baseline_v4/train_rehab.log
