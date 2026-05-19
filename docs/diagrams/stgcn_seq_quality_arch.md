# STGCN-Seq (UI-PRMD Quality) — Architecture

This document describes the architecture and data flow of `stgcn_seq_quality.py`.

## Summary

STGCN-Seq builds a spatial-temporal GCN backbone over 12 ROM angle nodes to predict exercise labels, validity (real vs. cropped), and a continuous quality score. Spatial structure is learned per-exercise via learnable adjacency matrices; temporal relations are modeled with a Gaussian-initialized learnable adjacency. Training runs in three stages: exercise classification, validity classification (with synthetic invalids), and quality regression (L1) with optional backbone freezing and checkpointing by validation MAD.

## Step-by-step

1. Load NPZ files: `train_npz` and `test_npz` arrays (`sequences` / `rom_angles`, `exercise_ids`, `subject_ids`, `quality_labels`).
2. Build sequences:
  - If `--split_mode=random`: extract sliding windows (size `M`, stride 50) from frame-level data.
  - Else (subject split): use per-file sequences and lengths.
3. (Optional) Fit per-exercise GMMs on "correct" sequences and compute continuous quality scores to replace binary labels (`--use_gmm_scores`).
4. Create `UIPromdSeqDataset`: normalize angles (`/ ROM_NORM`), resize/interpolate to `M`, optionally augment (`_augment_rom`) and add synthetic invalid crops for VC training.
5. Create `DataLoader`s producing batches with `poses` shaped `(B, M, J, C)` and corresponding `exercise_id`, `quality_score`, `validity`.
6. Forward pass through `STGCNSeq`:
  - Flatten `(B, M, J, C)` → `(B*M, J, C)` and process with `SpatialGCN` using `A_spatial[exercise_id]`.
  - Reshape to `(B, M, J*hidden)` and process with `TemporalGCN` using `A_temporal`.
  - Global-average pool to `(B, hidden)` and pass to three heads: `ExerciseClassifier`, `ValidityClassifier`, `QualityHead`.
7. Training schedule:
  - Stage 1: train `ExerciseClassifier` (CrossEntropy) across `epochs_ec`.
  - Stage 2: train `ValidityClassifier` (CrossEntropy) across `epochs_vc` (train set includes synthetic invalids).
  - Stage 3: train `QualityHead` (L1) across `epochs_reg`; backbone can be frozen and differential learning rates applied. Save best checkpoint by validation MAD.
8. Evaluation: reload best checkpoint; compute EC/VC accuracies and per-exercise quality metrics (MAD, RMSE, MAPE) via `evaluate_quality`; save CSV.


```mermaid
graph TD
  A[NPZ files\n(`train_npz` / `test_npz`)] --> B[Loading arrays\n(rom_angles, exercise_ids,\nsubject_ids, quality_labels)]
  B --> C[Sequence Builder\n(sliding-window or _build_items_from_arrays)]
  C --> D[`UIPromdSeqDataset`]
  D --> E[`DataLoader`]
  E --> F[`STGCNSeq` Model]
  subgraph Model
    F1[STGCNBackbone]
    F2[ExerciseClassifier]
    F3[ValidityClassifier]
    F4[QualityHead]
    F --> F1
    F --> F2
    F --> F3
    F --> F4
    subgraph Backbone
      S[SpatialGCN\n(per-exercise A_spatial)\nInput: (B*M, J, C)\nOutput: (B*M, J, hidden)]
      T[TemporalGCN\n(A_temporal, gaussian-init)\nInput: (B, M, J*hidden)\nOutput: (B, hidden)]
      F1 -->|uses feat| S
      S --> T
      T --> F1
    end
  end
  E -->|batches (poses, ex_id,... )| F
  F2 --> G[Stage 1: Exercise Classifier\nLoss: CrossEntropy]
  F3 --> H[Stage 2: Validity Classifier\nLoss: CrossEntropy]
  F4 --> I[Stage 3: Quality Regression\nLoss: L1]
  G --> J[Optimizer: Adam]
  H --> J
  I --> K[Checkpointing\n(save best by val MAD)]
  K --> L[Evaluation\nevaluate_quality / EC/VC tests]
```

**Component Map**
- **`UIPromdSeqDataset`**: loads NPZ, applies subject/random split, builds items list of (seq_resized M×12, eid, qs, validity). Uses helpers: `_resize_sequence`, `_build_invalid_seq`, `_augment_rom`.
- **`STGCNSeq`**: top-level model containing:
  - `STGCNBackbone`: spatial → temporal pipeline.
  - `SpatialGCN`: per-exercise learnable adjacency `A_spatial[eid]` (nn.ParameterList), linear layers, BatchNorm; processes `(B*M, J, C)` and outputs per-joint features.
  - `TemporalGCN`: learnable temporal adjacency `A_temporal` (gaussian init), linear layers, BN; pools over time → global feature `(B, hidden)`.
  - Heads: `ExerciseClassifier` (10-way), `ValidityClassifier` (2-way), `QualityHead` (sigmoid regression to [0,1]).
- **Training stages**:
  1. Stage 1 — EC: train `ExerciseClassifier` (CrossEntropy).
  2. Stage 2 — VC: train `ValidityClassifier` (CrossEntropy) using synthetic cropped invalids.
  3. Stage 3 — QR: train `QualityHead` (L1) with optional backbone freeze and differential LR.
- **GMM scoring**: `_compute_gmm_from_arrays` / `compute_gmm_scores` can replace binary labels with continuous quality scores from per-exercise GMM likelihoods.

**Important Data Shapes**
- Raw frames: `(N_frames, 12)` — per-frame ROM angles (degrees).
- Windowed sequences: `(N_seqs, M, 12)` after sliding-window extraction or resizing.
- Dataset items (batched): `poses` tensor shape `(B, M, J, C)` with `J=12`, `C=1`.
- SpatialGCN input after flattening: `(B*M, J, C)` → output `(B*M, J, hidden)`.
- TemporalGCN input: `(B, M, J*hidden)` → output: `(B, hidden)`.
- Heads operate on `(B, hidden)` producing `(B, n_classes)` or `(B,1)`.

**Legend & Notes**
- `ROM_J = 12` nodes; topology given by `ROM_EDGES` in the source file.
- Normalization: ROM angles divided by `ROM_NORM=180.0` → range ~[0,1].
- Augmentations: speed augmentation (drop/repeat), linear interpolation to `M`, small Gaussian noise.
- Checkpointing: during QR stage the script saves best model by validation MAD and reloads it for final evaluation.
- Evaluation outputs: EC/VC accuracy and per-exercise quality metrics (MAD, RMSE, MAPE) saved to CSV.

**See also**
- Implementation: [stgcn_seq_quality.py](stgcn_seq_quality.py#L1)

---
Want this exported as an SVG/PNG diagram file saved in the repo (e.g. `docs/diagrams/stgcn_seq_quality_arch.svg`)? Reply with your preferred format and I'll generate and save it.
