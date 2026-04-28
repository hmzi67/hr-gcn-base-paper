"""
UI-PRMD dataset preprocessor for the GCADA rehabilitation pipeline.

Reads Vicon .txt position files, maps 39 Vicon joints to COCO-WholeBody
133-joint format (body joints only; face/hands zero-padded), projects 3D→2D
via orthographic projection, computes 8 clinical ROM angles geometrically,
and saves train/test NPZ files compatible with the existing H3WB data pipeline.

Usage:
    python utils/prepare_data_uiprmd.py \
        --data_dir data/UI-PRMD/raw \
        --output_dir data/
"""

import argparse
import glob
import os

import numpy as np


# ---------------------------------------------------------------------------
# Vicon 39-joint → COCO body 23-joint mapping
# Identified from mean 3D positions (z=height, y=anterior-posterior):
#   Joints 23/24: hip LASI/RASI (z≈732 mm)
#   Joints 25/26: hip LPSI/RPSI (z≈788 mm, posterior)
#   Joints 27/33: knees (z≈480-607 mm)
#   Joints 29/35: ankles (z≈309-316 mm)
#   Joints 30-32 / 36-38: feet (z≈50-91 mm)
#   Joints 0-3: head markers (z≈1300 mm)
#   Joints 4: C7 / neck (z≈1194 mm)
#   Joints 8/15: shoulders (z≈1142/1190 mm)
#   Joints 10/17: elbows (z≈1209/1201 mm)
#   Joints 12/19: wrists (z≈1368/1367 mm) — approximate
# ---------------------------------------------------------------------------
VICON_TO_COCO_BODY = {
    # Vicon joint index : COCO body joint index (0-22)
    # Head / face (approximate — head markers)
    0:  0,   # LFHD → nose proxy
    # Shoulders
    8:  5,   # LSHO → left shoulder
    15: 6,   # RSHO → right shoulder
    # Elbows
    10: 7,   # LELB → left elbow
    17: 8,   # RELB → right elbow
    # Wrists
    12: 9,   # LWRA → left wrist
    19: 10,  # RWRA → right wrist
    # Hips (ASIS — anterior superior iliac spine)
    23: 11,  # LASI → left hip
    24: 12,  # RASI → right hip
    # Knees — indices 27/33 are thigh wands (LTHI/RTHI) at Z≈715mm;
    # true lateral knee epicondyle markers are at 28/34 (Z≈507mm).
    28: 13,  # LKNE → left knee
    34: 14,  # RKNE → right knee
    # Ankles
    29: 15,  # LANK → left ankle
    35: 16,  # RANK → right ankle
    # Feet
    30: 17,  # left big toe proxy
    31: 18,  # left small toe proxy
    32: 19,  # left heel proxy
    36: 20,  # right big toe proxy
    37: 21,  # right small toe proxy
    38: 22,  # right heel proxy
}

# Subject-level split (1-indexed to match filename s01-s10)
TRAIN_SUBJECTS = list(range(1, 9))   # s01-s08
TEST_SUBJECTS  = [9, 10]             # s09-s10

# ROM angle output order (must match JOINT_LIMIT_TENSOR_ORDER in anatomical_constraints.py)
# cervical_yaw and cervical_roll removed: UI-PRMD has no cervical rotation/roll ground truth
# (single-frame geometry cannot recover axial rotation without a reference orientation).
ROM_NAMES = [
    'cervical_pitch',
    'trunk_flex',
    'left_hip', 'right_hip',
    'left_knee', 'right_knee',
]


# ---------------------------------------------------------------------------
# Geometric helpers
# ---------------------------------------------------------------------------

def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle (degrees) between two 3-D vectors."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    cos_a = np.dot(v1, v2) / (n1 * n2)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))


def _angle_at_vertex(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at vertex b formed by vectors b→a and b→c (degrees)."""
    return _angle_between(a - b, c - b)


def compute_rom_angles(pos: np.ndarray) -> np.ndarray:
    """
    Compute 6 clinical ROM angles per frame from Vicon 3D positions.

    pos: (T, 39, 3)  — Vicon joint positions in any consistent unit
    returns: (T, 6)  — angles in degrees, order matches ROM_NAMES:
        [cervical_pitch, trunk_flex, left_hip, right_hip, left_knee, right_knee]

    Hip flexion: angle between global vertical (Vicon +Z) and the hip-to-knee
    vector. Equivalent to "thigh deviation from upright" — symmetric by definition
    across left and right sides for a standing subject. Using the measured spine
    direction as reference instead introduced a systematic 20–30° L/R asymmetry
    because the spine vector tilts toward one lateral side in this lab's coordinate
    system.

    Knee flexion: standard 3-point angle at the knee between femur and tibia.
    Knee markers: indices 28 (LKNE) and 34 (RKNE) — the lateral knee epicondyle
    markers at Z≈507mm. Indices 27/33 are thigh wands (LTHI/RTHI) at Z≈715mm
    and must NOT be used here.
    """
    T = pos.shape[0]
    angles = np.zeros((T, 6), dtype=np.float32)

    # Anatomical landmarks
    lasi  = pos[:, 23]   # left  hip ASIS
    rasi  = pos[:, 24]   # right hip ASIS
    lpsi  = pos[:, 25]   # left  hip PSIS
    rpsi  = pos[:, 26]   # right hip PSIS
    lkne  = pos[:, 28]   # left  knee (LKNE, lateral epicondyle, Z≈507mm)
    rkne  = pos[:, 34]   # right knee (RKNE, lateral epicondyle, Z≈507mm)
    lank  = pos[:, 29]   # left  ankle
    rank  = pos[:, 35]   # right ankle
    lsho  = pos[:,  8]   # left  shoulder (used for trunk flex only)
    rsho  = pos[:, 15]   # right shoulder
    head  = pos[:,  0]   # head proxy (LFHD)
    c7    = pos[:,  4]   # C7 (cervical vertebra 7)

    # Hip joint centres = average of ASIS and PSIS markers
    l_hip_ctr = (lasi + lpsi) / 2.0
    r_hip_ctr = (rasi + rpsi) / 2.0
    mid_hip   = (l_hip_ctr + r_hip_ctr) / 2.0
    mid_sho   = (lsho + rsho) / 2.0

    up = np.array([0.0, 0.0, 1.0])   # global vertical (Z-up in Vicon)

    for t in range(T):
        spine = mid_sho[t] - mid_hip[t]

        # --- Cervical pitch: angle between head-neck vector and vertical ---
        neck_to_head = head[t] - c7[t]
        angles[t, 0] = _angle_between(neck_to_head, up)

        # --- Trunk flexion: angle between spine vector and vertical ---
        angles[t, 1] = _angle_between(spine, up)

        # --- Hip flexion (clinical convention: 0° = upright, increases with flexion) ---
        # _angle_between(up, femur) gives ~180° for upright (femur antiparallel to up).
        # Clinical hip flexion = 180° minus that value → 0° upright, ~90° thigh horizontal.
        angles[t, 2] = 180.0 - _angle_between(up, lkne[t] - l_hip_ctr[t])
        angles[t, 3] = 180.0 - _angle_between(up, rkne[t] - r_hip_ctr[t])

        # --- Knee flexion (clinical convention: 0° = straight, increases with bend) ---
        # _angle_at_vertex gives ~180° for a straight leg.
        # Clinical knee flexion = 180° minus that value.
        angles[t, 4] = 180.0 - _angle_at_vertex(l_hip_ctr[t], lkne[t], lank[t])
        angles[t, 5] = 180.0 - _angle_at_vertex(r_hip_ctr[t], rkne[t], rank[t])

    return angles


# ---------------------------------------------------------------------------
# 2D projection
# ---------------------------------------------------------------------------

def orthographic_projection_2d(pos3d: np.ndarray) -> np.ndarray:
    """
    Frontal orthographic projection: (Vicon_X, −Vicon_Z).
    Vicon Z is the vertical axis (Z-up).  Image y increases downward, so we
    negate Z to match image convention (head → negative y, feet → positive y).
    This matches H3WB's camera view where x=lateral, y=downward-vertical.

    pos3d: (..., 3) in mm
    returns: (..., 2) in mm, same units — normalisation happens in normalize_2d
    """
    xz = np.stack([pos3d[..., 0], -pos3d[..., 2]], axis=-1)
    return xz.astype(np.float32)


# Physical scale for 2D normalisation.
# H3WB uses normalize_screen_coordinates(w=1000px) → divide by 500px.
# At focal_length≈1145px and subject distance≈3 000mm, 500px ≈ 1310mm.
# We use 1000mm so body joints span roughly the same range as H3WB inputs
# (empirically gives 2D std ≈ 0.15–0.20, matching H3WB's std ≈ 0.167).
_2D_SCALE_MM = 1000.0


def normalize_2d(poses_2d: np.ndarray) -> np.ndarray:
    """
    Root-center at hip midpoint (COCO joints 11 & 12), scale by fixed
    physical constant to match H3WB camera-normalised coordinate range.
    Joints that are zero (unmapped Vicon→COCO joints, e.g. face/hands)
    are kept at zero — they must not inherit the hip-centre offset.

    poses_2d: (N, 133, 2) in mm
    returns:  (N, 133, 2) normalised, zero-masked for unmapped joints
    """
    # Boolean mask: joints that have real Vicon data (non-zero before centering)
    mapped = np.any(poses_2d != 0, axis=-1, keepdims=True)  # (N, 133, 1)

    hip_center = (poses_2d[:, 11:12, :] + poses_2d[:, 12:13, :]) / 2.0
    centered = (poses_2d - hip_center) * mapped   # unmapped joints stay at 0

    return (centered / _2D_SCALE_MM).astype(np.float32)


def normalize_3d(poses_3d: np.ndarray) -> np.ndarray:
    """
    Root-center at hip midpoint and convert mm → metres.
    Unmapped joints (zero in poses_3d) are kept at zero after centering.

    poses_3d: (N, 133, 3) in mm
    returns:  (N, 133, 3) in metres, hip-centred, zero-masked
    """
    mapped = np.any(poses_3d != 0, axis=-1, keepdims=True)  # (N, 133, 1)
    hip_center = (poses_3d[:, 11:12, :] + poses_3d[:, 12:13, :]) / 2.0
    centered = (poses_3d - hip_center) * mapped
    return (centered / 1000.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Per-file loader
# ---------------------------------------------------------------------------

def load_vicon_file(pos_path: str):
    """
    Load one Vicon positions .txt file.

    Returns:
        poses_133_3d: (T, 133, 3) in mm — COCO 133-joint format, body only
        poses_133_2d: (T, 133, 2) — orthographic 2D, un-normalized
        rom_angles:   (T, 6)     — geometric ROM degrees
    """
    raw = np.loadtxt(pos_path)            # (T, 117)
    T = raw.shape[0]
    pos39 = raw.reshape(T, 39, 3)         # (T, 39, 3)  in mm

    # ---- Map to 133-joint COCO-WholeBody format ----
    poses_3d = np.zeros((T, 133, 3), dtype=np.float32)
    for vicon_idx, coco_idx in VICON_TO_COCO_BODY.items():
        poses_3d[:, coco_idx, :] = pos39[:, vicon_idx, :]

    # ---- Orthographic 2D (use xy plane) ----
    poses_2d = np.zeros((T, 133, 2), dtype=np.float32)
    poses_2d[:, :, :] = orthographic_projection_2d(poses_3d)

    # ---- ROM angles ----
    rom_angles = compute_rom_angles(pos39)   # (T, 6)

    return poses_3d, poses_2d, rom_angles


# ---------------------------------------------------------------------------
# Main preprocessing pipeline
# ---------------------------------------------------------------------------

def preprocess(data_dir: str, output_dir: str):
    vicon_pos_dir = os.path.join(data_dir, 'Movements', 'Vicon', 'Positions')
    files = sorted(glob.glob(os.path.join(vicon_pos_dir, 'm??_s??_positions.txt')))

    if not files:
        raise FileNotFoundError(
            f'No position files found in {vicon_pos_dir}. '
            'Make sure Vicon/Positions/ contains m??_s??_positions.txt files.'
        )

    print(f'Found {len(files)} Vicon position files.')

    splits = {'train': [], 'test': []}

    for fpath in files:
        base = os.path.basename(fpath)           # e.g. m03_s07_positions.txt
        parts = base.split('_')
        exercise_id = int(parts[0][1:]) - 1      # 0-indexed (0-9)
        subject_id  = int(parts[1][1:])          # 1-indexed (1-10)

        split = 'train' if subject_id in TRAIN_SUBJECTS else 'test'

        print(f'  [{split}] {base}', end='  ', flush=True)
        poses_3d, poses_2d, rom_angles = load_vicon_file(fpath)
        T = poses_3d.shape[0]
        print(f'T={T}')

        splits[split].append({
            'poses_3d':    poses_3d,
            'poses_2d':    poses_2d,
            'rom_angles':  rom_angles,
            'subject_ids': np.full(T, subject_id - 1, dtype=np.int32),   # 0-indexed
            'exercise_ids': np.full(T, exercise_id,   dtype=np.int32),
            'frame_ids':   np.arange(T, dtype=np.int32),
        })

    os.makedirs(output_dir, exist_ok=True)

    for split_name, samples in splits.items():
        if not samples:
            print(f'WARNING: no samples for split "{split_name}"')
            continue

        all_3d  = np.concatenate([s['poses_3d']    for s in samples], axis=0)
        all_2d  = np.concatenate([s['poses_2d']    for s in samples], axis=0)
        all_rom = np.concatenate([s['rom_angles']  for s in samples], axis=0)
        all_sub = np.concatenate([s['subject_ids'] for s in samples], axis=0)
        all_exc = np.concatenate([s['exercise_ids']for s in samples], axis=0)
        all_frm = np.concatenate([s['frame_ids']   for s in samples], axis=0)

        # Normalize after concatenation for consistent statistics
        all_3d_norm = normalize_3d(all_3d)
        all_2d_norm = normalize_2d(all_2d)

        out_path = os.path.join(output_dir, f'uiprmd_{split_name}.npz')
        np.savez_compressed(
            out_path,
            poses_2d    = all_2d_norm,
            poses_3d    = all_3d_norm,
            rom_angles  = all_rom,
            subject_ids = all_sub,
            exercise_ids= all_exc,
            frame_ids   = all_frm,
        )
        print(f'\nSaved {split_name}: {out_path}')
        print(f'  poses_2d:    {all_2d_norm.shape}')
        print(f'  poses_3d:    {all_3d_norm.shape}')
        print(f'  rom_angles:  {all_rom.shape}')
        print(f'  subject_ids: {all_sub.shape}  unique={np.unique(all_sub)}')
        print(f'  exercise_ids:{all_exc.shape}  unique={np.unique(all_exc)}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='UI-PRMD dataset preprocessor')
    parser.add_argument('--data_dir',   default='data/UI-PRMD/raw',
                        help='Path to UI-PRMD raw data directory')
    parser.add_argument('--output_dir', default='data/',
                        help='Output directory for NPZ files')
    args = parser.parse_args()

    preprocess(args.data_dir, args.output_dir)
    print('\nDone.')


if __name__ == '__main__':
    main()
