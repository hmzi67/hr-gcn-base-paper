"""
UI-PRMD dataset preprocessor for the GCADA rehabilitation pipeline.

Reads Vicon .txt position files, maps 39 Vicon joints to COCO-WholeBody
133-joint format (body joints only; face/hands zero-padded), projects 3D→2D
via orthographic projection, computes 8 clinical ROM angles geometrically,
and saves train/test NPZ files compatible with the existing H3WB data pipeline.

Quality scores are loaded from Scores/m{ex:02d}_s{sub:02d}_scores.txt if
present. Otherwise they are derived from the Incorrect Movements folder
(correct=1.0, incorrect=0.0). Missing files fall back to -1.0 (sentinel).

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
# Verified from mean 3D positions and Y-axis convention:
#   Y-axis: positive Y = patient's LEFT (confirmed via LASI idx 23 Y>0)
#           negative Y = patient's RIGHT (confirmed via RASI idx 24 Y<0)
#   Joints 23/24: hip LASI/RASI (z≈732 mm)
#   Joints 25/26: hip LPSI/RPSI (z≈788 mm, posterior)
#   Joints 27/33: knee wands / 28/34: knee epicondyles (z≈480-607 mm)
#   Joints 29/35: ankles (z≈309-316 mm)
#   Joints 30-32 / 36-38: feet (z≈50-91 mm)
#   Joints 0-3: head markers (z≈1300 mm)
#   Joint  4: C7 / neck (z≈1194 mm)
#   Joints 8/9:  shoulders — idx 8=RSHO (Y<0, right), idx 9=LSHO (Y>0, left)
#   Joints 11/17: elbows   — idx 11=LELB (Y>0, left), idx 17=RELB (Y<0, right)
#   Joints 13/19: wrists   — idx 13=LWRA (Y>0, left), idx 19=RWRA (Y<0, right)
#   (indices 10,12,16,18 are wand/intermediate markers — skipped)
# ---------------------------------------------------------------------------
VICON_TO_COCO_BODY = {
    # Vicon joint index : COCO body joint index (0-22)
    #
    # Y-axis convention confirmed from hip markers:
    #   LASI (idx 23) Y>0  → positive Y = PATIENT'S LEFT
    #   RASI (idx 24) Y<0  → negative Y = PATIENT'S RIGHT
    #
    # Arm marker layout (39-marker set):
    #   idx  8: RSHO  (right shoulder, Y<0)
    #   idx  9: LSHO  (left shoulder,  Y>0)
    #   idx 10: LUPA  (left upper-arm wand — skipped)
    #   idx 11: LELB  (left elbow,      Y>0)
    #   idx 12: LFRM  (left forearm wand — skipped)
    #   idx 13: LWRA  (left wrist,      Y>0)
    #   idx 14: LWRB  (skipped)
    #   idx 15: LFIN  (left finger — skipped)
    #   idx 16: RUPA  (right upper-arm wand — skipped)
    #   idx 17: RELB  (right elbow,     Y<0)
    #   idx 18: RFRM  (right forearm wand — skipped)
    #   idx 19: RWRA  (right wrist,     Y<0)
    #
    # Head / face (approximate — head markers)
    0:  0,   # LFHD → nose proxy
    # Shoulders  (FIXED: was 8→LSHO, 15→RSHO — both wrong side/index)
    9:  5,   # LSHO → left shoulder
    8:  6,   # RSHO → right shoulder
    # Elbows     (FIXED: left elbow was 10=LUPA, true LELB is 11)
    11: 7,   # LELB → left elbow
    17: 8,   # RELB → right elbow
    # Wrists     (FIXED: left wrist was 12=LFRM, true LWRA is 13)
    13: 9,   # LWRA → left wrist
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
ROM_NAMES = [
    'cervical_pitch', 'trunk_flex',
    'left_shoulder_flex', 'right_shoulder_flex',
    'left_shoulder_abd',  'right_shoulder_abd',
    'left_hip', 'right_hip',
    'left_knee', 'right_knee',
    'left_ankle', 'right_ankle',
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
    Compute 12 clinical ROM angles per frame from Vicon 3D positions.

    pos: (T, 39, 3)  — Vicon joint positions in mm (Z-up)
    returns: (T, 12) — degrees, order matches ROM_NAMES:
        [cerv_pitch, trunk_flex,
         l_sho_flex, r_sho_flex, l_sho_abd, r_sho_abd,
         l_hip, r_hip, l_knee, r_knee, l_ankle, r_ankle]

    Conventions (all clinical, 0° = anatomical neutral):
      - Hip/knee: 0° upright/straight, increases with flexion
      - Shoulder flex: 0° arm at side, increases as arm raises forward
      - Shoulder abd:  0° arm at side, increases as arm raises sideways
      - Ankle: 90° = neutral (tibia vertical), >90° = dorsiflexion, <90° = plantarflexion

    Vectorised — no Python loop.
    """
    def _va(v1, v2):
        """Batch angle (degrees) between (T,3) vectors."""
        n1 = np.linalg.norm(v1, axis=-1, keepdims=True).clip(1e-8, None)
        n2 = np.linalg.norm(v2, axis=-1, keepdims=True).clip(1e-8, None)
        cos = ((v1 / n1) * (v2 / n2)).sum(-1).clip(-1.0 + 1e-7, 1.0 - 1e-7)
        return np.degrees(np.arccos(cos))

    T = pos.shape[0]

    # Anatomical landmarks (Vicon indices)
    lasi = pos[:, 23]   # left  hip ASIS
    rasi = pos[:, 24]   # right hip ASIS
    lpsi = pos[:, 25]   # left  hip PSIS
    rpsi = pos[:, 26]   # right hip PSIS
    lkne = pos[:, 28]   # left  knee (LKNE lateral epicondyle, Z≈507mm)
    rkne = pos[:, 34]   # right knee
    lank = pos[:, 29]   # left  ankle
    rank = pos[:, 35]   # right ankle
    lsho = pos[:,  9]   # left  shoulder (LSHO, Y>0)
    rsho = pos[:,  8]   # right shoulder (RSHO, Y<0)
    lelb = pos[:, 11]   # left  elbow    (LELB, Y>0)
    relb = pos[:, 17]   # right elbow    (RELB, Y<0)
    head = pos[:,  0]   # head proxy (LFHD)
    c7   = pos[:,  4]   # C7 cervical vertebra
    ltoe = pos[:, 30]   # left  big-toe proxy
    rtoe = pos[:, 36]   # right big-toe proxy
    lhee = pos[:, 32]   # left  heel proxy
    rhee = pos[:, 38]   # right heel proxy

    l_hip_ctr = (lasi + lpsi) / 2.0
    r_hip_ctr = (rasi + rpsi) / 2.0
    mid_hip   = (l_hip_ctr + r_hip_ctr) / 2.0
    mid_sho   = (lsho + rsho) / 2.0
    spine     = mid_sho - mid_hip                       # (T,3) points upward
    up        = np.zeros((T, 3), dtype=np.float32)
    up[:, 2]  = 1.0                                     # Z-up in Vicon

    # ── 0. Cervical pitch ────────────────────────────────────────────────────
    cerv_pitch = _va(head - c7, up)

    # ── 1. Trunk flexion ─────────────────────────────────────────────────────
    trunk_flex = _va(spine, up)

    # ── 2-3. Shoulder flexion (0° arm at side, increases forward) ────────────
    # 180 - angle(spine, upper_arm): spine points up, resting arm points down
    # → antiparallel → 180° → 180-180=0° at rest ✓
    l_upper_arm = lelb - lsho
    r_upper_arm = relb - rsho
    l_sho_flex  = 180.0 - _va(spine, l_upper_arm)
    r_sho_flex  = 180.0 - _va(spine, r_upper_arm)

    # ── 4-5. Shoulder abduction (frontal-plane projection) ───────────────────
    # frontal_normal = forward direction = cross(lr_axis, up)
    lr_axis        = rsho - lsho
    frontal_normal = np.cross(lr_axis, up)
    fn_norm        = np.linalg.norm(frontal_normal, axis=-1, keepdims=True).clip(1e-8, None)
    frontal_normal = frontal_normal / fn_norm
    # project upper arm onto frontal plane, then 180 - angle(up, proj)
    l_proj = l_upper_arm - (l_upper_arm * frontal_normal).sum(-1, keepdims=True) * frontal_normal
    r_proj = r_upper_arm - (r_upper_arm * frontal_normal).sum(-1, keepdims=True) * frontal_normal
    l_sho_abd = 180.0 - _va(up, l_proj)
    r_sho_abd = 180.0 - _va(up, r_proj)

    # ── 6-7. Hip flexion (0° upright) ────────────────────────────────────────
    l_femur = lkne - l_hip_ctr
    r_femur = rkne - r_hip_ctr
    l_hip   = 180.0 - _va(up, l_femur)
    r_hip   = 180.0 - _va(up, r_femur)

    # ── 8-9. Knee flexion (0° straight) ──────────────────────────────────────
    l_tibia = lank - lkne
    r_tibia = rank - rkne
    l_knee  = 180.0 - _va(-l_femur, l_tibia)
    r_knee  = 180.0 - _va(-r_femur, r_tibia)

    # ── 10-11. Ankle dorsiflexion (90°=neutral, >90°=dorsiflex, <90°=plantarflex)
    # Use heel→toe as foot long axis (horizontal at neutral) — NOT ankle→toe
    # which points mostly downward and gives ~12° instead of ~90°.
    l_foot  = ltoe - lhee   # heel→toe, roughly horizontal
    r_foot  = rtoe - rhee
    l_ankle = _va(l_tibia, l_foot)   # 90° at neutral, increases with dorsiflex
    r_ankle = _va(r_tibia, r_foot)

    return np.stack([
        cerv_pitch, trunk_flex,
        l_sho_flex, r_sho_flex,
        l_sho_abd,  r_sho_abd,
        l_hip,      r_hip,
        l_knee,     r_knee,
        l_ankle,    r_ankle,
    ], axis=1).astype(np.float32)


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
# Quality score helpers
# ---------------------------------------------------------------------------

def load_quality_score(scores_dir: str, exercise_id: int, subject_id: int) -> float:
    """
    Load a per-sequence quality score scalar from the Scores directory.

    UI-PRMD Scores files (when present) contain a single float per sequence.
    Returns -1.0 (sentinel) if the file does not exist.

    exercise_id: 0-indexed  → filename uses 1-indexed (exercise_id+1)
    subject_id:  1-indexed  (matches filename)
    """
    fname = f'm{exercise_id + 1:02d}_s{subject_id:02d}_scores.txt'
    fpath = os.path.join(scores_dir, fname)
    if not os.path.isfile(fpath):
        return -1.0
    try:
        val = float(np.loadtxt(fpath).flat[0])
        return float(np.clip(val, 0.0, 1.0))
    except Exception:
        return -1.0


def derive_quality_from_incorrect(data_dir: str, exercise_id: int, subject_id: int) -> float:
    """
    Derive binary quality score from the Incorrect Movements folder.

    Returns 1.0 if the subject/exercise exists ONLY in Movements/ (correct),
    0.0  if a matching file also exists in 'Incorrect Movements/Vicon/Positions/'.
    Falls back to -1.0 if no Incorrect Movements folder is found at all.

    exercise_id: 0-indexed
    subject_id:  1-indexed
    """
    incorrect_dir = os.path.join(
        data_dir, 'Incorrect Movements', 'Vicon', 'Positions'
    )
    if not os.path.isdir(incorrect_dir):
        return -1.0
    fname = f'm{exercise_id + 1:02d}_s{subject_id:02d}_positions.txt'
    # If an incorrect version exists → quality 0.0; correct-only → 1.0
    return 0.0 if os.path.isfile(os.path.join(incorrect_dir, fname)) else 1.0


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

def preprocess(data_dir: str, output_dir: str,
               correct_only: bool = False,
               incorrect_only: bool = False,
               split_filter: str = 'all'):
    """
    split_filter: 'all' | 'train_only' | 'test_only'
    correct_only:   load only from Movements/ (correct executions)
    incorrect_only: load only from 'Incorrect Movements/' (_inc files)
    Default (neither flag): loads both folders, reproducing the full mixed NPZ.
    """
    scores_dir = os.path.join(data_dir, 'Movements', 'Vicon', 'Scores')

    # ── Collect source files ──────────────────────────────────────────────────
    file_entries = []   # list of (fpath, is_correct: bool)

    if not incorrect_only:
        correct_dir = os.path.join(data_dir, 'Movements', 'Vicon', 'Positions')
        for fp in sorted(glob.glob(os.path.join(correct_dir, 'm??_s??_positions.txt'))):
            file_entries.append((fp, True))

    if not correct_only:
        incorrect_dir = os.path.join(data_dir, 'Incorrect Movements', 'Vicon', 'Positions')
        for fp in sorted(glob.glob(os.path.join(incorrect_dir, 'm??_s??_positions_inc.txt'))):
            file_entries.append((fp, False))

    if not file_entries:
        raise FileNotFoundError(
            f'No position files found under {data_dir}. '
            'Check that Movements/Vicon/Positions/ contains m??_s??_positions.txt '
            'and/or "Incorrect Movements/Vicon/Positions/" contains m??_s??_positions_inc.txt'
        )

    has_scores_dir = os.path.isdir(scores_dir)
    mode_tag = 'correct-only' if correct_only else ('incorrect-only' if incorrect_only else 'mixed')
    print(f'Found {len(file_entries)} Vicon files ({mode_tag}).')
    if has_scores_dir:
        print(f'Quality scores: loading from {scores_dir}')
    else:
        print('Quality scores: Scores/ dir not found — deriving from Incorrect Movements/')

    # ── Output filename suffix ────────────────────────────────────────────────
    type_suffix = '_correct' if correct_only else ('_incorrect' if incorrect_only else '')

    splits = {'train': [], 'test': []}

    for fpath, is_correct in file_entries:
        base  = os.path.basename(fpath)
        parts = base.split('_')
        exercise_id = int(parts[0][1:]) - 1   # 0-indexed
        subject_id  = int(parts[1][1:])        # 1-indexed

        split = 'train' if subject_id in TRAIN_SUBJECTS else 'test'

        # Skip splits not requested
        if split_filter == 'train_only' and split != 'train':
            continue
        if split_filter == 'test_only'  and split != 'test':
            continue

        print(f'  [{split}] {base}', end='  ', flush=True)
        poses_3d, poses_2d, rom_angles = load_vicon_file(fpath)
        T = poses_3d.shape[0]

        if is_correct:
            if has_scores_dir:
                q_score = load_quality_score(scores_dir, exercise_id, subject_id)
            else:
                q_score = derive_quality_from_incorrect(data_dir, exercise_id, subject_id)
        else:
            q_score = 0.0   # incorrect movement → quality 0

        print(f'T={T}  quality={q_score:.3f}')

        splits[split].append({
            'poses_3d':       poses_3d,
            'poses_2d':       poses_2d,
            'rom_angles':     rom_angles,
            'subject_ids':    np.full(T, subject_id - 1, dtype=np.int32),
            'exercise_ids':   np.full(T, exercise_id,    dtype=np.int32),
            'frame_ids':      np.arange(T, dtype=np.int32),
            'quality_scores': np.full(T, q_score,        dtype=np.float32),
        })

    os.makedirs(output_dir, exist_ok=True)

    for split_name, samples in splits.items():
        if not samples:
            continue   # silently skip splits that were filtered out

        all_3d  = np.concatenate([s['poses_3d']       for s in samples], axis=0)
        all_2d  = np.concatenate([s['poses_2d']       for s in samples], axis=0)
        all_rom = np.concatenate([s['rom_angles']     for s in samples], axis=0)
        all_sub = np.concatenate([s['subject_ids']    for s in samples], axis=0)
        all_exc = np.concatenate([s['exercise_ids']   for s in samples], axis=0)
        all_frm = np.concatenate([s['frame_ids']      for s in samples], axis=0)
        all_qsc = np.concatenate([s['quality_scores'] for s in samples], axis=0)

        all_3d_norm = normalize_3d(all_3d)
        all_2d_norm = normalize_2d(all_2d)

        out_path = os.path.join(output_dir, f'uiprmd_{split_name}{type_suffix}.npz')
        np.savez_compressed(
            out_path,
            poses_2d       = all_2d_norm,
            poses_3d       = all_3d_norm,
            rom_angles     = all_rom,
            subject_ids    = all_sub,
            exercise_ids   = all_exc,
            frame_ids      = all_frm,
            quality_scores = all_qsc,
        )
        n_valid = int((all_qsc >= 0).sum())
        print(f'\nSaved {split_name}: {out_path}')
        print(f'  poses_2d:      {all_2d_norm.shape}')
        print(f'  poses_3d:      {all_3d_norm.shape}')
        print(f'  rom_angles:    {all_rom.shape}')
        print(f'  subject_ids:   {all_sub.shape}  unique={np.unique(all_sub)}')
        print(f'  exercise_ids:  {all_exc.shape}  unique={np.unique(all_exc)}')
        print(f'  quality_scores:{all_qsc.shape}  valid={n_valid}/{len(all_qsc)}  '
              f'mean(valid)={all_qsc[all_qsc >= 0].mean():.3f}' if n_valid else
              f'  quality_scores:{all_qsc.shape}  valid=0/{len(all_qsc)}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='UI-PRMD dataset preprocessor')
    parser.add_argument('--data_dir',   default='data/UI-PRMD/raw',
                        help='Path to UI-PRMD raw data directory')
    parser.add_argument('--output_dir', default='data/',
                        help='Output directory for NPZ files')
    parser.add_argument('--correct_only', action='store_true',
                        help='Load only correct executions from Movements/')
    parser.add_argument('--incorrect_only', action='store_true',
                        help='Load only incorrect executions from Incorrect Movements/')
    parser.add_argument('--split', default='all',
                        choices=['all', 'train_only', 'test_only'],
                        help='Which split(s) to write (default: all)')
    args = parser.parse_args()

    if args.correct_only and args.incorrect_only:
        raise ValueError('--correct_only and --incorrect_only are mutually exclusive')

    preprocess(args.data_dir, args.output_dir,
               correct_only=args.correct_only,
               incorrect_only=args.incorrect_only,
               split_filter=args.split)
    print('\nDone.')


if __name__ == '__main__':
    main()
