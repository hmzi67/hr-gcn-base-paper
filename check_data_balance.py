"""
Task 1b: Data balance check.

Checks frame counts per exercise and subject, and shoulder vs lower body ratio.
Also checks L vs R shoulder angle distribution consistency.
"""
import os
import numpy as np

EXERCISE_NAMES = [f'E{i+1}' for i in range(10)]
EXERCISE_DESCRIPTIONS = {
    0: 'Deep Squat',
    1: 'Hurdle Step',
    2: 'Inline Lunge',
    3: 'Side Lunge',
    4: 'Sit to Stand',
    5: 'Standing Active Straight Leg Raise',
    6: 'Standing Shoulder Abduction',
    7: 'Shoulder Extension',
    8: 'Standing Shoulder Internal Rotation',
    9: 'Standing Shoulder External Rotation',
}
JOINT_NAMES = [
    'Cerv Pitch', 'Trunk Flex',
    'L Sho Flex', 'R Sho Flex',
    'L Sho Abd',  'R Sho Abd',
    'L Hip',      'R Hip',
    'L Knee',     'R Knee',
    'L Ankle',    'R Ankle',
]

d = np.load('data/uiprmd_train.npz', allow_pickle=True)
exercise_ids = d['exercise_ids']
subject_ids  = d['subject_ids']
rom_angles   = d['rom_angles'].astype(np.float32)

print("=== Train NPZ keys:", list(d.keys()))
print(f"Total train frames: {len(exercise_ids)}")

lines = []
lines.append("=" * 70)
lines.append("DATA BALANCE REPORT — uiprmd_train.npz")
lines.append("=" * 70)
lines.append(f"Total frames: {len(exercise_ids)}")
lines.append("")

# ---------------------------------------------------------------------------
# Frames per exercise
# ---------------------------------------------------------------------------
lines.append("--- FRAMES PER EXERCISE ---")
shoulder_frames = 0
lower_body_frames = 0
shoulder_exercises = {6, 7, 8, 9}   # E7–E10 (0-indexed 6–9)
lower_body_exercises = {0, 1, 2, 3, 4, 5}

for ex in range(10):
    mask = exercise_ids == ex
    n = mask.sum()
    desc = EXERCISE_DESCRIPTIONS.get(ex, '')
    tag = ''
    if ex in shoulder_exercises:
        shoulder_frames += n
        tag = ' [SHOULDER]'
    else:
        lower_body_frames += n
        tag = ' [LOWER/TRUNK]'
    lines.append(f"  {EXERCISE_NAMES[ex]} ({desc:<40}) : {n:>5} frames{tag}")

lines.append("")
lines.append(f"  Shoulder exercises  (E7-E10): {shoulder_frames:>5} frames  "
             f"({100*shoulder_frames/len(exercise_ids):.1f}%)")
lines.append(f"  Lower/Trunk exs (E1-E6): {lower_body_frames:>5} frames  "
             f"({100*lower_body_frames/len(exercise_ids):.1f}%)")
lines.append(f"  Ratio shoulder/lower:  {shoulder_frames/max(lower_body_frames,1):.3f}")
lines.append("")

# ---------------------------------------------------------------------------
# Frames per subject
# ---------------------------------------------------------------------------
lines.append("--- FRAMES PER SUBJECT ---")
unique_subjects = np.unique(subject_ids)
for s in sorted(unique_subjects):
    n = (subject_ids == s).sum()
    lines.append(f"  Subject {s+1:02d} (S{s+1:02d}): {n:>5} frames")
lines.append("")
lines.append(f"  (Train subjects: {sorted(list(unique_subjects+1))})")

lines.append(f"  (Test subjects: S9, S10 — 0-indexed: 8, 9)")
lines.append("")

# ---------------------------------------------------------------------------
# L vs R shoulder annotation consistency
# ---------------------------------------------------------------------------
lines.append("--- L vs R SHOULDER ANGLE STATISTICS (train) ---")
# Indices: L_Sho_Abd=4, R_Sho_Abd=5, L_Sho_Flex=2, R_Sho_Flex=3
l_abd  = rom_angles[:, 4]
r_abd  = rom_angles[:, 5]
l_flex = rom_angles[:, 2]
r_flex = rom_angles[:, 3]

lines.append(f"  L Sho Abd  — mean={l_abd.mean():.2f}°  std={l_abd.std():.2f}°  "
             f"min={l_abd.min():.2f}°  max={l_abd.max():.2f}°")
lines.append(f"  R Sho Abd  — mean={r_abd.mean():.2f}°  std={r_abd.std():.2f}°  "
             f"min={r_abd.min():.2f}°  max={r_abd.max():.2f}°")
lines.append(f"  L Sho Flex — mean={l_flex.mean():.2f}°  std={l_flex.std():.2f}°  "
             f"min={l_flex.min():.2f}°  max={l_flex.max():.2f}°")
lines.append(f"  R Sho Flex — mean={r_flex.mean():.2f}°  std={r_flex.std():.2f}°  "
             f"min={r_flex.min():.2f}°  max={r_flex.max():.2f}°")
lines.append("")

# Within shoulder exercises only
for ex in sorted(shoulder_exercises):
    mask = exercise_ids == ex
    if mask.sum() == 0:
        continue
    la = rom_angles[mask, 4]
    ra = rom_angles[mask, 5]
    lines.append(f"  {EXERCISE_NAMES[ex]} ({EXERCISE_DESCRIPTIONS[ex]}):")
    lines.append(f"    L Sho Abd  mean={la.mean():.2f}°  std={la.std():.2f}°")
    lines.append(f"    R Sho Abd  mean={ra.mean():.2f}°  std={ra.std():.2f}°")
    diff = abs(la.mean() - ra.mean())
    lines.append(f"    |L-R| mean diff = {diff:.2f}° {'<-- ASYMMETRY' if diff > 10 else ''}")
lines.append("")

# ---------------------------------------------------------------------------
# NaN / zero check
# ---------------------------------------------------------------------------
lines.append("--- ANNOTATION QUALITY ---")
n_nan   = np.isnan(rom_angles).sum()
n_zero_rows = (np.abs(rom_angles) < 1e-6).all(axis=1).sum()
lines.append(f"  NaN values in rom_angles: {n_nan}")
lines.append(f"  Frames where ALL angles ≈ 0: {n_zero_rows}")
lines.append("")

# Per-joint NaN
for i, name in enumerate(JOINT_NAMES):
    n_nan_j = np.isnan(rom_angles[:, i]).sum()
    if n_nan_j > 0:
        lines.append(f"  Joint {name}: {n_nan_j} NaN frames")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
lines.append("")
lines.append("--- IMBALANCE SUMMARY ---")
if shoulder_frames < lower_body_frames:
    lines.append(f"  UNDERREPRESENTED: shoulder exercises have "
                 f"{lower_body_frames - shoulder_frames} fewer frames than lower body.")
    lines.append(f"  This contributes to higher shoulder angle errors.")
else:
    lines.append(f"  Shoulder exercises are NOT underrepresented vs lower body.")

r_greater_than_l = r_abd.mean() > l_abd.mean()
lines.append(f"  R Sho Abd mean ({r_abd.mean():.1f}°) vs L Sho Abd mean ({l_abd.mean():.1f}°): "
             f"{'R > L' if r_greater_than_l else 'L > R'}")
if r_greater_than_l:
    lines.append("  Right shoulder has LARGER mean abduction angle — harder to regress.")

lines.append("=" * 70)

report = "\n".join(lines)
print("\n" + report)
os.makedirs('results', exist_ok=True)
with open('results/data_balance_report.txt', 'w') as f:
    f.write(report + "\n")
print("\n[SAVED] results/data_balance_report.txt")
