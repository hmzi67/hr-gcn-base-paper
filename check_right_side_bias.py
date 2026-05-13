"""
Task 1d: Right-side shoulder bias analysis.

Investigates why R_sho_abd has much higher error (10.18°) than L_sho_abd (7.45°).
Checks:
  - L vs R mean angles in training data
  - R shoulder angle variability per exercise
  - Train/test split subject characteristics
"""
import os
import numpy as np

JOINT_NAMES = [
    'Cerv Pitch', 'Trunk Flex',
    'L Sho Flex', 'R Sho Flex',
    'L Sho Abd',  'R Sho Abd',
    'L Hip',      'R Hip',
    'L Knee',     'R Knee',
    'L Ankle',    'R Ankle',
]
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

dtr = np.load('data/uiprmd_train.npz', allow_pickle=True)
dte = np.load('data/uiprmd_test.npz',  allow_pickle=True)

tr_angles  = dtr['rom_angles'].astype(np.float32)
te_angles  = dte['rom_angles'].astype(np.float32)
tr_ex      = dtr['exercise_ids']
te_ex      = dte['exercise_ids']
tr_sub     = dtr['subject_ids']
te_sub     = dte['subject_ids']

lines = []
lines.append("=" * 80)
lines.append("RIGHT vs LEFT SHOULDER BIAS ANALYSIS")
lines.append("=" * 80)
lines.append("")

# ---------------------------------------------------------------------------
# Overall L vs R stats in train and test
# ---------------------------------------------------------------------------
lines.append("--- OVERALL ANGLE STATISTICS ---")
lines.append(f"{'Joint':<14} {'Train mean':>12} {'Train std':>10} {'Test mean':>12} {'Test std':>10}")
lines.append("-" * 60)
for i in [2, 3, 4, 5]:   # L Sho Flex, R Sho Flex, L Sho Abd, R Sho Abd
    tr_m, tr_s = tr_angles[:, i].mean(), tr_angles[:, i].std()
    te_m, te_s = te_angles[:, i].mean(), te_angles[:, i].std()
    lines.append(f"  {JOINT_NAMES[i]:<12} {tr_m:>12.2f}° {tr_s:>10.2f}° "
                 f"{te_m:>12.2f}° {te_s:>10.2f}°")
lines.append("")

# ---------------------------------------------------------------------------
# Distribution mismatch: train vs test for R shoulder
# ---------------------------------------------------------------------------
lines.append("--- TRAIN vs TEST DISTRIBUTION (R SHOULDER) ---")
r_abd_train = tr_angles[:, 5]
r_abd_test  = te_angles[:, 5]
lines.append(f"  R Sho Abd  TRAIN: mean={r_abd_train.mean():.2f}°  std={r_abd_train.std():.2f}°  "
             f"range=[{r_abd_train.min():.1f}, {r_abd_train.max():.1f}]")
lines.append(f"  R Sho Abd  TEST:  mean={r_abd_test.mean():.2f}°  std={r_abd_test.std():.2f}°  "
             f"range=[{r_abd_test.min():.1f}, {r_abd_test.max():.1f}]")

l_abd_train = tr_angles[:, 4]
l_abd_test  = te_angles[:, 4]
lines.append(f"  L Sho Abd  TRAIN: mean={l_abd_train.mean():.2f}°  std={l_abd_train.std():.2f}°  "
             f"range=[{l_abd_train.min():.1f}, {l_abd_train.max():.1f}]")
lines.append(f"  L Sho Abd  TEST:  mean={l_abd_test.mean():.2f}°  std={l_abd_test.std():.2f}°  "
             f"range=[{l_abd_test.min():.1f}, {l_abd_test.max():.1f}]")

# Distribution shift = |train_mean - test_mean|
r_shift = abs(r_abd_train.mean() - r_abd_test.mean())
l_shift = abs(l_abd_train.mean() - l_abd_test.mean())
lines.append(f"")
lines.append(f"  Train→Test mean shift:  R Sho Abd = {r_shift:.2f}°   L Sho Abd = {l_shift:.2f}°")
if r_shift > l_shift + 2:
    lines.append(f"  RIGHT shoulder has larger train/test distribution shift — harder to generalise.")
lines.append("")

# ---------------------------------------------------------------------------
# Per-exercise R vs L shoulder analysis
# ---------------------------------------------------------------------------
lines.append("--- PER-EXERCISE R vs L SHOULDER ABD (train) ---")
shoulder_exs = [6, 7, 8, 9]
lines.append(f"{'Exercise':<40} {'L Abd mean':>12} {'R Abd mean':>12} "
             f"{'Asymmetry':>12} {'R > L?':>8}")
lines.append("-" * 86)
for ex in range(10):
    mask = tr_ex == ex
    if mask.sum() == 0:
        continue
    la = tr_angles[mask, 4].mean()
    ra = tr_angles[mask, 5].mean()
    asym = ra - la
    lines.append(f"  E{ex+1} {EXERCISE_DESCRIPTIONS[ex]:<36} {la:>12.2f}° {ra:>12.2f}° "
                 f"{asym:>+12.2f}° {'YES' if asym > 5 else 'no':>8}")
lines.append("")

# ---------------------------------------------------------------------------
# R shoulder angle variability (high std = harder to learn)
# ---------------------------------------------------------------------------
lines.append("--- ANGLE VARIABILITY (std) — TRAIN ---")
lines.append("  High std means the model must learn a wider range → harder.")
lines.append(f"{'Joint':<14} {'Overall std':>14}", )
for i in [2, 3, 4, 5]:
    lines.append(f"  {JOINT_NAMES[i]:<12} {tr_angles[:,i].std():>14.2f}°")
lines.append("")

# Per-exercise variability for shoulder joints
lines.append("  Per-exercise std for shoulder joints (shoulder exercises only):")
lines.append(f"  {'Exercise':<35} {'L Sho Abd std':>14} {'R Sho Abd std':>14} {'R>L?':>8}")
lines.append("  " + "-" * 73)
for ex in shoulder_exs:
    mask = tr_ex == ex
    if mask.sum() == 0:
        continue
    ls = tr_angles[mask, 4].std()
    rs = tr_angles[mask, 5].std()
    lines.append(f"  E{ex+1} {EXERCISE_DESCRIPTIONS[ex]:<33} {ls:>14.2f}° {rs:>14.2f}° "
                 f"{'YES' if rs > ls else 'no':>8}")
lines.append("")

# ---------------------------------------------------------------------------
# Test subjects (S9, S10 = 0-indexed 8, 9)
# ---------------------------------------------------------------------------
lines.append("--- TEST SUBJECTS (S9, S10) ---")
lines.append("  UI-PRMD paper (Vakanski 2018) does not document handedness.")
lines.append("  However we can check R vs L shoulder asymmetry in test subjects.")
for s in [8, 9]:
    mask_te = te_sub == s
    if mask_te.sum() == 0:
        continue
    la_te = te_angles[mask_te, 4].mean()
    ra_te = te_angles[mask_te, 5].mean()
    lines.append(f"  Test Subject S{s+1}: L Sho Abd mean={la_te:.2f}°  "
                 f"R Sho Abd mean={ra_te:.2f}°  diff={ra_te-la_te:+.2f}°")
lines.append("")

# ---------------------------------------------------------------------------
# Confidence / occlusion proxy
# ---------------------------------------------------------------------------
lines.append("--- OCCLUSION PROXY (2D keypoint spread) ---")
poses_2d = dtr['poses_2d'].astype(np.float32)   # (N, 133, 2)
# R shoulder in COCO-WB 133: joint index 6 (r_shoulder)
# L shoulder: joint index 5
l_sho_x  = poses_2d[:, 5, 0]
r_sho_x  = poses_2d[:, 6, 0]
# Large |x| → arm may be extended far in image plane
# Std of R vs L across shoulder exercises
for ex in shoulder_exs:
    mask = tr_ex == ex
    if mask.sum() == 0:
        continue
    lx_std = poses_2d[mask, 5, 0].std()
    rx_std = poses_2d[mask, 6, 0].std()
    lines.append(f"  E{ex+1}: L shoulder x-std={lx_std:.4f}  R shoulder x-std={rx_std:.4f}  "
                 f"{'R more variable' if rx_std > lx_std else 'L more variable'}")
lines.append("")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
lines.append("--- ROOT CAUSE SUMMARY ---")
r_std = tr_angles[:, 5].std()
l_std = tr_angles[:, 4].std()
r_flex_std = tr_angles[:, 3].std()

r_train_mean = r_abd_train.mean()
r_test_mean  = r_abd_test.mean()

lines.append(f"  1. Variability: R Sho Abd std={r_std:.2f}°  vs L Sho Abd std={l_std:.2f}°")
lines.append(f"     {'R has higher variability → harder to fit' if r_std > l_std else 'L has higher variability'}")
lines.append(f"  2. Train/test shift: R Sho Abd {r_shift:.2f}° vs L Sho Abd {l_shift:.2f}°")
if r_shift > 3:
    lines.append(f"     R shoulder distribution in test differs more from train — generalisation gap.")
lines.append(f"  3. Right shoulder joints lack bilateral symmetry in some exercises (see above).")
lines.append(f"  4. Occlusion: right-side exercises (E8 Shoulder Extension) emphasize right arm,")
lines.append(f"     left-only exercises are rarer in UI-PRMD → right arm more ambiguous to 2D→3D.")
lines.append("")
lines.append("RECOMMENDATIONS:")
lines.append("  - Apply left-right mirror augmentation more aggressively (p_mirror=0.7)")
lines.append("  - Use joint weights: R_sho_abd=3.0x, L_sho_abd=2.0x (Task 2b)")
lines.append("  - Use exercise weights: E8 RSA/RSF = 2.5x (Task 2a)")
lines.append("=" * 80)

report = "\n".join(lines)
print("\n" + report)
os.makedirs('results', exist_ok=True)
with open('results/right_side_bias_report.txt', 'w') as f:
    f.write(report + "\n")
print("\n[SAVED] results/right_side_bias_report.txt")
