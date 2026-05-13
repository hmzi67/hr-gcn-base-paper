"""
Task 1a: Metric alignment check.

Determines whether our Normalized MAD (0.0521) is comparable to
Kourbane et al. (2025) Normalized MAD (0.009).
"""
import os
import textwrap
import numpy as np

# ---------------------------------------------------------------------------
# Our normalization formula (from eval_normalized_mad.py)
# ---------------------------------------------------------------------------
ANGLE_RANGES = np.array([
    45.0,   # Cervical Pitch: ±22.5°
    60.0,   # Trunk Flex: ±30°
    180.0,  # L Shoulder Flex
    180.0,  # R Shoulder Flex
    180.0,  # L Shoulder Abd
    180.0,  # R Shoulder Abd
    120.0,  # L Hip
    120.0,  # R Hip
    140.0,  # L Knee
    140.0,  # R Knee
    60.0,   # L Ankle
    60.0,   # R Ankle
], dtype=np.float32)

JOINT_NAMES = [
    'Cervical Pitch', 'Trunk Flex',
    'L Sho Flex', 'R Sho Flex',
    'L Sho Abd', 'R Sho Abd',
    'L Hip', 'R Hip',
    'L Knee', 'R Knee',
    'L Ankle', 'R Ankle',
]

def our_normalized_mad(gt, pred):
    """Our metric: mean over joints of (|error| / range)."""
    per_joint_error = np.abs(gt - pred)
    return float(np.mean(per_joint_error / ANGLE_RANGES[:gt.shape[1]]))


# ---------------------------------------------------------------------------
# Load test NPZ
# ---------------------------------------------------------------------------
npz_path = 'data/uiprmd_test.npz'
d = np.load(npz_path, allow_pickle=True)
keys = list(d.keys())
print("=== Test NPZ keys:", keys)

has_quality = 'quality_scores' in d

lines = []
lines.append("=" * 70)
lines.append("METRIC ALIGNMENT REPORT")
lines.append("=" * 70)
lines.append("")

# ---------------------------------------------------------------------------
# Section 1: Our metric
# ---------------------------------------------------------------------------
lines.append("--- OUR METRIC ---")
lines.append("Formula: Normalized_MAD = mean( |pred_angle - gt_angle| / range )")
lines.append("")
lines.append("Per-joint ranges used for normalization:")
for name, rng in zip(JOINT_NAMES, ANGLE_RANGES):
    lines.append(f"  {name:<20} {rng:>6.1f}°")
lines.append("")
lines.append("This gives a dimensionless error fraction (0–1).")
lines.append("Example: 5° error on Cervical Pitch (45° range) = 5/45 = 0.111")
lines.append("Example: 5° error on Shoulder Flex (180° range) = 5/180 = 0.028")
lines.append("")

# ---------------------------------------------------------------------------
# Section 2: Kourbane's metric
# ---------------------------------------------------------------------------
lines.append("--- KOURBANE et al. (2025) METRIC ---")
lines.append(textwrap.dedent("""\
  Source: Kourbane et al., "Assessment of Physical Rehabilitation Exercises
  Using Graph Convolutional Networks," Computers in Biology and Medicine, 2025.

  What Kourbane measures:
    - Input: skeleton joint angles per frame
    - Target: exercise quality SCORE in [0, 1] per repetition
      (derived from GMM: Vakanski et al., 2018)
    - Prediction: quality score regressed from joint angles
    - Metric: MAD = mean(|pred_quality - gt_quality|)
      where both values are already in [0, 1] scale

  Their reported Normalized MAD = 0.009 means the model's average
  absolute error on the quality score (0–1) is 0.009 — i.e., ~0.9%.

  This is NOT the same as dividing angle errors by ROM ranges.
"""))

# ---------------------------------------------------------------------------
# Section 3: Are they the same?
# ---------------------------------------------------------------------------
lines.append("--- ARE THESE THE SAME METRIC? ---")
lines.append("ANSWER: NO — these are fundamentally different metrics.")
lines.append("")
lines.append(textwrap.dedent("""\
  Key differences:
  1. TARGET VARIABLE:
     - Ours:      ROM joint angles (degrees, 12 values per frame)
     - Kourbane:  Exercise quality score (scalar, 0–1 per repetition)

  2. NORMALIZATION DENOMINATOR:
     - Ours:      Fixed clinical ROM range (45°–180° depending on joint)
     - Kourbane:  Inherently [0,1] because quality score IS [0,1]

  3. GRANULARITY:
     - Ours:      Per-frame, per-joint angle error
     - Kourbane:  Per-repetition quality score error

  4. INTERPRETATION:
     - Our 0.0521 = average joint angle error is ~5.21% of joint ROM
     - Their 0.009 = average quality score error is 0.9% of [0,1] scale

  CONCLUSION: Direct numerical comparison is INVALID.
  The metrics measure different things at different scales.
  Our 0.0521 cannot be compared to their 0.009 as "5.8x worse".
"""))

# ---------------------------------------------------------------------------
# Section 4: Quality score check
# ---------------------------------------------------------------------------
lines.append(f"--- QUALITY SCORES IN OUR DATA ---")
lines.append(f"  'quality_scores' key present in test NPZ: {has_quality}")

if has_quality:
    qs = d['quality_scores'].astype(np.float32)
    lines.append(f"  Shape: {qs.shape}")
    lines.append(f"  Range: [{qs.min():.4f}, {qs.max():.4f}]")
    lines.append(f"  Mean:  {qs.mean():.4f}  Std: {qs.std():.4f}")
    lines.append("")
    lines.append("  NOTE: If we had quality score predictions, we could compute")
    lines.append("  MAD on [0,1] scale to directly match Kourbane's metric.")
else:
    lines.append("  NOT PRESENT — cannot compute quality-score MAD.")
    lines.append("  To match Kourbane's metric exactly, we would need:")
    lines.append("  1. Per-repetition quality scores (GMM-derived, Vakanski 2018)")
    lines.append("  2. A quality score regression head (QualityScoreHead in models/)")
    lines.append("  3. Report MAD on [0,1] quality predictions")
    lines.append("")
    lines.append("  Our current normalized MAD (0.0521) and Kourbane's (0.009)")
    lines.append("  are NOT directly comparable numbers.")

# ---------------------------------------------------------------------------
# Section 5: What a fair comparison would look like
# ---------------------------------------------------------------------------
lines.append("")
lines.append("--- FAIR COMPARISON STRATEGY ---")
lines.append(textwrap.dedent("""\
  Option A (recommended for thesis):
    Report our metric as "Normalized ROM MAE" (not Normalized MAD).
    State clearly: "We report per-joint angle accuracy normalized by
    clinical ROM ranges, yielding 0.0521. Kourbane et al. report
    quality score prediction accuracy (0.009); these are different tasks."

  Option B (apples-to-apples):
    Add quality_scores to NPZ (compute from GMM using Vakanski code),
    train QualityScoreHead, report MAD on [0,1].
    This requires the Vakanski (2018) GMM implementation.

  Option C (intermediate):
    Report our raw Mean MAE (5.51°) vs published degree-level errors
    from other pose estimation papers on UI-PRMD.
    (e.g., Liao et al. 2020, Dittakavi et al. 2021 use joint angle MAE)
"""))

lines.append("=" * 70)

report = "\n".join(lines)
print("\n" + report)

os.makedirs('results', exist_ok=True)
with open('results/metric_alignment_report.txt', 'w') as f:
    f.write(report + "\n")

print("\n[SAVED] results/metric_alignment_report.txt")
