"""
Anatomical joint angle constraints for rehabilitation pose estimation.
Applied as a soft penalty during training and hard clamp at inference.

Novel contribution: No prior whole-body pose method enforces rehabilitation-
specific anatomical angle limits in the loss function.
"""
import torch
import torch.nn as nn
from typing import Dict, Tuple


# Valid ROM ranges (degrees) based on clinical literature
# Format: (min_degrees, max_degrees)
REHAB_JOINT_LIMITS: Dict[str, Tuple[float, float]] = {
    'cervical_yaw':   (-80.0,  80.0),   # L/R neck rotation
    'cervical_pitch': (-60.0,  60.0),   # fwd/back neck bend
    'cervical_roll':  (-45.0,  45.0),   # lateral neck tilt
    'trunk_flex':     (-90.0,  90.0),   # trunk flexion/extension
    'left_hip':       (-30.0, 120.0),   # hip flexion
    'right_hip':      (-30.0, 120.0),
    'left_knee':      (  0.0, 150.0),   # knee flexion only
    'right_knee':     (  0.0, 150.0),
}

JOINT_LIMIT_TENSOR_ORDER = [
    'cervical_yaw', 'cervical_pitch', 'cervical_roll',
    'trunk_flex', 'left_hip', 'right_hip',
    'left_knee', 'right_knee',
]


def build_limit_tensors(device: torch.device):
    """Return (min_vals, max_vals) tensors of shape (8,)."""
    mins = torch.tensor(
        [REHAB_JOINT_LIMITS[k][0] for k in JOINT_LIMIT_TENSOR_ORDER],
        dtype=torch.float32, device=device,
    )
    maxs = torch.tensor(
        [REHAB_JOINT_LIMITS[k][1] for k in JOINT_LIMIT_TENSOR_ORDER],
        dtype=torch.float32, device=device,
    )
    return mins, maxs


class AnatomicalConstraintLoss(nn.Module):
    """
    Soft penalty when predicted angles fall outside valid ROM ranges.
    Loss is zero inside valid range, increases quadratically outside.

    Shape: pred_angles (B, 8) in degrees
    """
    def __init__(self):
        super().__init__()
        mins = [REHAB_JOINT_LIMITS[k][0] for k in JOINT_LIMIT_TENSOR_ORDER]
        maxs = [REHAB_JOINT_LIMITS[k][1] for k in JOINT_LIMIT_TENSOR_ORDER]
        self.register_buffer('mins', torch.tensor(mins, dtype=torch.float32))
        self.register_buffer('maxs', torch.tensor(maxs, dtype=torch.float32))

    def forward(self, pred_angles: torch.Tensor) -> torch.Tensor:
        """pred_angles: (B, 8) predicted ROM angles in degrees."""
        below = torch.clamp(self.mins - pred_angles, min=0.0)
        above = torch.clamp(pred_angles - self.maxs, min=0.0)
        violation = below + above   # (B, 8), zero when within range
        return (violation ** 2).mean()


def clamp_angles_to_valid_range(angles: torch.Tensor) -> torch.Tensor:
    """
    Hard clamp at inference time. angles: (B, 8) or (8,).
    Returns same shape with values clamped to valid ROM ranges.
    """
    mins = torch.tensor(
        [REHAB_JOINT_LIMITS[k][0] for k in JOINT_LIMIT_TENSOR_ORDER],
        dtype=angles.dtype, device=angles.device,
    )
    maxs = torch.tensor(
        [REHAB_JOINT_LIMITS[k][1] for k in JOINT_LIMIT_TENSOR_ORDER],
        dtype=angles.dtype, device=angles.device,
    )
    return torch.clamp(angles, min=mins, max=maxs)
