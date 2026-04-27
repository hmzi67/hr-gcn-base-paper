"""
Clinical ROM angle regression head.
Takes body joint 3D positions from HR-GCN output and regresses
clinical joint angles in degrees.

Novel: Directly supervised on clinical ROM degrees during training.
No prior whole-body pose method does this.
"""
import numpy as np
import torch
import torch.nn as nn

from common.anatomical_constraints import clamp_angles_to_valid_range


class ClinicalAngleHead(nn.Module):
    """
    Lightweight MLP that maps body 3D joints (23 x 3 = 69 features)
    to 8 ROM angles in degrees.

    Output order matches JOINT_LIMIT_TENSOR_ORDER:
    [cerv_yaw, cerv_pitch, cerv_roll,
     trunk_flex, l_hip, r_hip, l_knee, r_knee]
    """
    def __init__(self, in_features: int = 69, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 8),   # 8 ROM angles
        )

    def forward(self, body_joints_3d: torch.Tensor) -> torch.Tensor:
        """
        body_joints_3d: (B, 23, 3) — body joints from HR-GCN output[:23]
        returns: (B, 8) ROM angles in degrees
        """
        B = body_joints_3d.shape[0]
        x = body_joints_3d.reshape(B, -1)   # (B, 69)
        return self.net(x)


def compute_rom_angles_geometric(joints_3d: torch.Tensor) -> torch.Tensor:
    """
    Geometric fallback: compute ROM angles from 3D joint positions
    using vector algebra. Used for validation / comparison.

    joints_3d: (B, 133, 3) full body output from HR-GCN
    returns: (B, 8) angles in degrees

    COCO body joint indices used:
        0=nose, 5=Lshoulder, 6=Rshoulder, 11=Lhip, 12=Rhip
        13=Lknee, 14=Rknee, 15=Lankle, 16=Rankle
    """
    def angle_3d(a, b, c):
        """Angle at vertex b, vectors b->a and b->c."""
        v1 = a - b
        v2 = c - b
        cos = (v1 * v2).sum(-1) / (
            v1.norm(dim=-1) * v2.norm(dim=-1) + 1e-8
        )
        return torch.acos(cos.clamp(-1 + 1e-6, 1 - 1e-6)) * 180 / np.pi

    B = joints_3d.shape[0]
    j = joints_3d  # (B, 133, 3)

    # Spine vector for trunk flex: mid-shoulder to mid-hip
    mid_hip = (j[:, 11] + j[:, 12]) / 2
    mid_sho = (j[:, 5]  + j[:, 6])  / 2
    spine_vec = mid_sho - mid_hip
    vertical = torch.zeros_like(spine_vec)
    vertical[:, 1] = 1.0   # Y-up
    trunk = torch.acos(
        (spine_vec * vertical).sum(-1) /
        (spine_vec.norm(dim=-1) * vertical.norm(dim=-1) + 1e-8)
    ).clamp(0) * 180 / np.pi

    l_hip  = angle_3d(j[:, 5],  j[:, 11], j[:, 13])
    r_hip  = angle_3d(j[:, 6],  j[:, 12], j[:, 14])
    l_knee = angle_3d(j[:, 11], j[:, 13], j[:, 15])
    r_knee = angle_3d(j[:, 12], j[:, 14], j[:, 16])

    # Cervical: use nose (0), neck approx as mid-shoulder
    neck = mid_sho
    head_vec = j[:, 0] - neck
    cerv_pitch = torch.acos(
        (head_vec * vertical).sum(-1) /
        (head_vec.norm(dim=-1) + 1e-8)
    ).clamp(0) * 180 / np.pi

    # Yaw and roll: approximated as zero for geometric method
    zeroes = torch.zeros(B, device=joints_3d.device)

    angles = torch.stack([
        zeroes,      # cerv_yaw (needs multi-frame temporal data)
        cerv_pitch,
        zeroes,      # cerv_roll
        trunk,
        l_hip, r_hip,
        l_knee, r_knee,
    ], dim=1)  # (B, 8)

    return angles
