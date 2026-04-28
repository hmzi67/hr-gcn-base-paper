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
    to 6 ROM angles in degrees.

    Output order matches JOINT_LIMIT_TENSOR_ORDER:
    [cerv_pitch, trunk_flex, l_hip, r_hip, l_knee, r_knee]
    """
    def __init__(self, in_features: int = 69, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 6),   # 6 ROM angles
        )

    def forward(self, body_joints_3d: torch.Tensor) -> torch.Tensor:
        """
        body_joints_3d: (B, 23, 3) — body joints from HR-GCN output[:23]
        returns: (B, 6) ROM angles in degrees
        """
        B = body_joints_3d.shape[0]
        x = body_joints_3d.reshape(B, -1)   # (B, 69)
        return self.net(x)


def compute_rom_angles_geometric(joints_3d: torch.Tensor) -> torch.Tensor:
    """
    Geometric fallback: compute ROM angles from 3D joint positions
    using vector algebra. Used for validation / comparison.

    joints_3d: (B, 133, 3) full body output from HR-GCN
    returns: (B, 6) angles in degrees

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

    mid_hip = (j[:, 11] + j[:, 12]) / 2
    mid_sho = (j[:, 5]  + j[:, 6])  / 2
    spine_vec = mid_sho - mid_hip  # points upward (Z-up, matching Vicon/UI-PRMD)
    # Z is the vertical axis in both Vicon and the COCO-mapped 3D poses
    vertical = torch.zeros_like(spine_vec)
    vertical[:, 2] = 1.0   # Z-up

    def vec_cos(a, b):
        return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1) + 1e-8)

    trunk = torch.acos(vec_cos(spine_vec, vertical).clamp(-1 + 1e-6, 1 - 1e-6)) * 180 / np.pi

    # Cervical: angle between head direction and vertical
    head_vec = j[:, 0] - mid_sho
    cerv_pitch = torch.acos(vec_cos(head_vec, vertical).clamp(-1 + 1e-6, 1 - 1e-6)) * 180 / np.pi

    # Hip flexion (clinical: 0° upright, increases with flexion)
    # angle(up, femur) is ~180° when upright; clinical = 180 - that
    l_femur = j[:, 13] - j[:, 11]   # L_hip → L_knee
    r_femur = j[:, 14] - j[:, 12]
    l_hip = 180.0 - torch.acos(vec_cos(vertical, l_femur).clamp(-1 + 1e-6, 1 - 1e-6)) * 180 / np.pi
    r_hip = 180.0 - torch.acos(vec_cos(vertical, r_femur).clamp(-1 + 1e-6, 1 - 1e-6)) * 180 / np.pi

    # Knee flexion (clinical: 0° straight, increases with bend)
    # angle_at_vertex(hip, knee, ankle) is ~180° when straight; clinical = 180 - that
    l_tibia = j[:, 15] - j[:, 13]
    r_tibia = j[:, 16] - j[:, 14]
    l_knee = 180.0 - torch.acos(vec_cos(-l_femur, l_tibia).clamp(-1 + 1e-6, 1 - 1e-6)) * 180 / np.pi
    r_knee = 180.0 - torch.acos(vec_cos(-r_femur, r_tibia).clamp(-1 + 1e-6, 1 - 1e-6)) * 180 / np.pi

    angles = torch.stack([
        cerv_pitch,
        trunk,
        l_hip, r_hip,
        l_knee, r_knee,
    ], dim=1)  # (B, 6)

    return angles
