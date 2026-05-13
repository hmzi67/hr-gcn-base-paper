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


class QualityScoreHead(nn.Module):
    """
    Predicts a scalar exercise quality score in [0, 1] from a sequence of ROM angles.

    Input:  (B, T, 12) — T frames of ROM angles, or (B, 12) for a single frame
    Output: (B, 1)     — quality score in [0, 1]

    Temporal mean pooling collapses T; a small MLP regresses to [0, 1] via Sigmoid.
    Designed for comparison against Deb et al. (2022, IEEE TNSRE) and
    Kourbane et al. (2025, Computers Bio & Med) on the UI-PRMD dataset.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(12, 32),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        """
        angles: (B, T, 12) or (B, 12)
        returns: (B, 1) quality score in [0, 1]
        """
        if angles.dim() == 2:
            angles = angles.unsqueeze(1)   # (B, 1, 12)
        x = angles.mean(dim=1)             # temporal mean: (B, 12)
        return self.net(x)                 # (B, 1)


class ClinicalAngleHead(nn.Module):
    """
    Lightweight MLP that maps body 3D joints (23 x 3 = 69 features)
    to 12 ROM angles in degrees.

    Output order matches JOINT_LIMIT_TENSOR_ORDER:
    [cerv_pitch, trunk_flex,
     l_sho_flex, r_sho_flex, l_sho_abd, r_sho_abd,
     l_hip, r_hip, l_knee, r_knee, l_ankle, r_ankle]
    """
    def __init__(self, in_features: int = 69, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 12),   # 12 ROM angles
        )

    def forward(self, body_joints_3d: torch.Tensor) -> torch.Tensor:
        """
        body_joints_3d: (B, 23, 3) — body joints from HR-GCN output[:23]
        returns: (B, 12) ROM angles in degrees
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

    # Shoulder flexion (0° arm at side, increases forward)
    l_upper_arm = j[:, 7] - j[:, 5]   # L_shoulder → L_elbow
    r_upper_arm = j[:, 8] - j[:, 6]
    spine_vec_norm = spine_vec / (spine_vec.norm(dim=-1, keepdim=True) + 1e-8)
    l_sho_flex = 180.0 - torch.acos(vec_cos(spine_vec, l_upper_arm).clamp(-1+1e-6, 1-1e-6)) * 180 / np.pi
    r_sho_flex = 180.0 - torch.acos(vec_cos(spine_vec, r_upper_arm).clamp(-1+1e-6, 1-1e-6)) * 180 / np.pi

    # Shoulder abduction (frontal-plane projection)
    lr_axis = j[:, 6] - j[:, 5]
    frontal_n = torch.cross(lr_axis, vertical, dim=-1)
    frontal_n = frontal_n / (frontal_n.norm(dim=-1, keepdim=True) + 1e-8)
    l_proj = l_upper_arm - (l_upper_arm * frontal_n).sum(-1, keepdim=True) * frontal_n
    r_proj = r_upper_arm - (r_upper_arm * frontal_n).sum(-1, keepdim=True) * frontal_n
    l_sho_abd = 180.0 - torch.acos(vec_cos(vertical, l_proj).clamp(-1+1e-6, 1-1e-6)) * 180 / np.pi
    r_sho_abd = 180.0 - torch.acos(vec_cos(vertical, r_proj).clamp(-1+1e-6, 1-1e-6)) * 180 / np.pi

    # Ankle dorsiflexion (90° = neutral)
    l_foot = j[:, 17] - j[:, 15]   # L_ankle → L_big_toe
    r_foot = j[:, 20] - j[:, 16]
    l_ankle = 180.0 - torch.acos(vec_cos(-l_tibia, l_foot).clamp(-1+1e-6, 1-1e-6)) * 180 / np.pi
    r_ankle = 180.0 - torch.acos(vec_cos(-r_tibia, r_foot).clamp(-1+1e-6, 1-1e-6)) * 180 / np.pi

    angles = torch.stack([
        cerv_pitch, trunk,
        l_sho_flex, r_sho_flex,
        l_sho_abd,  r_sho_abd,
        l_hip,      r_hip,
        l_knee,     r_knee,
        l_ankle,    r_ankle,
    ], dim=1)  # (B, 12)

    return angles
