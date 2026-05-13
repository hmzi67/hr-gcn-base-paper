"""
Novel clinical angle supervision loss.

L_total = L_MPJPE + lambda_angle * L_clinical_angle
                   + lambda_constraint * L_anatomical_constraint

This is the core novel contribution:
- L_MPJPE: standard 3D joint position error (mm) — from original HR-GCN
- L_clinical_angle: direct ROM degree error between predicted and Vicon angles
- L_anatomical_constraint: soft penalty for physically impossible poses

No prior whole-body pose estimation method combines all three.
Reference: HR-GCN (Zhang et al., 2025) uses L_MPJPE only.
"""
import torch
import torch.nn as nn

from common.anatomical_constraints import AnatomicalConstraintLoss

# Joint indices for the 12-joint ROM vector:
#   0=CervPitch  1=TrunkFlex  2=LShoFlex  3=RShoFlex
#   4=LShoAbd    5=RShoAbd    6=LHip      7=RHip
#   8=LKnee      9=RKnee     10=LAnkle   11=RAnkle

# Per-joint loss multipliers (use_joint_weights=True)
_JOINT_WEIGHTS = torch.tensor(
    [1.0, 1.0, 1.0, 2.5, 2.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    dtype=torch.float32)

# Exercise-specific per-joint multipliers (use_exercise_weights=True)
# Keys are 0-indexed exercise IDs; values map joint_idx → multiplier.
_EXERCISE_ANGLE_WEIGHTS = {
    3: {6: 2.0, 7: 2.0, 8: 2.0, 9: 2.0},   # E4 Side Lunge: LH, RH, LK, RK
    6: {4: 2.0, 5: 2.0},                      # E7 Sho Abd: LSA, RSA
    7: {3: 2.5, 5: 2.5},                      # E8 Sho Ext: RSF, RSA
    9: {4: 2.0, 5: 2.0},                      # E10 Sho Ext Rot: LSA, RSA
}


def _build_exercise_weight_table(n_joints: int = 12) -> torch.Tensor:
    """Precompute (10, n_joints) table for vectorised exercise weighting."""
    table = torch.ones(10, n_joints, dtype=torch.float32)
    for ex_id, jdict in _EXERCISE_ANGLE_WEIGHTS.items():
        for jidx, mult in jdict.items():
            if jidx < n_joints:
                table[ex_id, jidx] = mult
    return table


class ClinicalPoseLoss(nn.Module):
    def __init__(
        self,
        lambda_angle: float = 0.1,
        lambda_constraint: float = 0.05,
        lambda_body: float = 1.0,
        lambda_face: float = 0.5,
        lambda_hand: float = 0.5,
        mpjpe_lambda: float = 0.5,   # existing HR-GCN MSE/L1 blend
        use_joint_weights: bool = False,
        use_exercise_weights: bool = False,
    ):
        super().__init__()
        self.lambda_angle        = lambda_angle
        self.lambda_constraint   = lambda_constraint
        self.lambda_body         = lambda_body
        self.lambda_face         = lambda_face
        self.lambda_hand         = lambda_hand
        self.mpjpe_lambda        = mpjpe_lambda
        self.use_joint_weights   = use_joint_weights
        self.use_exercise_weights = use_exercise_weights
        self.mse                 = nn.MSELoss()
        self.l1                  = nn.L1Loss()
        self.constraint_loss     = AnatomicalConstraintLoss()

        if use_joint_weights:
            self.register_buffer('joint_weights', _JOINT_WEIGHTS.clone())
        if use_exercise_weights:
            self.register_buffer('ex_weight_table', _build_exercise_weight_table())

    def part_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Identical to original HR-GCN per-part loss."""
        return ((1 - self.mpjpe_lambda) * self.mse(pred, target)
                + self.mpjpe_lambda     * self.l1(pred, target))

    def forward(
        self,
        pred_body:      torch.Tensor,   # (B, 23, 3)
        pred_face:      torch.Tensor,   # (B, 68, 3)
        pred_lhand:     torch.Tensor,   # (B, 21, 3)
        pred_rhand:     torch.Tensor,   # (B, 21, 3)
        target_body:    torch.Tensor,   # (B, 23, 3)
        target_face:    torch.Tensor,
        target_lhand:   torch.Tensor,
        target_rhand:   torch.Tensor,
        pred_angles:    torch.Tensor,   # (B, n_joints)  from ClinicalAngleHead
        target_angles:  torch.Tensor,   # (B, n_joints)  Vicon ground truth degrees
        exercise_ids:   torch.Tensor = None,  # (B,) int64, 0-indexed exercise IDs
    ) -> dict:
        # 1. Standard position losses (preserves original HR-GCN behavior)
        L_pos = (
            self.lambda_body * self.part_loss(pred_body,  target_body)
            + self.lambda_face * self.part_loss(pred_face,  target_face)
            + self.lambda_hand * self.part_loss(pred_lhand, target_lhand)
            + self.lambda_hand * self.part_loss(pred_rhand, target_rhand)
        )

        # 2. Novel: clinical angle supervision with optional per-joint / per-exercise weights
        n_joints = pred_angles.shape[1]
        per_sample = (pred_angles - target_angles).abs()   # (B, n_joints)

        if self.use_joint_weights or self.use_exercise_weights:
            w = torch.ones_like(per_sample)   # (B, n_joints)
            if self.use_joint_weights:
                w = w * self.joint_weights[:n_joints].unsqueeze(0)
            if self.use_exercise_weights and exercise_ids is not None:
                ex_w = self.ex_weight_table[exercise_ids, :n_joints]  # (B, n_joints)
                w = w * ex_w
            L_angle = (per_sample * w).mean()
        else:
            L_angle = per_sample.mean()

        # 3. Novel: anatomical constraint penalty
        L_constraint = self.constraint_loss(pred_angles)

        total = (L_pos
                 + self.lambda_angle      * L_angle
                 + self.lambda_constraint * L_constraint)

        return {
            'total':        total,
            'L_pos':        L_pos.item(),
            'L_angle':      L_angle.item(),
            'L_constraint': L_constraint.item(),
        }
