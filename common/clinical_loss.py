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


class ClinicalPoseLoss(nn.Module):
    def __init__(
        self,
        lambda_angle: float = 0.1,
        lambda_constraint: float = 0.05,
        lambda_body: float = 1.0,
        lambda_face: float = 0.5,
        lambda_hand: float = 0.5,
        mpjpe_lambda: float = 0.5,   # existing HR-GCN MSE/L1 blend
    ):
        super().__init__()
        self.lambda_angle      = lambda_angle
        self.lambda_constraint = lambda_constraint
        self.lambda_body       = lambda_body
        self.lambda_face       = lambda_face
        self.lambda_hand       = lambda_hand
        self.mpjpe_lambda      = mpjpe_lambda
        self.mse              = nn.MSELoss()
        self.l1               = nn.L1Loss()
        self.constraint_loss  = AnatomicalConstraintLoss()

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
        pred_angles:    torch.Tensor,   # (B, 6)  from ClinicalAngleHead
        target_angles:  torch.Tensor,   # (B, 6)  Vicon ground truth degrees
    ) -> dict:
        # 1. Standard position losses (preserves original HR-GCN behavior)
        L_pos = (
            self.lambda_body * self.part_loss(pred_body,  target_body)
            + self.lambda_face * self.part_loss(pred_face,  target_face)
            + self.lambda_hand * self.part_loss(pred_lhand, target_lhand)
            + self.lambda_hand * self.part_loss(pred_rhand, target_rhand)
        )

        # 2. Novel: clinical angle supervision (degrees)
        L_angle = self.l1(pred_angles, target_angles)

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
