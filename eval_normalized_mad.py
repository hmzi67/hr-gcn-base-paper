"""
Per-exercise evaluation with NORMALIZED MAD (matching paper metric).

Computes both raw (degrees) and normalized (0-1) MAD by dividing by angle ranges.
This allows fair comparison with SOTA paper results.

UI-PRMD angle ranges (clinical limits in degrees):
  Cervical Pitch: ±22.5° (total 45°)
  Trunk Flexion:  ±30°   (total 60°)
  Shoulder Flex:  0-180° (total 180°)
  Shoulder Abd:   0-180° (total 180°)
  Hip:            0-120° (total 120°)
  Knee:           0-140° (total 140°)
  Ankle:          ±30°   (total 60°)

Example:
    python eval_normalized_mad.py \\
        --checkpoint checkpoint_rehab_baseline_scratch/ckpt_best_rehab.pth.tar \\
        --cfg w32_adam_lr1e-3.yaml \\
        --data_test data/uiprmd_test.npz \\
        --save_csv eval_normalized_mad_results.csv
"""
from __future__ import print_function, absolute_import, division

import argparse
import csv
import os.path as path
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from lib.config import cfg
from common.graph_utils import adj_mx_from_skeleton
from common.loss import mpjpe
from utils.prepare_data_h3wb import Human3WBDataset

import models.graph_hrnet_multi_branch as ghrmb
import models.graph_resnet as GraphRes
import models.graph_hrnet as ghr
from models.graph_sh import GraphSH


JOINT_NAMES = [
    'Cervical Pitch',  'Trunk Flex',
    'L Sho Flex',      'R Sho Flex',
    'L Sho Abd',       'R Sho Abd',
    'L Hip',           'R Hip',
    'L Knee',          'R Knee',
    'L Ankle',         'R Ankle',
]

# ROM angle ranges (in degrees) — used for normalization
# Based on clinical normal limits and Vicon data ranges in UI-PRMD
ANGLE_RANGES = np.array([
    45.0,    # Cervical Pitch: ±22.5°
    60.0,    # Trunk Flex: ±30°
    180.0,   # L Shoulder Flex
    180.0,   # R Shoulder Flex
    180.0,   # L Shoulder Abd
    180.0,   # R Shoulder Abd
    120.0,   # L Hip
    120.0,   # R Hip
    140.0,   # L Knee
    140.0,   # R Knee
    60.0,    # L Ankle (dorsi+plantarflex)
    60.0,    # R Ankle
], dtype=np.float32)

EXERCISE_NAMES = [f'E{i+1}' for i in range(10)]


def parse_args():
    parser = argparse.ArgumentParser(description='Normalized MAD evaluation')
    parser.add_argument('--checkpoint', required=True, type=str)
    parser.add_argument('--cfg', default='w32_adam_lr1e-3.yaml', type=str)
    parser.add_argument('--gcn', default='dc_preagg', type=str)
    parser.add_argument('--model', default=1, type=int)
    parser.add_argument('--data_test', default='data/uiprmd_test.npz', type=str)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--save_csv', default='', type=str)
    return parser.parse_args()


class UIRPMDDatasetWithMeta(TensorDataset):
    def __init__(self, npz_path: str):
        d = np.load(npz_path, allow_pickle=True)
        self.poses_2d   = torch.from_numpy(d['poses_2d']).float()
        self.poses_3d   = torch.from_numpy(d['poses_3d']).float()
        self.rom_angles = torch.from_numpy(d['rom_angles']).float()
        self.exercise_ids = torch.from_numpy(d['exercise_ids']).long()
        super().__init__(self.poses_2d, self.poses_3d, self.rom_angles,
                         self.exercise_ids)


def build_backbone(args, cfg, adj, device):
    p_dropout = None
    skeleton = Human3WBDataset('data/h3wb_train.npz',
                               'data/h3wb_test.npz').skeleton()
    if args.model == 1:
        model = ghrmb.get_pose_net(cfg, True, adj, p_dropout, args.gcn,
                                   skeleton.joints_group())
    elif args.model == 2:
        model = GraphRes.get_pose_net(True, adj, p_dropout, args.gcn, 50, True)
    elif args.model == 3:
        model = ghr.get_pose_net(cfg, True, adj, p_dropout, args.gcn,
                                 skeleton.joints_group())
    elif args.model == 4:
        model = GraphSH(adj, 64, skeleton.joints_group(), num_layers=4,
                        p_dropout=p_dropout, gcn_type=args.gcn)
    else:
        raise ValueError(f'Unknown model: {args.model}')
    return model.to(device)


class _AngleHead(nn.Module):
    def __init__(self, in_features: int, hidden: int, n_joints: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, n_joints),
        )

    def forward(self, x):
        return self.net(x)


def main():
    args = parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
    device = torch.device('cuda:0' if args.device == 'cuda' else 'cpu')

    if not path.isfile(args.checkpoint):
        raise FileNotFoundError(f'Checkpoint not found: {args.checkpoint}')

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    epoch = ckpt.get('epoch', '?')
    best_mae = ckpt.get('best_mae')
    print(f'==> Loaded checkpoint: {args.checkpoint}')
    print(f'    epoch={epoch}  best_mae={best_mae:.4f}°')

    angle_state = ckpt.get('angle_head_state_dict')
    if angle_state is None:
        raise KeyError('Missing angle_head_state_dict')
    hidden = angle_state['net.0.weight'].shape[0]
    n_joints = angle_state['net.5.weight'].shape[0]
    print(f'    angle_head: hidden={hidden}, n_joints={n_joints}')

    cfg.merge_from_file(args.cfg)
    skeleton = Human3WBDataset('data/h3wb_train.npz',
                               'data/h3wb_test.npz').skeleton()
    adj = adj_mx_from_skeleton(skeleton).to(device)

    print('==> Building model...')
    model = build_backbone(args, cfg, adj, device)
    angle_head = _AngleHead(in_features=69, hidden=hidden,
                           n_joints=n_joints).to(device)

    model.load_state_dict(ckpt['state_dict'], strict=False)
    angle_head.load_state_dict(angle_state)

    print(f'==> Loading test data: {args.data_test}')
    test_set = UIRPMDDatasetWithMeta(args.data_test)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers,
                             pin_memory=(args.device == 'cuda'))
    print(f'    Test frames: {len(test_set)}')

    model.eval()
    angle_head.eval()

    # Store predictions and ground truth per exercise
    per_exercise_pred = {i: [] for i in range(10)}
    per_exercise_gt = {i: [] for i in range(10)}
    body_mpjpe_sum = 0.0
    n_total = 0

    with torch.no_grad():
        for poses_2d, poses_3d, gt_angles, exercise_ids in test_loader:
            poses_2d = poses_2d.to(device)
            poses_3d = poses_3d.to(device)
            gt_angles = gt_angles.to(device)
            B = poses_2d.shape[0]

            body_3d, _, _, _ = model(poses_2d)
            pred_angles = angle_head(body_3d.reshape(B, -1))

            body_mpjpe_sum += mpjpe(body_3d, poses_3d[:, :23]).item() * 1000 * B

            gt_used = gt_angles[:, :n_joints]
            
            for i in range(B):
                ex_id = int(exercise_ids[i].item())
                per_exercise_pred[ex_id].append(pred_angles[i].cpu().numpy())
                per_exercise_gt[ex_id].append(gt_used[i].cpu().numpy())

            n_total += B

    body_mpjpe_mm = body_mpjpe_sum / n_total

    # Helper functions
    def _mad_raw(y, yhat):
        """Raw MAD in degrees"""
        return float(np.mean(np.abs(y - yhat)))

    def _mad_normalized(y, yhat):
        """Normalized MAD (0-1 scale by angle ranges)"""
        per_joint_error = np.abs(y - yhat)  # (N, n_joints)
        normalized_error = per_joint_error / ANGLE_RANGES[:n_joints]
        return float(np.mean(normalized_error))

    # Compute per-exercise metrics
    print('\n' + '=' * 100)
    print(f'Per-Exercise MAD (Raw & Normalized)')
    print(f'Body MPJPE: {body_mpjpe_mm:.2f} mm')
    print('=' * 100)
    print(f'{"Exercise":<10} {"Raw MAD (°)":>15} {"Norm MAD":>12} {"Frames":>8}')
    print('-' * 100)

    per_ex_metrics = {}
    for ex_id in range(10):
        if len(per_exercise_pred[ex_id]) == 0:
            continue
        pred_arr = np.array(per_exercise_pred[ex_id])
        gt_arr = np.array(per_exercise_gt[ex_id])
        
        raw_mad = _mad_raw(gt_arr, pred_arr)
        norm_mad = _mad_normalized(gt_arr, pred_arr)
        n_frames = len(per_exercise_pred[ex_id])
        
        per_ex_metrics[ex_id] = {'raw': raw_mad, 'norm': norm_mad, 'n': n_frames}
        print(f'{EXERCISE_NAMES[ex_id]:<10} {raw_mad:>15.4f} {norm_mad:>12.4f} {n_frames:>8}')

    print('-' * 100)

    # Overall metrics
    all_pred = np.concatenate([np.array(per_exercise_pred[i]) for i in range(10)
                              if len(per_exercise_pred[i]) > 0])
    all_gt = np.concatenate([np.array(per_exercise_gt[i]) for i in range(10)
                            if len(per_exercise_gt[i]) > 0])
    overall_raw_mad = _mad_raw(all_gt, all_pred)
    overall_norm_mad = _mad_normalized(all_gt, all_pred)

    print(f'{"Average":<10} {overall_raw_mad:>15.4f} {overall_norm_mad:>12.4f} {n_total:>8}')
    print('=' * 100)

    # Per-joint breakdown
    print(f'\nPer-Joint MAD (Raw & Normalized):')
    print('-' * 60)
    print(f'{"Joint":<20} {"Raw (°)":>15} {"Normalized":>15}')
    print('-' * 60)
    per_joint_raw_error = np.abs(all_gt - all_pred).mean(axis=0)
    per_joint_norm_error = per_joint_raw_error / ANGLE_RANGES[:n_joints]
    for i, name in enumerate(JOINT_NAMES[:n_joints]):
        print(f'{name:<20} {per_joint_raw_error[i]:>15.4f} {per_joint_norm_error[i]:>15.4f}')
    print('-' * 60)

    if args.save_csv:
        with open(args.save_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Exercise', 'Raw_MAD_deg', 'Normalized_MAD', 'Frames'])
            for ex_id in range(10):
                if ex_id in per_ex_metrics:
                    m = per_ex_metrics[ex_id]
                    writer.writerow([EXERCISE_NAMES[ex_id], f'{m["raw"]:.4f}',
                                   f'{m["norm"]:.4f}', m['n']])
            writer.writerow([])
            writer.writerow(['Overall', f'{overall_raw_mad:.4f}',
                           f'{overall_norm_mad:.4f}', n_total])
            writer.writerow(['Body_MPJPE_mm', f'{body_mpjpe_mm:.2f}'])
            writer.writerow([])
            writer.writerow(['Joint', 'Raw_MAD_deg', 'Normalized_MAD'])
            for i, name in enumerate(JOINT_NAMES[:n_joints]):
                writer.writerow([name, f'{per_joint_raw_error[i]:.4f}',
                               f'{per_joint_norm_error[i]:.4f}'])
        print(f'\nResults saved to: {args.save_csv}')


if __name__ == '__main__':
    main()
