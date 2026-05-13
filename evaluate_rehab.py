"""
Standalone evaluation for GCADA rehab checkpoints.

Loads a fine-tuned rehab checkpoint, runs the model + clinical angle head
on the test NPZ, and prints body MPJPE plus per-joint ROM MAE. Inference
only — no training, no optimizer, no checkpoint writes.

Example:
    python evaluate_rehab.py \\
        --checkpoint checkpoint_rehab_baseline_v8/ckpt_best_rehab.pth.tar \\
        --cfg w32_adam_lr1e-3.yaml \\
        --data_test data/uiprmd_test.npz
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


JOINT_NAMES_FULL = [
    'Cervical Pitch', 'Trunk Flex',
    'L Sho Flex',     'R Sho Flex',
    'L Sho Abd',      'R Sho Abd',
    'L Hip',          'R Hip',
    'L Knee',         'R Knee',
    'L Ankle',        'R Ankle',
]


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a rehab checkpoint')
    parser.add_argument('--checkpoint', required=True, type=str,
                        help='Path to ckpt_best_rehab.pth.tar')
    parser.add_argument('--cfg', default='w32_adam_lr1e-3.yaml', type=str)
    parser.add_argument('--gcn', default='dc_preagg', type=str)
    parser.add_argument('--model', default=1, type=int)
    parser.add_argument('--data_test', default='data/uiprmd_test.npz', type=str)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--device', default='cuda', type=str,
                        choices=['cuda', 'cpu'])
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--hid_dim', default=64, type=int)
    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--save_csv', default='', type=str,
                        help='Optional CSV output path')
    return parser.parse_args()


class UIRPMDDataset(TensorDataset):
    def __init__(self, npz_path: str):
        d = np.load(npz_path, allow_pickle=True)
        poses_2d   = torch.from_numpy(d['poses_2d']).float()
        poses_3d   = torch.from_numpy(d['poses_3d']).float()
        rom_angles = torch.from_numpy(d['rom_angles']).float()
        super().__init__(poses_2d, poses_3d, rom_angles)


def build_backbone(args, cfg, adj, device):
    p_dropout = None if 0.0 == 0.0 else 0.0
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
        model = GraphSH(adj, args.hid_dim, skeleton.joints_group(),
                        num_layers=args.num_layers, p_dropout=p_dropout,
                        gcn_type=args.gcn)
    else:
        raise ValueError(f'Unknown model index: {args.model}')
    return model.to(device)


class _AngleHead(nn.Module):
    """ClinicalAngleHead with a configurable output dim. Module name `net`
    matches the saved checkpoints' state-dict prefix."""
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


def build_angle_head(in_features: int, hidden: int, n_joints: int):
    return _AngleHead(in_features, hidden, n_joints)


def detect_head_dims(angle_state):
    """Recover (hidden, n_joints) from a checkpoint's angle_head state dict."""
    hidden = angle_state['net.0.weight'].shape[0]
    n_joints = angle_state['net.5.weight'].shape[0]
    return hidden, n_joints


def main():
    args = parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print('==> CUDA unavailable, falling back to CPU')
        args.device = 'cpu'
    device = torch.device('cuda:0' if args.device == 'cuda' else 'cpu')

    if not path.isfile(args.checkpoint):
        raise FileNotFoundError(f'Checkpoint not found: {args.checkpoint}')

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    epoch = ckpt.get('epoch', '?')
    best_mae = ckpt.get('best_mae')
    best_mae_str = f'{best_mae:.4f}' if isinstance(best_mae, (int, float)) else '?'
    print(f'==> Loaded checkpoint: {args.checkpoint}')
    print(f'    epoch={epoch}  best_mae(stored)={best_mae_str}°')

    angle_state = ckpt.get('angle_head_state_dict')
    if angle_state is None:
        raise KeyError('Checkpoint missing angle_head_state_dict')
    hidden, n_joints = detect_head_dims(angle_state)
    if n_joints not in (6, 12):
        print(f'    WARNING: unexpected angle-head output dim {n_joints}')
    print(f'    angle head: hidden={hidden}  n_joints={n_joints}')

    cfg.merge_from_file(args.cfg)

    skeleton = Human3WBDataset('data/h3wb_train.npz',
                               'data/h3wb_test.npz').skeleton()
    adj = adj_mx_from_skeleton(skeleton).to(device)

    print('==> Building model...')
    model = build_backbone(args, cfg, adj, device)
    angle_head = build_angle_head(in_features=69, hidden=hidden,
                                  n_joints=n_joints).to(device)

    model.load_state_dict(ckpt['state_dict'], strict=False)
    angle_head.load_state_dict(angle_state)

    print(f'==> Loading test data: {args.data_test}')
    test_set = UIRPMDDataset(args.data_test)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers,
                             pin_memory=(args.device == 'cuda'))
    print(f'    Frames: {len(test_set)}')

    model.eval()
    angle_head.eval()

    body_mpjpe_sum = 0.0
    rom_mae_sum = np.zeros(n_joints, dtype=np.float64)
    n_total = 0

    with torch.no_grad():
        for inputs_2d, targets_3d, gt_angles in test_loader:
            inputs_2d  = inputs_2d.to(device)
            targets_3d = targets_3d.to(device)
            gt_angles  = gt_angles.to(device)
            B = inputs_2d.shape[0]

            body_3d, _, _, _ = model(inputs_2d)
            B_h, J_h, _ = body_3d.shape
            pred_angles = angle_head(body_3d.reshape(B_h, -1))

            body_mpjpe_sum += mpjpe(body_3d, targets_3d[:, :23]).item() * 1000 * B

            gt_used = gt_angles[:, :n_joints]
            mae = (pred_angles - gt_used).abs().mean(dim=0).cpu().numpy()
            rom_mae_sum += mae * B
            n_total += B

    body_mpjpe_mm = body_mpjpe_sum / n_total
    rom_mae = rom_mae_sum / n_total
    mean_mae = float(rom_mae.mean())

    joint_names = JOINT_NAMES_FULL[:n_joints]
    worst_idx = int(np.argmax(rom_mae))

    bar = '=' * 50
    sep = '-' * 34
    print('\n' + bar)
    print(f'Checkpoint: {args.checkpoint}')
    print(f'Test frames: {n_total}')
    print(bar)
    print(f'\nBody MPJPE:   {body_mpjpe_mm:.2f} mm')
    print(f'Mean ROM MAE: {mean_mae:.4f}°\n')
    print(f'{"Joint":<20} | {"MAE (deg)":>10}')
    print(sep)
    for i, (name, mae_v) in enumerate(zip(joint_names, rom_mae)):
        marker = ' *' if i == worst_idx else ''
        print(f'{name:<20} | {mae_v:>10.2f}{marker}')
    print(sep)
    print(f'{"Mean":<20} | {mean_mae:>10.4f}')
    print(bar)

    if isinstance(best_mae, (int, float)):
        delta = abs(best_mae - mean_mae)
        if delta > 0.05:
            print(f'\nNote: live MAE differs from stored best_mae by {delta:.4f}° '
                  f'(stored {best_mae:.4f}, computed {mean_mae:.4f})')

    if args.save_csv:
        with open(args.save_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Joint', 'MAE_deg'])
            for name, mae_v in zip(joint_names, rom_mae):
                writer.writerow([name, f'{mae_v:.4f}'])
            writer.writerow(['Mean', f'{mean_mae:.4f}'])
            writer.writerow(['Body_MPJPE_mm', f'{body_mpjpe_mm:.2f}'])
            writer.writerow(['Test_frames', n_total])
            writer.writerow(['Checkpoint', args.checkpoint])
            writer.writerow(['Epoch', epoch])
        print(f'\nResults saved to: {args.save_csv}')


if __name__ == '__main__':
    main()
