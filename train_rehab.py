"""
\"\"\"
Fine-tuning/training script for GCADA rehabilitation pipeline.

Train or fine-tune HR-GCN on UI-PRMD data with clinical angle supervision.

RECOMMENDED: from-scratch training (best for H3WB->Vicon domain shift):
    python train_rehab.py --from_scratch --lr 5e-4 --batch_size 64 \\
        --backbone_lr_factor 1.0 --warmup_epochs 3 --progressive_weights

KEY FLAGS:
  --from_scratch: Train from random init (ignore pretrained)
  --progressive_weights: Ramp angle loss 0.1x->1.0x over epochs
  --warmup_epochs 3: Minimal warmup (was 11, now configurable)
  --batch_size 64: Smaller for domain adaptation (was 256)
  --lr 5e-4: from-scratch learning rate (was 1e-4)
"""

from __future__ import print_function, absolute_import, division

import argparse
import datetime
import logging
import os
import os.path as path
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from lib.config import cfg
from common.graph_utils import adj_mx_from_skeleton
from common.loss import mpjpe
from common.clinical_loss import ClinicalPoseLoss
from models.clinical_angle_head import ClinicalAngleHead
from utils.prepare_data_h3wb import Human3WBDataset

import models.graph_hrnet_multi_branch as ghrmb
import models.graph_resnet as GraphRes
import models.graph_hrnet as ghr
from models.graph_sh import GraphSH

ROM_JOINT_NAMES = [
    'Cervical Pitch',
    'Trunk Flex', 'Left Hip', 'Right Hip',
    'Left Knee', 'Right Knee',
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='GCADA rehab fine-tuning')

    parser.add_argument('--pretrained', default='', type=str,
                        help='Path to H3WB pre-trained checkpoint')
    parser.add_argument('-cfg', '--cfg', default='w32_adam_lr1e-3.yaml', type=str,
                        help='Model config yaml')
    parser.add_argument('--gcn', default='dc_preagg', type=str,
                        help='GCN variant (must match pretrained checkpoint)')
    parser.add_argument('-m', '--model', default=1, type=int,
                        help='Model index (1-4, must match pretrained checkpoint)')
    parser.add_argument('-e', '--epochs', default=50, type=int)
    parser.add_argument('--lr', default=5e-4, type=float,
                        help='Learning rate. Recommended: 5e-4 (from-scratch) or 1e-4 (fine-tune)')
    parser.add_argument('-b', '--batch_size', default=64, type=int,
                        help='Batch size. Smaller (32-64) better for domain adaptation.')
    parser.add_argument('--freeze_backbone', action='store_true',
                        help='Freeze all HR-GCN weights; train angle head only')
    parser.add_argument('--backbone_lr_factor', default=0.5, type=float,
                        help='Backbone LR = args.lr * backbone_lr_factor. '
                             'Recommend 0.5 (2x lower) for fine-tune; 1.0 for from-scratch.')
    parser.add_argument('--lambda_angle', default=0.01, type=float,
                        help='Weight of clinical angle supervision loss. Start low (0.01-0.05), '
                             'increase with --progressive_weights')
    parser.add_argument('--lambda_constraint', default=0.01, type=float,
                        help='Weight of anatomical constraint penalty. Start low with angle loss.')
    parser.add_argument('--warmup_epochs', default=3, type=int,
                        help='Freeze backbone for N initial epochs (let angle head initialize). '
                             'Set to 0 for no warmup (recommended for from-scratch). Default 3.')
    parser.add_argument('--progressive_weights', action='store_true',
                        help='Enable progressive loss weighting: low angle/constraint early, '
                             'increase during training. Recommended for domain adaptation.')
    parser.add_argument('--from_scratch', action='store_true',
                        help='Ignore --pretrained; train from random init. '
                             'Recommended if H3WB model is mismatched. Use with --lr 5e-4 '
                             '--backbone_lr_factor 1.0')
    parser.add_argument('-c', '--checkpoint', default='checkpoint_rehab', type=str,
                        help='Output checkpoint directory')
    parser.add_argument('--data_train', default='data/uiprmd_train.npz', type=str)
    parser.add_argument('--data_test',  default='data/uiprmd_test.npz',  type=str)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--hid_dim', default=64, type=int)
    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--log_file', default='', type=str,
                        help='Path to log file. Defaults to <checkpoint>/train_rehab.log')

    return parser.parse_args()


def setup_logging(log_path: str):
    """Send all print() output to both stdout and a log file."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode='w'),
        ],
    )
    # Redirect built-in print to logging so existing print calls are captured.
    import builtins
    _real_print = builtins.print

    def _print(*args, **kwargs):
        kwargs.pop('file', None)
        kwargs.pop('flush', None)
        logging.info(' '.join(str(a) for a in args))

    builtins.print = _print


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class UIRPMDDataset(TensorDataset):
    """Thin wrapper around the NPZ format produced by prepare_data_uiprmd.py."""
    def __init__(self, npz_path: str):
        d = np.load(npz_path, allow_pickle=True)
        poses_2d   = torch.from_numpy(d['poses_2d']).float()    # (N, 133, 2)
        poses_3d   = torch.from_numpy(d['poses_3d']).float()    # (N, 133, 3)
        rom_angles = torch.from_numpy(d['rom_angles']).float()  # (N, 6)
        super().__init__(poses_2d, poses_3d, rom_angles)


# ---------------------------------------------------------------------------
# Model builder (mirrors HRNet_GCN_WB.py)
# ---------------------------------------------------------------------------

def build_model(args, cfg, adj, device):
    p_dropout = None if args.dropout == 0.0 else args.dropout
    skeleton  = _get_skeleton()

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


def _get_skeleton():
    """Load skeleton from H3WB NPZ (used only for graph structure)."""
    dataset = Human3WBDataset('data/h3wb_train.npz', 'data/h3wb_test.npz')
    return dataset.skeleton()


# ---------------------------------------------------------------------------
# Training / Evaluation helpers
# ---------------------------------------------------------------------------

def train_one_epoch(loader, model, angle_head, criterion, optimizer, device, epoch):
    model.train()
    angle_head.train()

    total_loss = total_pos = total_ang = total_con = 0.0
    n_batches = len(loader)

    for i, (inputs_2d, targets_3d, target_angles) in enumerate(loader):
        inputs_2d     = inputs_2d.to(device)
        targets_3d    = targets_3d.to(device)
        target_angles = target_angles.to(device)

        body_3d, face_3d, lhand_3d, rhand_3d = model(inputs_2d)
        pred_angles = angle_head(body_3d)

        loss_dict = criterion(
            body_3d,  face_3d,  lhand_3d,  rhand_3d,
            targets_3d[:, :23],
            targets_3d[:, 23:91],
            targets_3d[:, 91:112],
            targets_3d[:, 112:],
            pred_angles,
            target_angles,
        )

        optimizer.zero_grad()
        loss_dict['total'].backward()
        nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(angle_head.parameters()),
            max_norm=1.0,
        )
        optimizer.step()

        total_loss += loss_dict['total'].item()
        total_pos  += loss_dict['L_pos']
        total_ang  += loss_dict['L_angle']
        total_con  += loss_dict['L_constraint']

        if (i + 1) % 10 == 0 or (i + 1) == n_batches:
            print(f'  Batch {i+1:4d}/{n_batches} | '
                  f'total={total_loss/(i+1):.4f}  '
                  f'L_pos={total_pos/(i+1):.4f}  '
                  f'L_angle={total_ang/(i+1):.4f}  '
                  f'L_constr={total_con/(i+1):.4f}')

    return total_loss / n_batches


@torch.no_grad()
def evaluate(loader, model, angle_head, device):
    model.eval()
    angle_head.eval()

    body_mpjpe_sum = 0.0
    rom_mae_sum    = np.zeros(6, dtype=np.float64)
    n_samples      = 0

    for inputs_2d, targets_3d, target_angles in loader:
        inputs_2d     = inputs_2d.to(device)
        targets_3d    = targets_3d.to(device)
        target_angles = target_angles.to(device)

        body_3d, face_3d, lhand_3d, rhand_3d = model(inputs_2d)
        pred_angles = angle_head(body_3d)

        # Body MPJPE (mm)
        body_gt = targets_3d[:, :23]
        body_mpjpe_sum += mpjpe(body_3d, body_gt).item() * 1000 * inputs_2d.shape[0]

        # ROM MAE per joint (degrees)
        mae = (pred_angles - target_angles).abs().mean(dim=0).cpu().numpy()
        rom_mae_sum += mae * inputs_2d.shape[0]
        n_samples   += inputs_2d.shape[0]

    body_mpjpe_mm = body_mpjpe_sum / n_samples
    rom_mae       = rom_mae_sum    / n_samples
    mean_rom_mae  = rom_mae.mean()

    return body_mpjpe_mm, rom_mae, mean_rom_mae


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    os.makedirs(args.checkpoint, exist_ok=True)
    log_path = args.log_file or path.join(args.checkpoint, 'train_rehab.log')
    setup_logging(log_path)

    print('==> Log file:', log_path)
    print('==> Settings:', vars(args))
    
    # Validate settings
    if args.progressive_weights and args.from_scratch:
        print('=> INFO: from-scratch + progressive_weights recommended combination')
    if args.from_scratch and args.pretrained:
        print('=> WARNING: --from_scratch set; ignoring --pretrained')
        args.pretrained = ''

    device = torch.device('cuda:0')
    cudnn.benchmark = True

    # ---- Config ----
    cfg.merge_from_file(args.cfg)

    # ---- Skeleton + adjacency ----
    print('==> Loading skeleton...')
    skeleton = _get_skeleton()
    adj = adj_mx_from_skeleton(skeleton).to(device)

    # ---- Backbone ----
    print('==> Building model...')
    model = build_model(args, cfg, adj, device)
    print('    Total parameters: {:.2f}M'.format(
        sum(p.numel() for p in model.parameters()) / 1e6))

    if args.pretrained and not args.from_scratch:
        if not path.isfile(args.pretrained):
            raise FileNotFoundError(f'Checkpoint not found: {args.pretrained}')
        print(f'==> Loading pretrained weights from {args.pretrained}')
        ckpt = torch.load(args.pretrained, map_location=device)
        missing, unexpected = model.load_state_dict(ckpt['state_dict'], strict=False)
        print(f'    Loaded epoch {ckpt.get("epoch", "?")} '
              f'error={ckpt.get("error", "?")}')
        if missing:
            print(f'    Missing keys ({len(missing)}): {missing[:3]}...')
        if unexpected:
            print(f'    Unexpected keys ({len(unexpected)}): ignored '
                  f'(e.g. cross-attention layers not in base model)')
    elif args.from_scratch:
        print('==> Training from random initialization (--from_scratch)')

    # ---- Clinical angle head ----
    angle_head = ClinicalAngleHead(in_features=69, hidden=128).to(device)

    # ---- Optionally freeze backbone ----
    if args.freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
        print('==> Backbone FROZEN — training angle head only')

    # Optimizer with differential learning rates
    backbone_lr = args.lr * args.backbone_lr_factor
    if args.freeze_backbone:
        optimizer = torch.optim.Adam(angle_head.parameters(), lr=args.lr)
        print(f'==> Optimizer: angle_head lr={args.lr:.2e}')
    else:
        optimizer = torch.optim.Adam([
            {'params': model.parameters(),      'lr': backbone_lr},
            {'params': angle_head.parameters(), 'lr': args.lr},
        ])
        status = '(from-scratch)' if args.from_scratch else '(fine-tune)'
        print(f'==> Optimizer {status}: backbone lr={backbone_lr:.2e}  angle_head lr={args.lr:.2e}')

    # ---- Loss ----
    criterion = ClinicalPoseLoss(
        lambda_angle=args.lambda_angle,
        lambda_constraint=args.lambda_constraint,
    ).to(device)
    if args.progressive_weights:
        print('==> Progressive weighting enabled: 0.1x → 0.5x → 1.0x over training')

    # ---- Data ----
    print('==> Loading UI-PRMD data...')
    train_set = UIRPMDDataset(args.data_train)
    test_set  = UIRPMDDataset(args.data_test)
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=True)
    print(f'    Train frames: {len(train_set)}  Test frames: {len(test_set)}')

    # ---- LR scheduler: halve LR when ROM MAE stops improving ----
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5,
    )

    # ---- Output dir ----
    best_ckpt = path.join(args.checkpoint, 'ckpt_best_rehab.pth.tar')

    # ---- Training loop ----
    best_mae = float('inf')
    history  = []

    for epoch in range(args.epochs):
        # Backbone warmup (optional): freeze for first N epochs to let angle head initialize.
        if not args.freeze_backbone:
            if epoch < args.warmup_epochs:
                for p in model.parameters():
                    p.requires_grad = False
                if epoch == 0 and args.warmup_epochs > 0:
                    print(f'==> Warmup: backbone frozen for epochs 1-{args.warmup_epochs}')
            else:
                for p in model.parameters():
                    p.requires_grad = True
                if epoch == args.warmup_epochs and args.warmup_epochs > 0:
                    print(f'==> Warmup complete: backbone unfrozen')
        
        # Progressive loss weighting: low angle/constraint early, increase later.
        if args.progressive_weights:
            if epoch < args.epochs // 3:
                criterion.lambda_angle = args.lambda_angle * 0.1
                criterion.lambda_constraint = args.lambda_constraint * 0.1
            elif epoch < 2 * args.epochs // 3:
                criterion.lambda_angle = args.lambda_angle * 0.5
                criterion.lambda_constraint = args.lambda_constraint * 0.5
            else:
                criterion.lambda_angle = args.lambda_angle
                criterion.lambda_constraint = args.lambda_constraint

        # Show current LRs and loss weights
        lrs = [pg['lr'] for pg in optimizer.param_groups]
        lr_str = '  '.join(f'{lr:.2e}' for lr in lrs)
        weights_str = f'  λ_angle={criterion.lambda_angle:.3f} λ_constr={criterion.lambda_constraint:.3f}' if args.progressive_weights else ''
        print(f'\nEpoch {epoch+1}/{args.epochs}  lr=[{lr_str}]{weights_str}')

        train_loss = train_one_epoch(
            train_loader, model, angle_head, criterion, optimizer, device, epoch)

        body_mpjpe, rom_mae, mean_mae = evaluate(
            test_loader, model, angle_head, device)

        # Per-joint MAE breakdown every epoch
        print(f'  [Eval] Body MPJPE: {body_mpjpe:.2f} mm | Mean ROM MAE: {mean_mae:.2f} deg')
        print(f'  {"Joint":<16} | {"MAE (deg)":>9}')
        print(f'  {"-"*16}-+-{"-"*9}')
        for name, mae_val in zip(ROM_JOINT_NAMES, rom_mae):
            marker = ' *' if mae_val == rom_mae.max() else ''
            print(f'  {name:<16} | {mae_val:>9.1f}{marker}')
        print(f'  {"-"*16}-+-{"-"*9}')

        # Step scheduler on ROM MAE
        scheduler.step(mean_mae)

        history.append({
            'epoch':      epoch + 1,
            'train_loss': train_loss,
            'body_mpjpe': body_mpjpe,
            'rom_mae':    rom_mae.tolist(),
            'mean_mae':   mean_mae,
        })

        if mean_mae < best_mae:
            best_mae = mean_mae
            torch.save({
                'epoch':                  epoch + 1,
                'state_dict':             model.state_dict(),
                'angle_head_state_dict':  angle_head.state_dict(),
                'optimizer':              optimizer.state_dict(),
                'best_mae':               best_mae,
                'args':                   vars(args),
            }, best_ckpt)
            print(f'  --> New best Mean ROM MAE: {best_mae:.2f} deg  '
                  f'(checkpoint saved)')

    # ---- Final summary table ----
    _, final_rom_mae, final_mean_mae = evaluate(test_loader, model, angle_head, device)

    print('\n' + '='*46)
    print(f'{"Joint":<16} | {"MAE (deg)":>9}')
    print('-'*16 + '-+-' + '-'*9)
    for name, mae in zip(ROM_JOINT_NAMES, final_rom_mae):
        print(f'{name:<16} | {mae:>9.1f}')
    print('-'*16 + '-+-' + '-'*9)
    print(f'{"Mean":<16} | {final_mean_mae:>9.1f}')
    print('='*46)
    print(f'\nBest checkpoint: {best_ckpt}  (best MAE: {best_mae:.2f} deg)')


if __name__ == '__main__':
    main()
