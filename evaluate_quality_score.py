"""
Quality score evaluation script for GCADA rehabilitation pipeline.

Loads a checkpoint that contains both angle_head_state_dict and
quality_head_state_dict, runs inference on the test split, and computes
MAD, RMSE, MAPE per exercise (Ex1-Ex10) and overall average.

Metrics match Deb et al. (2022, IEEE TNSRE) and Kourbane et al. (2025,
Computers Bio & Med) for fair comparison on the UI-PRMD dataset.

Usage:
    python evaluate_quality_score.py \\
        --checkpoint checkpoint_rehab/ckpt_best_rehab.pth.tar \\
        --cfg w32_adam_lr1e-3.yaml \\
        --data_test data/uiprmd_test.npz
"""

from __future__ import print_function, absolute_import, division

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset

from lib.config import cfg
from common.graph_utils import adj_mx_from_skeleton
from models.clinical_angle_head import ClinicalAngleHead, QualityScoreHead
from utils.prepare_data_h3wb import Human3WBDataset

import models.graph_hrnet_multi_branch as ghrmb
import models.graph_resnet as GraphRes
import models.graph_hrnet as ghr
from models.graph_sh import GraphSH


# ---------------------------------------------------------------------------
# Dataset (read-only — no augmentation, returns exercise_id for grouping)
# ---------------------------------------------------------------------------

class UIRPMDEvalDataset(Dataset):
    def __init__(self, npz_path: str):
        d = np.load(npz_path, allow_pickle=True)

        if 'quality_scores' not in d:
            raise KeyError(
                f"\n[ERROR] 'quality_scores' key not found in {npz_path}.\n"
                "UI-PRMD quality scores (Vakanski et al., 2018 GMM) must be added\n"
                "to the NPZ before running quality evaluation.\n"
                "Expected: np.savez(..., quality_scores=array_shape_(N,)_range_0_to_1)"
            )

        self.poses_2d      = torch.from_numpy(d['poses_2d']).float()         # (N, 133, 2)
        self.quality       = torch.from_numpy(
            d['quality_scores'].astype(np.float32)).float()                  # (N,)
        self.exercise_ids  = torch.from_numpy(
            d['exercise_ids'].astype(np.int64))                              # (N,) 0-indexed

    def __len__(self):
        return len(self.poses_2d)

    def __getitem__(self, idx):
        return self.poses_2d[idx], self.quality[idx], self.exercise_ids[idx]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_skeleton():
    dataset = Human3WBDataset('data/h3wb_train.npz', 'data/h3wb_test.npz')
    return dataset.skeleton()


def build_model(args, cfg, adj, device):
    p_dropout = None
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
        model = GraphSH(adj, 64, skeleton.joints_group(), num_layers=4,
                        p_dropout=p_dropout, gcn_type=args.gcn)
    else:
        raise ValueError(f'Unknown model index: {args.model}')
    return model.to(device)


def _mad(y, yhat):
    """Mean Absolute Deviation"""
    return float(np.mean(np.abs(y - yhat)))


def _rmse(y, yhat):
    """Root Mean Squared Error"""
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def _mape(y, yhat, eps=1e-8):
    """Mean Absolute Percentage Error (%)"""
    return float(np.mean(np.abs(y - yhat) / (np.abs(y) + eps)) * 100)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Quality score evaluation for GCADA')
    parser.add_argument('--checkpoint', required=True, type=str,
                        help='Path to rehab checkpoint with quality_head_state_dict')
    parser.add_argument('-cfg', '--cfg', default='w32_adam_lr1e-3.yaml', type=str)
    parser.add_argument('--gcn', default='dc_preagg', type=str)
    parser.add_argument('-m', '--model', default=1, type=int)
    parser.add_argument('--data_test', default='data/uiprmd_test.npz', type=str)
    parser.add_argument('-b', '--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--output', default='results/quality_score_results.json', type=str,
                        help='Path to save JSON results')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if not os.path.isfile(args.checkpoint):
        print(f'[ERROR] Checkpoint not found: {args.checkpoint}')
        sys.exit(1)

    device = torch.device('cuda:0')
    cudnn.benchmark = True

    cfg.merge_from_file(args.cfg)

    # ---- Build models ----
    print(f'==> Loading checkpoint: {args.checkpoint}')
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    if 'quality_head_state_dict' not in ckpt:
        print('[ERROR] Checkpoint does not contain quality_head_state_dict.')
        print('        Train with --train_quality_head first.')
        sys.exit(1)

    skeleton = _get_skeleton()
    adj = adj_mx_from_skeleton(skeleton).to(device)

    # Override model/gcn from checkpoint args if available
    saved_args = ckpt.get('args', {})
    if 'model' in saved_args:
        args.model = saved_args['model']
    if 'gcn' in saved_args:
        args.gcn = saved_args['gcn']

    model = build_model(args, cfg, adj, device)
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model.eval()

    angle_head = ClinicalAngleHead(in_features=69, hidden=128).to(device)
    angle_head.load_state_dict(ckpt['angle_head_state_dict'])
    angle_head.eval()

    quality_head = QualityScoreHead().to(device)
    quality_head.load_state_dict(ckpt['quality_head_state_dict'])
    quality_head.eval()

    print(f'    Checkpoint epoch: {ckpt.get("epoch", "?")}')

    # ---- Data ----
    print(f'==> Loading test data: {args.data_test}')
    dataset = UIRPMDEvalDataset(args.data_test)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=True)
    print(f'    Test frames: {len(dataset)}')

    # ---- Inference ----
    all_gt   = []
    all_pred = []
    all_exid = []

    with torch.no_grad():
        for poses_2d, gt_quality, exercise_ids in loader:
            poses_2d   = poses_2d.to(device)

            body_3d, _, _, _ = model(poses_2d)
            pred_angles      = angle_head(body_3d)             # (B, 12)
            pred_quality     = quality_head(pred_angles)       # (B, 1)

            all_gt.append(gt_quality.cpu().numpy())
            all_pred.append(pred_quality.squeeze(1).cpu().numpy())
            all_exid.append(exercise_ids.cpu().numpy())

    all_gt   = np.concatenate(all_gt)    # (N,)
    all_pred = np.concatenate(all_pred)  # (N,)
    all_exid = np.concatenate(all_exid)  # (N,) 0-indexed

    # ---- Metrics per exercise ----
    results = {}
    n_exercises = int(all_exid.max()) + 1

    print('\n' + '=' * 60)
    print(f'{"Exercise":<12} {"MAD":>8} {"RMSE":>8} {"MAPE (%)":>10} {"N":>6}')
    print('-' * 60)

    for ex_idx in range(n_exercises):
        mask = (all_exid == ex_idx)
        if mask.sum() == 0:
            continue
        y    = all_gt[mask]
        yhat = all_pred[mask]
        mad  = _mad(y, yhat)
        rmse = _rmse(y, yhat)
        mape = _mape(y, yhat)
        label = f'Ex{ex_idx + 1}'
        results[label] = {'MAD': mad, 'RMSE': rmse, 'MAPE': mape, 'N': int(mask.sum())}
        print(f'{label:<12} {mad:>8.4f} {rmse:>8.4f} {mape:>10.2f} {mask.sum():>6}')

    # Overall
    mad_all  = _mad(all_gt, all_pred)
    rmse_all = _rmse(all_gt, all_pred)
    mape_all = _mape(all_gt, all_pred)
    results['Average'] = {'MAD': mad_all, 'RMSE': rmse_all, 'MAPE': mape_all, 'N': len(all_gt)}

    print('-' * 60)
    print(f'{"Average":<12} {mad_all:>8.4f} {rmse_all:>8.4f} {mape_all:>10.2f} {len(all_gt):>6}')
    print('=' * 60)

    # ---- Save JSON ----
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n==> Results saved to {args.output}')


if __name__ == '__main__':
    main()
