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
import os
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

EXERCISE_NAMES = [
    'Deep Squat',     'Hurdle Step',    'Inline Lunge',  'Side Lunge',
    'Sit to Stand',   'Str Leg Raise',  'Sho Abduction', 'Sho Extension',
    'Sho Int-Ext Rot','Sho Scaption',
]


def compute_procrustes_aligned_mpjpe(predicted, ground_truth):
    """
    predicted, ground_truth: (N, J, 3) numpy arrays (any unit, same for both).
    Per-frame rotation + translation Procrustes alignment (no scaling).
    Returns mean P-MPJPE in the same unit as the inputs.
    """
    N = predicted.shape[0]
    errors = np.empty(N, dtype=np.float64)
    for i in range(N):
        pred = predicted[i].copy()
        gt   = ground_truth[i].copy()
        # Centre both clouds
        pred -= pred.mean(axis=0)
        gt   -= gt.mean(axis=0)
        # Optimal rotation via SVD
        H = pred.T @ gt
        U, S, Vt = np.linalg.svd(H)
        # Reflection fix: ensure det(R) = +1
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        D = np.diag([1.0, 1.0, d])
        R = Vt.T @ D @ U.T
        aligned = pred @ R.T
        errors[i] = np.mean(np.linalg.norm(aligned - gt, axis=-1))
    return float(errors.mean())


def compute_rom_angles_from_coco(body_3d: np.ndarray) -> np.ndarray:
    """
    Compute 12 ROM angles from predicted COCO-body 23-joint positions using the
    same formulas as compute_rom_angles() in utils/prepare_data_uiprmd.py.

    body_3d: (N, 23, 3) — meters, hip-centered, Z-up (Vicon convention preserved
             through normalize_3d).  Angle computation is scale-invariant so the
             unit does not matter.

    COCO-23 joint indices used:
      0=nose  5=lsho  6=rsho  7=lelb  8=relb
      11=lhip 12=rhip 13=lkne 14=rkne
      15=lank 16=rank 17=ltoe 19=lhee 20=rtoe 22=rhee

    Approximations vs raw Vicon:
      c7    → mid-shoulder (joints 5+6); no C7 marker in COCO
      hip_ctr → COCO hip joint alone (no PSIS posterior marker available)

    Returns: (N, 12) degrees, order matches ROM_NAMES / JOINT_NAMES_FULL.
    """
    def _va(v1, v2):
        n1 = np.linalg.norm(v1, axis=-1, keepdims=True).clip(1e-8, None)
        n2 = np.linalg.norm(v2, axis=-1, keepdims=True).clip(1e-8, None)
        cos = ((v1 / n1) * (v2 / n2)).sum(-1).clip(-1.0 + 1e-7, 1.0 - 1e-7)
        return np.degrees(np.arccos(cos))

    N = body_3d.shape[0]

    head  = body_3d[:, 0]   # nose — head proxy
    lsho  = body_3d[:, 5]
    rsho  = body_3d[:, 6]
    lelb  = body_3d[:, 7]
    relb  = body_3d[:, 8]
    l_hip = body_3d[:, 11]
    r_hip = body_3d[:, 12]
    lkne  = body_3d[:, 13]
    rkne  = body_3d[:, 14]
    lank  = body_3d[:, 15]
    rank  = body_3d[:, 16]
    ltoe  = body_3d[:, 17]
    lhee  = body_3d[:, 19]
    rtoe  = body_3d[:, 20]
    rhee  = body_3d[:, 22]

    # Z-up unit vector (same as Vicon space preserved through normalize_3d)
    up = np.zeros((N, 3), dtype=np.float32)
    up[:, 2] = 1.0

    # Derived reference points
    c7_proxy = (lsho + rsho) / 2.0   # mid-shoulder ≈ base of neck
    mid_hip  = (l_hip + r_hip) / 2.0
    spine    = c7_proxy - mid_hip    # points upward along torso

    # 0. Cervical pitch
    cerv_pitch = _va(head - c7_proxy, up)

    # 1. Trunk flexion
    trunk_flex = _va(spine, up)

    # 2-3. Shoulder flexion (0° arm at side, increases as arm raises forward)
    l_upper_arm = lelb - lsho
    r_upper_arm = relb - rsho
    l_sho_flex  = 180.0 - _va(spine, l_upper_arm)
    r_sho_flex  = 180.0 - _va(spine, r_upper_arm)

    # 4-5. Shoulder abduction (frontal-plane projection)
    lr_axis        = rsho - lsho
    frontal_normal = np.cross(lr_axis, up)
    fn_norm        = np.linalg.norm(frontal_normal, axis=-1, keepdims=True).clip(1e-8, None)
    frontal_normal = frontal_normal / fn_norm
    l_proj = l_upper_arm - (l_upper_arm * frontal_normal).sum(-1, keepdims=True) * frontal_normal
    r_proj = r_upper_arm - (r_upper_arm * frontal_normal).sum(-1, keepdims=True) * frontal_normal
    l_sho_abd = 180.0 - _va(up, l_proj)
    r_sho_abd = 180.0 - _va(up, r_proj)

    # 6-7. Hip flexion (0° upright)
    l_femur   = lkne - l_hip
    r_femur   = rkne - r_hip
    l_hip_ang = 180.0 - _va(up, l_femur)
    r_hip_ang = 180.0 - _va(up, r_femur)

    # 8-9. Knee flexion (0° straight)
    l_tibia = lank - lkne
    r_tibia = rank - rkne
    l_knee  = 180.0 - _va(-l_femur, l_tibia)
    r_knee  = 180.0 - _va(-r_femur, r_tibia)

    # 10-11. Ankle (90°=neutral, >90°=dorsiflexion)
    l_foot  = ltoe - lhee
    r_foot  = rtoe - rhee
    l_ankle = _va(l_tibia, l_foot)
    r_ankle = _va(r_tibia, r_foot)

    return np.stack([
        cerv_pitch, trunk_flex,
        l_sho_flex, r_sho_flex,
        l_sho_abd,  r_sho_abd,
        l_hip_ang,  r_hip_ang,
        l_knee,     r_knee,
        l_ankle,    r_ankle,
    ], axis=1).astype(np.float32)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a rehab checkpoint')
    parser.add_argument('--checkpoint', default='', type=str,
                        help='Path to ckpt_best_rehab.pth.tar (required unless --full_report)')
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
    # Extended dual-checkpoint evaluation
    parser.add_argument('--full_report', action='store_true',
                        help='Run extended evaluation on baseline + novel checkpoints')
    parser.add_argument('--baseline_checkpoint',
                        default='checkpoint_rehab_baseline_v8/ckpt_best_rehab.pth.tar',
                        type=str)
    parser.add_argument('--novel_checkpoint',
                        default='checkpoint_rehab_novel_v1/ckpt_best_rehab.pth.tar',
                        type=str)
    # Geometric baseline comparison
    parser.add_argument('--geo_compare', action='store_true',
                        help='Compare geometric angles vs ROMEstimator MLP on correct-only test set')
    parser.add_argument('--save_geo_csv',
                        default='results/geometric_vs_romestimator.csv',
                        type=str,
                        help='CSV output path for geometric comparison table')
    return parser.parse_args()


class UIRPMDDataset(TensorDataset):
    def __init__(self, npz_path: str):
        d = np.load(npz_path, allow_pickle=True)
        poses_2d   = torch.from_numpy(d['poses_2d']).float()
        poses_3d   = torch.from_numpy(d['poses_3d']).float()
        rom_angles = torch.from_numpy(d['rom_angles']).float()
        super().__init__(poses_2d, poses_3d, rom_angles)


class UIRPMDDatasetFull(TensorDataset):
    """Like UIRPMDDataset but also returns exercise_ids per frame."""
    def __init__(self, npz_path: str):
        d = np.load(npz_path, allow_pickle=True)
        poses_2d     = torch.from_numpy(d['poses_2d']).float()
        poses_3d     = torch.from_numpy(d['poses_3d']).float()
        rom_angles   = torch.from_numpy(d['rom_angles']).float()
        exercise_ids = torch.from_numpy(d['exercise_ids'].astype(np.int64))
        super().__init__(poses_2d, poses_3d, rom_angles, exercise_ids)


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


def evaluate_one_checkpoint(ckpt_path, args, device):
    """
    Load a single checkpoint and evaluate fully.
    Returns a metrics dict, or None if the file is missing.
    """
    if not path.isfile(ckpt_path):
        return None

    print(f'  Loading: {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    angle_state = ckpt.get('angle_head_state_dict')
    if angle_state is None:
        raise KeyError(f'Missing angle_head_state_dict in {ckpt_path}')
    hidden, n_joints = detect_head_dims(angle_state)

    skeleton = Human3WBDataset('data/h3wb_train.npz', 'data/h3wb_test.npz').skeleton()
    adj      = adj_mx_from_skeleton(skeleton).to(device)
    model      = build_backbone(args, cfg, adj, device)
    angle_head = build_angle_head(in_features=69, hidden=hidden,
                                  n_joints=n_joints).to(device)
    model.load_state_dict(ckpt['state_dict'], strict=False)
    angle_head.load_state_dict(angle_state)
    model.eval()
    angle_head.eval()

    test_set = UIRPMDDatasetFull(args.data_test)
    loader   = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers,
                          pin_memory=(args.device == 'cuda'))

    all_pred_body, all_gt_body   = [], []
    all_pred_ang,  all_gt_ang    = [], []
    all_ex_ids                   = []

    with torch.no_grad():
        for inputs_2d, targets_3d, gt_ang_t, ex_ids in loader:
            inputs_2d  = inputs_2d.to(device)
            targets_3d = targets_3d.to(device)
            body_3d, _, _, _ = model(inputs_2d)
            B_h, J_h, _ = body_3d.shape
            pred_ang = angle_head(body_3d.reshape(B_h, -1))
            all_pred_body.append(body_3d.cpu().numpy())
            all_gt_body.append(targets_3d[:, :23].cpu().numpy())
            all_pred_ang.append(pred_ang.cpu().numpy())
            all_gt_ang.append(gt_ang_t[:, :n_joints].cpu().numpy())
            all_ex_ids.append(ex_ids.numpy())

    pred_body   = np.concatenate(all_pred_body,  axis=0) * 1000.0  # m → mm
    gt_body     = np.concatenate(all_gt_body,    axis=0) * 1000.0
    pred_angles = np.concatenate(all_pred_ang,   axis=0)
    gt_angles   = np.concatenate(all_gt_ang,     axis=0)
    ex_ids      = np.concatenate(all_ex_ids,     axis=0)
    N = pred_body.shape[0]

    mpjpe_mm   = float(np.mean(np.linalg.norm(pred_body - gt_body, axis=-1)))
    p_mpjpe_mm = compute_procrustes_aligned_mpjpe(pred_body, gt_body)

    abs_err  = np.abs(pred_angles - gt_angles)   # (N, n_joints)
    rom_mae  = abs_err.mean(axis=0)               # (n_joints,)
    mean_mae = float(rom_mae.mean())

    cerv_mae  = float(rom_mae[0])           if n_joints >= 1  else float('nan')
    trunk_mae = float(rom_mae[1])           if n_joints >= 2  else float('nan')
    sho_mae   = float(rom_mae[2:6].mean())  if n_joints >= 6  else float('nan')
    hip_mae   = float(rom_mae[6:8].mean())  if n_joints >= 8  else float('nan')
    knee_mae  = float(rom_mae[8:10].mean()) if n_joints >= 10 else float('nan')
    joints_lt5 = int(np.sum(rom_mae < 5.0))

    n_exercises = 10
    per_ex = np.full(n_exercises, float('nan'))
    for ex in range(n_exercises):
        mask = ex_ids == ex
        if mask.any():
            per_ex[ex] = float(abs_err[mask].mean())

    return dict(
        mpjpe_mm=mpjpe_mm, p_mpjpe_mm=p_mpjpe_mm,
        mean_mae=mean_mae, rom_mae=rom_mae,
        cerv_mae=cerv_mae, trunk_mae=trunk_mae,
        sho_mae=sho_mae, hip_mae=hip_mae, knee_mae=knee_mae,
        joints_lt5=joints_lt5, n_joints=n_joints,
        per_ex=per_ex, n_total=N,
    )


def run_full_report(args, device):
    """Evaluate baseline + novel checkpoints and print/save comparison tables."""
    print('\n==> Full Report Mode')
    print(f'    Baseline : {args.baseline_checkpoint}')
    print(f'    GCADA    : {args.novel_checkpoint}')

    cfg.merge_from_file(args.cfg)

    print('\n  Evaluating baseline checkpoint...')
    base  = evaluate_one_checkpoint(args.baseline_checkpoint, args, device)
    print('\n  Evaluating GCADA (novel) checkpoint...')
    novel = evaluate_one_checkpoint(args.novel_checkpoint,    args, device)

    if base  is None:
        print(f'  [MISSING] Baseline not found: {args.baseline_checkpoint}')
    if novel is None:
        print(f'  [MISSING] GCADA not found:    {args.novel_checkpoint}')

    def _f(v):
        return f'{v:.2f}' if (v is not None and not np.isnan(v)) else 'N/A'

    b = base  or {}
    n = novel or {}

    b_nj   = b.get('n_joints', 12)
    n_nj   = n.get('n_joints', 12)
    b_lt5  = f'{b.get("joints_lt5", "N/A")}/{b_nj}' if base  else 'N/A'
    n_lt5  = f'{n.get("joints_lt5", "N/A")}/{n_nj}' if novel else 'N/A'

    SEP = '-' * 48
    print('\n' + SEP)
    print(f'{"METRIC":<20} {"BASELINE":>12} {"GCADA FULL":>12}')
    print(SEP)
    print(f'{"MPJPE (mm)":<20} {_f(b.get("mpjpe_mm")):>12} {_f(n.get("mpjpe_mm")):>12}')
    print(f'{"P-MPJPE (mm)":<20} {_f(b.get("p_mpjpe_mm")):>12} {_f(n.get("p_mpjpe_mm")):>12}')
    print(f'{"Mean ROM MAE (°)":<20} {_f(b.get("mean_mae")):>12} {_f(n.get("mean_mae")):>12}')
    print(f'{"Knee MAE (°)":<20} {_f(b.get("knee_mae")):>12} {_f(n.get("knee_mae")):>12}')
    print(f'{"Hip MAE (°)":<20} {_f(b.get("hip_mae")):>12} {_f(n.get("hip_mae")):>12}')
    print(f'{"Shoulder MAE (°)":<20} {_f(b.get("sho_mae")):>12} {_f(n.get("sho_mae")):>12}')
    print(f'{"Trunk MAE (°)":<20} {_f(b.get("trunk_mae")):>12} {_f(n.get("trunk_mae")):>12}')
    print(f'{"Cervical MAE (°)":<20} {_f(b.get("cerv_mae")):>12} {_f(n.get("cerv_mae")):>12}')
    print(f'{"Joints < 5° (n/12)":<20} {b_lt5:>12} {n_lt5:>12}')
    print(SEP)

    per_ex_b = b.get('per_ex', np.full(10, float('nan')))
    per_ex_n = n.get('per_ex', np.full(10, float('nan')))

    print('\n' + SEP)
    print(f'{"EXERCISE":<16} {"BASELINE(°)":>12} {"GCADA(°)":>10} {"<5°?":>6}')
    print(SEP)
    for i, ex_name in enumerate(EXERCISE_NAMES):
        bv = per_ex_b[i]
        nv = per_ex_n[i]
        lt5 = 'Y' if (not np.isnan(nv) and nv < 5.0) else 'N'
        print(f'{ex_name:<16} {_f(bv):>12} {_f(nv):>10} {lt5:>6}')
    valid_b = per_ex_b[~np.isnan(per_ex_b)]
    valid_n = per_ex_n[~np.isnan(per_ex_n)]
    avg_b = _f(valid_b.mean()) if len(valid_b) else 'N/A'
    avg_n = _f(valid_n.mean()) if len(valid_n) else 'N/A'
    print(f'{"AVERAGE":<16} {avg_b:>12} {avg_n:>10}')
    print(SEP)

    os.makedirs('results', exist_ok=True)

    summary_path = 'results/final_summary_table.csv'
    with open(summary_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Metric', 'Baseline', 'GCADA_Full'])
        w.writerow(['MPJPE_mm',         _f(b.get('mpjpe_mm')),   _f(n.get('mpjpe_mm'))])
        w.writerow(['P-MPJPE_mm',       _f(b.get('p_mpjpe_mm')), _f(n.get('p_mpjpe_mm'))])
        w.writerow(['Mean_ROM_MAE_deg',  _f(b.get('mean_mae')),   _f(n.get('mean_mae'))])
        w.writerow(['Knee_MAE_deg',      _f(b.get('knee_mae')),   _f(n.get('knee_mae'))])
        w.writerow(['Hip_MAE_deg',       _f(b.get('hip_mae')),    _f(n.get('hip_mae'))])
        w.writerow(['Shoulder_MAE_deg',  _f(b.get('sho_mae')),    _f(n.get('sho_mae'))])
        w.writerow(['Trunk_MAE_deg',     _f(b.get('trunk_mae')),  _f(n.get('trunk_mae'))])
        w.writerow(['Cervical_MAE_deg',  _f(b.get('cerv_mae')),   _f(n.get('cerv_mae'))])
        w.writerow(['Joints_lt5',        b_lt5,                    n_lt5])
    print(f'\nSaved: {summary_path}')

    ex_path = 'results/final_per_exercise_table.csv'
    with open(ex_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Exercise', 'Baseline_deg', 'GCADA_deg', 'GCADA_lt5deg'])
        for i, ex_name in enumerate(EXERCISE_NAMES):
            bv = per_ex_b[i]
            nv = per_ex_n[i]
            lt5 = 'Y' if (not np.isnan(nv) and nv < 5.0) else 'N'
            w.writerow([ex_name, _f(bv), _f(nv), lt5])
        w.writerow(['AVERAGE', avg_b, avg_n, ''])
    print(f'Saved: {ex_path}')


def run_geometric_comparison(args, device):
    """
    Three-way MAE comparison on the correct-only test set:

      Method 1 (GEOMETRIC) — same formulas used to build ground truth labels,
                             applied to the baseline backbone's predicted 3D joints.
                             No MLP involved; isolates 3D-pose quality.
      Method 2 (BASELINE)  — baseline checkpoint MLP (λ_angle=0, λ_constraint=0).
      Method 3 (GCADA)     — full GCADA checkpoint MLP (λ_angle/λ_constraint > 0).

    Prints a comparison table and saves results/geometric_vs_romestimator.csv.
    """
    print('\n==> Geometric vs ROMEstimator Comparison')
    print(f'    Baseline : {args.baseline_checkpoint}')
    print(f'    GCADA    : {args.novel_checkpoint}')
    print(f'    Test data: {args.data_test}')

    cfg.merge_from_file(args.cfg)

    # ── Load & filter test data ───────────────────────────────────────────────
    d = np.load(args.data_test, allow_pickle=True)
    poses_2d  = d['poses_2d']    # (N, 133, 2)
    gt_angles = d['rom_angles']  # (N, 12)

    if 'quality_scores' in d:
        qscores      = d['quality_scores']
        correct_mask = qscores > 0.5
        n_total      = len(poses_2d)
        n_correct    = int(correct_mask.sum())
        print(f'\n    Test frames (all):     {n_total}')
        print(f'    Test frames (correct): {n_correct} ({100.0*n_correct/n_total:.1f}%)')
        poses_2d  = poses_2d[correct_mask]
        gt_angles = gt_angles[correct_mask]
    else:
        print('    WARNING: quality_scores not in NPZ — using all frames as correct')

    N = poses_2d.shape[0]
    print(f'    Evaluating on {N} frames')

    poses_2d_t = torch.from_numpy(poses_2d).float()

    # ── Shared skeleton / adjacency ───────────────────────────────────────────
    skeleton = Human3WBDataset('data/h3wb_train.npz', 'data/h3wb_test.npz').skeleton()
    adj      = adj_mx_from_skeleton(skeleton).to(device)

    def _infer(ckpt_path):
        """Load checkpoint; return (body_3d, pred_angles) as numpy, or (None, None)."""
        if not path.isfile(ckpt_path):
            print(f'    [MISSING] {ckpt_path}')
            return None, None
        print(f'    Loading: {ckpt_path}')
        ckpt        = torch.load(ckpt_path, map_location=device, weights_only=False)
        angle_state = ckpt.get('angle_head_state_dict')
        if angle_state is None:
            raise KeyError(f'Missing angle_head_state_dict in {ckpt_path}')
        hidden, n_joints = detect_head_dims(angle_state)

        model      = build_backbone(args, cfg, adj, device)
        angle_head = build_angle_head(in_features=69, hidden=hidden,
                                      n_joints=n_joints).to(device)
        model.load_state_dict(ckpt['state_dict'], strict=False)
        angle_head.load_state_dict(angle_state)
        model.eval()
        angle_head.eval()

        loader = DataLoader(TensorDataset(poses_2d_t), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=(args.device == 'cuda'))

        all_body, all_ang = [], []
        with torch.no_grad():
            for (inp,) in loader:
                inp = inp.to(device)
                body_3d, _, _, _ = model(inp)
                pred_ang = angle_head(body_3d.reshape(body_3d.shape[0], -1))
                all_body.append(body_3d.cpu().numpy())
                all_ang.append(pred_ang.cpu().numpy())

        return (np.concatenate(all_body, axis=0),   # (N, 23, 3)
                np.concatenate(all_ang,  axis=0))   # (N, n_joints)

    # ── Inference ─────────────────────────────────────────────────────────────
    print('\n  [1/2] Baseline checkpoint...')
    base_body, base_ang = _infer(args.baseline_checkpoint)

    print('\n  [2/2] GCADA checkpoint...')
    gcada_body, gcada_ang = _infer(args.novel_checkpoint)

    # ── Geometric angles from baseline backbone's predicted 3D joints ─────────
    geo_ang = compute_rom_angles_from_coco(base_body) if base_body is not None else None

    # ── Per-joint MAE ─────────────────────────────────────────────────────────
    gt_12 = gt_angles[:, :12]

    def _mae(pred):
        if pred is None:
            return np.full(12, float('nan'))
        return np.abs(pred[:, :12] - gt_12).mean(axis=0)

    geo_mae  = _mae(geo_ang)
    base_mae = _mae(base_ang)
    gcad_mae = _mae(gcada_ang)

    JLABELS = [
        'Cervical',   'Trunk',
        'L Sho Flex', 'R Sho Flex',
        'L Sho Abd',  'R Sho Abd',
        'L Hip',      'R Hip',
        'L Knee',     'R Knee',
        'L Ankle',    'R Ankle',
    ]

    def _f(v):
        return f'{v:6.2f}' if not np.isnan(v) else '   N/A'

    SEP = '-' * 56
    print('\n' + SEP)
    print(f'{"JOINT":<14} {"GEOMETRIC":>10} {"BASELINE":>10} {"GCADA":>10}')
    print(SEP)
    for i, name in enumerate(JLABELS):
        print(f'{name:<14} {_f(geo_mae[i]):>10} {_f(base_mae[i]):>10} {_f(gcad_mae[i]):>10}')
    print(SEP)
    print(f'{"MEAN":<14} {_f(float(np.nanmean(geo_mae))):>10}'
          f' {_f(float(np.nanmean(base_mae))):>10}'
          f' {_f(float(np.nanmean(gcad_mae))):>10}')
    print(SEP)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    os.makedirs('results', exist_ok=True)
    save_path = args.save_geo_csv
    with open(save_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Joint', 'Geometric_MAE_deg', 'Baseline_MAE_deg', 'GCADA_MAE_deg'])
        for i, name in enumerate(JLABELS):
            w.writerow([name,
                        f'{geo_mae[i]:.4f}'  if not np.isnan(geo_mae[i])  else 'N/A',
                        f'{base_mae[i]:.4f}' if not np.isnan(base_mae[i]) else 'N/A',
                        f'{gcad_mae[i]:.4f}' if not np.isnan(gcad_mae[i]) else 'N/A'])
        w.writerow(['MEAN',
                    f'{np.nanmean(geo_mae):.4f}',
                    f'{np.nanmean(base_mae):.4f}',
                    f'{np.nanmean(gcad_mae):.4f}'])
    print(f'\nSaved: {save_path}')


def main():
    args = parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print('==> CUDA unavailable, falling back to CPU')
        args.device = 'cpu'
    device = torch.device('cuda:0' if args.device == 'cuda' else 'cpu')

    if args.full_report:
        run_full_report(args, device)
        return

    if args.geo_compare:
        run_geometric_comparison(args, device)
        return

    if not args.checkpoint:
        print('ERROR: --checkpoint is required unless --full_report is set.')
        sys.exit(1)
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
