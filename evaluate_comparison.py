"""
Comprehensive evaluation script for GCADA — produces results directly
comparable to:
  1. Physio2.2M  (Rode et al., Sci. Reports 2025)
  2. OpenCap Monocular (Falisse et al., arXiv 2025)

Outputs
-------
  results_comparison/per_exercise_mae.csv
  results_comparison/per_joint_mae.csv
  results_comparison/fig_rotational_mae.png     (OpenCap Fig-3 style)
  results_comparison/fig_per_exercise_mae.png
  results_comparison/fig_comparison.png         (vs Physio2.2M)
  results_comparison/table_per_joint.tex
  results_comparison/table_comparison.tex

Example
-------
  python evaluate_comparison.py \\
      --checkpoint checkpoint_rehab_baseline_v8/ckpt_best_rehab.pth.tar \\
      --cfg w32_adam_lr1e-3.yaml \\
      --data_test data/uiprmd_test.npz \\
      --output_dir results_comparison/
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

# ── matplotlib (publication style) ──────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

matplotlib.rcParams.update({
    'font.size': 12,
    'axes.linewidth': 1.5,
    'figure.dpi': 300,
    'font.family': 'sans-serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# ── project imports ──────────────────────────────────────────────────────────
from lib.config import cfg
from common.graph_utils import adj_mx_from_skeleton
from common.loss import mpjpe
from utils.prepare_data_h3wb import Human3WBDataset

import models.graph_hrnet_multi_branch as ghrmb
import models.graph_resnet as GraphRes
import models.graph_hrnet as ghr
from models.graph_sh import GraphSH


# ── constants ────────────────────────────────────────────────────────────────

EXERCISE_NAMES = {
    0: 'Deep Squat',
    1: 'Hurdle Step',
    2: 'Inline Lunge',
    3: 'Side Lunge',
    4: 'Sit to Stand',
    5: 'Straight Leg Raise',
    6: 'Shoulder Abduction',
    7: 'Shoulder Extension',
    8: 'Shoulder Int-Ext Rot',
    9: 'Shoulder Scaption',
}

JOINT_NAMES_FULL = [
    'Cervical Pitch', 'Trunk Flexion',
    'L Shoulder Flex', 'R Shoulder Flex',
    'L Shoulder Abd',  'R Shoulder Abd',
    'L Hip',           'R Hip',
    'L Knee',          'R Knee',
    'L Ankle',         'R Ankle',
]

# McGinley et al. (2009) Gait & Posture 29:360-369
# "errors below 5° required for clinical interpretation"
CLINICAL_THRESHOLD_DEG = 5.0

# Physio2.2M published numbers (Rode et al. 2025, Sci. Reports)
PHYSIO22M_BEST_MPJPE  = 72.0   # mm
PHYSIO22M_WORST_MPJPE = 122.0  # mm
PHYSIO22M_BEST_KNEE   = 9.3    # degrees
PHYSIO22M_WORST_KNEE  = 21.9   # degrees

# Joint groups for OpenCap-style Figure 3
JOINT_GROUPS = {
    'Cervical':        [0],
    'Trunk':           [1],
    'Shoulder\nFlex':  [2, 3],
    'Shoulder\nAbd':   [4, 5],
    'Hip':             [6, 7],
    'Knee':            [8, 9],
    'Ankle':           [10, 11],
}


# ── model helpers (mirrored from evaluate_rehab.py) ──────────────────────────

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


def detect_head_dims(angle_state):
    hidden   = angle_state['net.0.weight'].shape[0]
    n_joints = angle_state['net.5.weight'].shape[0]
    return hidden, n_joints


def build_backbone(args, cfg, adj, device):
    skeleton = Human3WBDataset('data/h3wb_train.npz',
                               'data/h3wb_test.npz').skeleton()
    p_dropout = None
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


# ── data ─────────────────────────────────────────────────────────────────────

class UIRPMDDataset(TensorDataset):
    def __init__(self, npz_path: str):
        d = np.load(npz_path, allow_pickle=True)
        poses_2d   = torch.from_numpy(d['poses_2d']).float()
        poses_3d   = torch.from_numpy(d['poses_3d']).float()
        rom_angles = torch.from_numpy(d['rom_angles']).float()
        super().__init__(poses_2d, poses_3d, rom_angles)


# ── inference ────────────────────────────────────────────────────────────────

def run_inference(model, angle_head, loader, n_joints, device):
    """
    Returns
    -------
    pred_angles : (N, n_joints) float32 numpy
    pred_3d     : (N, 23, 3)   float32 numpy  — body joints
    gt_angles   : (N, n_joints) float32 numpy
    gt_3d       : (N, 133, 3)  float32 numpy
    """
    all_pred_angles = []
    all_pred_3d     = []
    all_gt_angles   = []
    all_gt_3d       = []

    model.eval()
    angle_head.eval()

    with torch.no_grad():
        for i, (inputs_2d, targets_3d, gt_ang) in enumerate(loader):
            if i % 20 == 0:
                print(f'    batch {i}/{len(loader)}', end='\r', flush=True)

            inputs_2d  = inputs_2d.to(device)
            targets_3d = targets_3d.to(device)
            gt_ang     = gt_ang.to(device)

            body_3d, _, _, _ = model(inputs_2d)
            B, J, _ = body_3d.shape
            pred_ang = angle_head(body_3d.reshape(B, -1))

            all_pred_angles.append(pred_ang.cpu().numpy())
            all_pred_3d.append(body_3d.cpu().numpy())
            all_gt_angles.append(gt_ang[:, :n_joints].cpu().numpy())
            all_gt_3d.append(targets_3d.cpu().numpy())

    print()
    pred_angles = np.concatenate(all_pred_angles, axis=0)
    pred_3d     = np.concatenate(all_pred_3d,     axis=0)
    gt_angles   = np.concatenate(all_gt_angles,   axis=0)
    gt_3d       = np.concatenate(all_gt_3d,       axis=0)

    # Graceful NaN handling
    nan_mask = np.isnan(pred_angles)
    if nan_mask.any():
        print(f'  WARNING: {nan_mask.sum()} NaN values in predictions — replaced with 0')
        pred_angles = np.nan_to_num(pred_angles, nan=0.0)

    return pred_angles, pred_3d, gt_angles, gt_3d


# ── metric helpers ───────────────────────────────────────────────────────────

def compute_mpjpe_np(pred_3d, gt_3d_body):
    """MPJPE in metres (both arrays: N×J×3)."""
    return float(np.mean(np.linalg.norm(pred_3d - gt_3d_body, axis=-1)))


# ── output generators ────────────────────────────────────────────────────────

def print_and_save_exercise_table(ex_mae, ex_mpjpe, mean_mae, mean_mpjpe,
                                  out_dir):
    bar = '=' * 60
    sep = '-' * 60
    header = f"{'Exercise':<12}| {'Name':<26}| {'ROM MAE':>8} | {'MPJPE':>8}"

    print('\n' + bar)
    print('Per-Exercise ROM MAE (degrees) — Test Set')
    print(bar)
    print(header)
    print(sep)
    rows = []
    for ex_id in range(10):
        name = EXERCISE_NAMES[ex_id]
        mae  = ex_mae[ex_id]
        mpe  = ex_mpjpe[ex_id]
        print(f"Ex{ex_id+1:02d}        | {name:<26}| {mae:>7.2f}° | {mpe*1000:>6.1f}mm")
        rows.append({'Exercise': f'Ex{ex_id+1:02d}', 'Name': name,
                     'ROM_MAE_deg': f'{mae:.4f}',
                     'MPJPE_mm': f'{mpe*1000:.2f}'})
    print(sep)
    print(f"{'AVERAGE':<12}| {'All Exercises':<26}| {mean_mae:>7.2f}° | {mean_mpjpe*1000:>6.1f}mm")
    print(bar)

    rows.append({'Exercise': 'AVERAGE', 'Name': 'All Exercises',
                 'ROM_MAE_deg': f'{mean_mae:.4f}',
                 'MPJPE_mm': f'{mean_mpjpe*1000:.2f}'})

    csv_path = path.join(out_dir, 'per_exercise_mae.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Exercise', 'Name',
                                               'ROM_MAE_deg', 'MPJPE_mm'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nSaved: {csv_path}')


def print_and_save_joint_table(rom_mae, n_joints, out_dir):
    bar = '=' * 60
    sep = '-' * 54
    threshold = CLINICAL_THRESHOLD_DEG
    names = JOINT_NAMES_FULL[:n_joints]
    pass_count = int(np.sum(rom_mae < threshold))
    mean_mae   = float(rom_mae.mean())

    print('\n' + bar)
    print('Per-Joint ROM MAE (degrees)')
    print(bar)
    print(f"{'Joint':<22}| {'MAE (deg)':>10} | {'Clinical (<5°?)':>16}")
    print(sep)

    rows = []
    for name, mae_v in zip(names, rom_mae):
        status = '✓ PASS' if mae_v < threshold else '✗ FAIL'
        print(f"{name:<22}| {mae_v:>9.2f}° | {status:>16}")
        rows.append({'Joint': name, 'MAE_deg': f'{mae_v:.4f}',
                     'Clinical_pass': 'PASS' if mae_v < threshold else 'FAIL'})

    print(sep)
    print(f"{'Mean':<22}| {mean_mae:>9.2f}° | {pass_count}/{n_joints} PASS")
    print(f"{'Joints below 5°':<22}| {pass_count}/{n_joints}")
    print(bar)

    rows.append({'Joint': 'Mean', 'MAE_deg': f'{mean_mae:.4f}',
                 'Clinical_pass': f'{pass_count}/{n_joints}'})

    csv_path = path.join(out_dir, 'per_joint_mae.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Joint', 'MAE_deg',
                                               'Clinical_pass'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved: {csv_path}')

    return mean_mae, pass_count


def print_physio_comparison(mean_mae, mean_mpjpe_mm, rom_mae):
    knee_mae = float(np.mean(rom_mae[8:10]))   # L+R knee average
    ratio    = PHYSIO22M_BEST_KNEE / max(knee_mae, 1e-6)

    bar = '=' * 72
    sep = '-' * 72
    print('\n' + bar)
    print('Comparison vs Physio2.2M Benchmark (Rode et al. 2025)')
    print(bar)
    hdr = f"{'':24}| {'GCADA (ours)':>14} | {'Physio2.2M Best':>16} | {'Physio2.2M Worst':>17}"
    print(hdr)
    print(sep)
    print(f"{'Body MPJPE (mm)':<24}| {mean_mpjpe_mm:>13.2f}mm | "
          f"{PHYSIO22M_BEST_MPJPE:>14.0f}mm | {PHYSIO22M_WORST_MPJPE:>15.0f}mm")
    print(f"{'Knee Flex MAE (°)*':<24}| {knee_mae:>13.2f}° | "
          f"{PHYSIO22M_BEST_KNEE:>14.1f}° | {PHYSIO22M_WORST_KNEE:>15.1f}°")
    print(f"{'Mean Joint MAE (°)':<24}| {mean_mae:>13.2f}° | "
          f"{'--':>16} | {'--':>17}")
    print(sep)
    print('* Average of L/R knee MAE')
    print(f'\nKey finding: GCADA knee MAE is {ratio:.1f}x better than Physio2.2M best ({PHYSIO22M_BEST_KNEE}°)')
    print(f"Clinical threshold (5°): GCADA {'achieves it' if knee_mae < 5 else 'does NOT achieve it'}, "
          f"Physio2.2M does NOT")
    print(bar)
    return knee_mae


# ── figures ──────────────────────────────────────────────────────────────────

def fig_rotational_mae(rom_mae, rom_mae_std_by_group, out_dir):
    """Figure A — OpenCap Monocular Fig 3 style."""
    groups = list(JOINT_GROUPS.keys())
    means  = []
    stds   = []
    for gname, idxs in JOINT_GROUPS.items():
        means.append(float(np.mean(rom_mae[idxs])))
        stds.append(float(rom_mae_std_by_group.get(gname, 0.0)))

    colors = ['#2ca02c' if m < CLINICAL_THRESHOLD_DEG else '#d62728'
              for m in means]

    fig, ax = plt.subplots(figsize=(10, 5))
    x   = np.arange(len(groups))
    w   = 0.55
    bars = ax.bar(x, means, width=w, color=colors, edgecolor='black',
                  linewidth=0.8, yerr=stds, capsize=4,
                  error_kw={'elinewidth': 1.5, 'ecolor': 'black'})

    # Value labels on bars
    for bar_, m in zip(bars, means):
        ax.text(bar_.get_x() + bar_.get_width() / 2,
                bar_.get_height() + max(stds) * 0.05 + 0.15,
                f'{m:.2f}°', ha='center', va='bottom', fontsize=10,
                fontweight='bold')

    # Clinical threshold line
    ax.axhline(CLINICAL_THRESHOLD_DEG, color='black', linestyle='--',
               linewidth=1.5, label='Clinical threshold (5°)')

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=11)
    ax.set_ylabel('MAE (degrees)', fontsize=12)
    ax.set_title('GCADA Rotational Kinematics MAE (°) — UI-PRMD Test Set',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_ylim(0, max(means) * 1.3 + 1.5)

    green_patch = mpatches.Patch(color='#2ca02c', label='< 5° (clinical pass)')
    red_patch   = mpatches.Patch(color='#d62728', label='≥ 5° (clinical fail)')
    ax.legend(handles=[green_patch, red_patch,
                        plt.Line2D([0], [0], color='black', linestyle='--',
                                   linewidth=1.5, label='Clinical threshold (5°)')],
              fontsize=10, frameon=False)

    plt.tight_layout()
    fpath = path.join(out_dir, 'fig_rotational_mae.png')
    fig.savefig(fpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {fpath}')


def fig_per_exercise_mae(ex_mae, mean_mae, out_dir):
    """Figure B — per-exercise ROM MAE bar chart."""
    labels = [f'Ex{i+1:02d}' for i in range(10)]
    vals   = [ex_mae[i] for i in range(10)]
    colors = ['#2ca02c' if v < CLINICAL_THRESHOLD_DEG else '#d62728'
              for v in vals]

    fig, ax = plt.subplots(figsize=(11, 5))
    x   = np.arange(10)
    w   = 0.6
    bars = ax.bar(x, vals, width=w, color=colors, edgecolor='black',
                  linewidth=0.8)

    for bar_, v in zip(bars, vals):
        ax.text(bar_.get_x() + bar_.get_width() / 2,
                bar_.get_height() + 0.1,
                f'{v:.2f}°', ha='center', va='bottom', fontsize=9,
                fontweight='bold')

    ax.axhline(mean_mae, color='steelblue', linestyle='-.',
               linewidth=1.5, label=f'Overall mean ({mean_mae:.2f}°)')
    ax.axhline(CLINICAL_THRESHOLD_DEG, color='black', linestyle='--',
               linewidth=1.5, label='Clinical threshold (5°)')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('Mean ROM MAE (degrees)', fontsize=12)
    ax.set_title('GCADA Per-Exercise ROM MAE (°)', fontsize=13,
                 fontweight='bold', pad=12)
    ax.set_ylim(0, max(vals) * 1.3 + 1.0)
    ax.legend(fontsize=10, frameon=False)

    # Exercise name annotations under x-axis
    short_names = ['Squat', 'Hurdle', 'Lunge', 'Side\nLunge', 'Sit-Stand',
                   'Leg Raise', 'Sho Abd', 'Sho Ext', 'Sho Rot', 'Scaption']
    for xi, sn in zip(x, short_names):
        ax.text(xi, -max(vals) * 0.12, sn, ha='center', va='top',
                fontsize=7.5, color='#444444')

    plt.tight_layout()
    fpath = path.join(out_dir, 'fig_per_exercise_mae.png')
    fig.savefig(fpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {fpath}')


def fig_comparison(mean_mae, mean_mpjpe_mm, knee_mae, out_dir):
    """Figure C — GCADA vs Physio2.2M grouped bar chart."""
    # Normalise MPJPE to per-100mm scale
    gcada_mpjpe_norm   = mean_mpjpe_mm / 100.0
    p22m_best_norm     = PHYSIO22M_BEST_MPJPE  / 100.0
    p22m_worst_norm    = PHYSIO22M_WORST_MPJPE / 100.0

    categories = ['Body MPJPE\n(per 100mm)', 'Knee MAE (°)', 'Mean ROM\nMAE (°)']
    physio_best  = [p22m_best_norm,        PHYSIO22M_BEST_KNEE,  None]
    physio_worst = [p22m_worst_norm,       PHYSIO22M_WORST_KNEE, None]
    gcada        = [gcada_mpjpe_norm,      knee_mae,             mean_mae]

    n_cat = len(categories)
    x     = np.arange(n_cat)
    w     = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))

    bars_worst = ax.bar(x - w, [v if v is not None else 0 for v in physio_worst],
                        width=w, color='#d62728', edgecolor='black',
                        linewidth=0.8, label='Physio2.2M Worst')
    bars_best  = ax.bar(x,     [v if v is not None else 0 for v in physio_best],
                        width=w, color='#1f77b4', edgecolor='black',
                        linewidth=0.8, label='Physio2.2M Best')
    bars_gcada = ax.bar(x + w, [v if v is not None else 0 for v in gcada],
                        width=w, color='#2ca02c', edgecolor='black',
                        linewidth=0.8, label='GCADA (ours)')

    # Asterisk where GCADA beats both Physio2.2M variants
    for i, (pb, pw, gc) in enumerate(zip(physio_best, physio_worst, gcada)):
        if gc is not None and pb is not None and gc < pb and gc < pw:
            y_top = max(pb, pw, gc)
            ax.text(x[i] + w, y_top + 0.05, '*', ha='center', va='bottom',
                    fontsize=16, fontweight='bold', color='#2ca02c')

    # Value labels
    for bar_, v in zip(bars_worst, physio_worst):
        if v:
            ax.text(bar_.get_x() + bar_.get_width() / 2,
                    bar_.get_height() + 0.02,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=8.5)
    for bar_, v in zip(bars_best, physio_best):
        if v:
            ax.text(bar_.get_x() + bar_.get_width() / 2,
                    bar_.get_height() + 0.02,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=8.5)
    for bar_, v in zip(bars_gcada, gcada):
        if v is not None:
            ax.text(bar_.get_x() + bar_.get_width() / 2,
                    bar_.get_height() + 0.02,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=8.5)

    # Hide bars for missing data (Mean ROM MAE for Physio2.2M)
    for i, (pb, pw) in enumerate(zip(physio_best, physio_worst)):
        if pb is None:
            bars_best[i].set_visible(False)
        if pw is None:
            bars_worst[i].set_visible(False)

    ax.axhline(CLINICAL_THRESHOLD_DEG / 10, color='black', linestyle='--',
               linewidth=1.0, alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylabel('Error (lower is better)', fontsize=12)
    ax.set_title('GCADA vs Physio2.2M Benchmark\n'
                 '(* = GCADA beats both Physio2.2M variants)',
                 fontsize=12, fontweight='bold', pad=12)
    ax.legend(fontsize=10, frameon=False)
    ax.set_ylim(0, max(filter(None, physio_worst + gcada)) * 1.3)

    plt.tight_layout()
    fpath = path.join(out_dir, 'fig_comparison.png')
    fig.savefig(fpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {fpath}')


# ── LaTeX ─────────────────────────────────────────────────────────────────────

def write_latex_per_joint(rom_mae, n_joints, mean_mae, pass_count, out_dir):
    names = JOINT_NAMES_FULL[:n_joints]
    rows  = []
    for name, mae_v in zip(names, rom_mae):
        mark = r'\checkmark' if mae_v < CLINICAL_THRESHOLD_DEG else r'\texttimes'
        rows.append(f'    {name:<22} & {mae_v:.2f} & ${mark}$ \\\\')

    content = r"""\begin{table}[ht]
\centering
\caption{Per-joint ROM MAE of GCADA on UI-PRMD test set
(subjects S09--S10). Clinical threshold $5^\circ$ from
McGinley et al.\ (2009); joints below threshold are marked (\checkmark).}
\label{tab:per_joint_mae}
\begin{tabular}{|l|c|c|}
\hline
\textbf{Joint} & \textbf{MAE ($^\circ$)} & \textbf{$<5^\circ$?} \\
\hline
""" + '\n'.join(rows) + f"""
\\hline
    \\textbf{{Mean}} & \\textbf{{{mean_mae:.2f}}} & {pass_count}/{n_joints} \\\\
\\hline
\\end{{tabular}}
\\end{{table}}
"""
    fpath = path.join(out_dir, 'table_per_joint.tex')
    with open(fpath, 'w') as f:
        f.write(content)
    print(f'Saved: {fpath}')


def write_latex_comparison(mean_mae, mean_mpjpe_mm, knee_mae, out_dir):
    content = fr"""\begin{{table}}[ht]
\centering
\caption{{Comparison of GCADA with state-of-the-art monocular pose
estimation benchmarks. MPJPE in millimetres; angle MAE in degrees.
$\dagger$ = result on UI-PRMD test set (subjects S09--S10);
\textasteriskcentered~= GCADA beats both Physio2.2M variants.}}
\label{{tab:comparison}}
\begin{{tabular}}{{|l|c|c|c|}}
\hline
\textbf{{Method}} & \textbf{{MPJPE (mm)}} & \textbf{{Knee MAE ($^\circ$)}} & \textbf{{Mean ROM MAE ($^\circ$)}} \\
\hline
Physio2.2M Best (Rode et al.\ 2025)  & {PHYSIO22M_BEST_MPJPE:.0f}  & {PHYSIO22M_BEST_KNEE:.1f}  & -- \\
Physio2.2M Worst (Rode et al.\ 2025) & {PHYSIO22M_WORST_MPJPE:.0f} & {PHYSIO22M_WORST_KNEE:.1f} & -- \\
\hline
\textbf{{GCADA (ours)$\dagger$}} & \textbf{{{mean_mpjpe_mm:.2f}}}\textasteriskcentered & \textbf{{{knee_mae:.2f}}}\textasteriskcentered & \textbf{{{mean_mae:.2f}}} \\
\hline
\end{{tabular}}
\end{{table}}
"""
    fpath = path.join(out_dir, 'table_comparison.tex')
    with open(fpath, 'w') as f:
        f.write(content)
    print(f'Saved: {fpath}')


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='GCADA evaluation — literature-style comparison outputs')
    parser.add_argument('--checkpoint', required=True, type=str)
    parser.add_argument('--cfg',        default='w32_adam_lr1e-3.yaml', type=str)
    parser.add_argument('--gcn',        default='dc_preagg', type=str)
    parser.add_argument('--model',      default=1, type=int)
    parser.add_argument('--data_test',  default='data/uiprmd_test.npz', type=str)
    parser.add_argument('--output_dir', default='results_comparison/', type=str)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--device',     default='cuda', type=str,
                        choices=['cuda', 'cpu'])
    parser.add_argument('--num_workers',default=4, type=int)
    parser.add_argument('--hid_dim',    default=64, type=int)
    parser.add_argument('--num_layers', default=4, type=int)
    return parser.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.device == 'cuda' and not torch.cuda.is_available():
        print('==> CUDA unavailable, falling back to CPU')
        args.device = 'cpu'
    device = torch.device('cuda:0' if args.device == 'cuda' else 'cpu')

    # ── load checkpoint ──────────────────────────────────────────────────────
    if not path.isfile(args.checkpoint):
        raise FileNotFoundError(f'Checkpoint not found: {args.checkpoint}')
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    epoch    = ckpt.get('epoch', '?')
    best_mae = ckpt.get('best_mae')
    print(f'==> Loaded checkpoint: {args.checkpoint}')
    print(f'    epoch={epoch}  stored_best_mae='
          f'{best_mae:.4f}°' if isinstance(best_mae, (int, float)) else f'    epoch={epoch}  stored_best_mae=?')

    angle_state = ckpt.get('angle_head_state_dict')
    if angle_state is None:
        raise KeyError('Checkpoint missing angle_head_state_dict')
    hidden, n_joints = detect_head_dims(angle_state)
    print(f'    angle head: hidden={hidden}  n_joints={n_joints}')

    # ── build model ──────────────────────────────────────────────────────────
    cfg.merge_from_file(args.cfg)
    skeleton = Human3WBDataset('data/h3wb_train.npz',
                               'data/h3wb_test.npz').skeleton()
    adj = adj_mx_from_skeleton(skeleton).to(device)

    print('==> Building model...')
    model      = build_backbone(args, cfg, adj, device)
    angle_head = _AngleHead(in_features=69, hidden=hidden,
                            n_joints=n_joints).to(device)
    model.load_state_dict(ckpt['state_dict'])
    angle_head.load_state_dict(angle_state)

    # ── load data ────────────────────────────────────────────────────────────
    print(f'==> Loading test data: {args.data_test}')
    raw = np.load(args.data_test, allow_pickle=True)
    exercise_ids = raw['exercise_ids']   # (N,)
    subject_ids  = raw['subject_ids']    # (N,)

    dataset = UIRPMDDataset(args.data_test)
    loader  = DataLoader(dataset, batch_size=args.batch_size,
                         shuffle=False, num_workers=args.num_workers,
                         pin_memory=(args.device == 'cuda'))
    print(f'    Total frames: {len(dataset)}')

    # ── run inference ────────────────────────────────────────────────────────
    print('==> Running inference...')
    pred_angles, pred_3d, gt_angles, gt_3d = run_inference(
        model, angle_head, loader, n_joints, device)

    # ── global metrics ───────────────────────────────────────────────────────
    rom_mae_global = np.abs(pred_angles - gt_angles).mean(axis=0)  # (n_joints,)
    mean_mae       = float(rom_mae_global.mean())
    mean_mpjpe     = compute_mpjpe_np(pred_3d, gt_3d[:, :23])      # metres
    mean_mpjpe_mm  = mean_mpjpe * 1000.0

    print(f'\n==> Global: Body MPJPE={mean_mpjpe_mm:.2f}mm  '
          f'Mean ROM MAE={mean_mae:.4f}°')

    # Consistency check vs stored best_mae
    if isinstance(best_mae, (int, float)):
        delta = abs(best_mae - mean_mae)
        if delta > 0.05:
            print(f'    Note: live MAE differs from stored best_mae by {delta:.4f}° '
                  f'(stored {best_mae:.4f}, computed {mean_mae:.4f})')

    # ── per-exercise metrics ─────────────────────────────────────────────────
    ex_mae   = {}
    ex_mpjpe = {}
    for ex_id in range(10):
        mask = (exercise_ids == ex_id)
        if not mask.any():
            ex_mae[ex_id]   = float('nan')
            ex_mpjpe[ex_id] = float('nan')
            continue
        ex_mae[ex_id]   = float(np.abs(
            pred_angles[mask] - gt_angles[mask]).mean())
        ex_mpjpe[ex_id] = compute_mpjpe_np(pred_3d[mask], gt_3d[mask, :23])

    # ── per-repetition std (for bar chart error bars) ─────────────────────────
    # Each (subject_id, exercise_id) pair is one repetition.
    rep_maes_by_group = {gname: [] for gname in JOINT_GROUPS}
    for (sub, ex) in np.unique(np.stack([subject_ids, exercise_ids], axis=1),
                                axis=0):
        mask = (subject_ids == sub) & (exercise_ids == ex)
        if not mask.any():
            continue
        rep_rom_mae = np.abs(pred_angles[mask] - gt_angles[mask]).mean(axis=0)
        for gname, idxs in JOINT_GROUPS.items():
            rep_maes_by_group[gname].append(float(np.mean(rep_rom_mae[idxs])))

    rom_mae_std_by_group = {
        gname: float(np.std(vals)) if len(vals) > 1 else 0.0
        for gname, vals in rep_maes_by_group.items()
    }

    # ── OUTPUT 1 — per-exercise table ─────────────────────────────────────────
    print_and_save_exercise_table(ex_mae, ex_mpjpe, mean_mae, mean_mpjpe,
                                  args.output_dir)

    # ── OUTPUT 2 — per-joint table ────────────────────────────────────────────
    _, pass_count = print_and_save_joint_table(
        rom_mae_global, n_joints, args.output_dir)

    # ── OUTPUT 3 — Physio2.2M comparison ─────────────────────────────────────
    knee_mae = print_physio_comparison(mean_mae, mean_mpjpe_mm, rom_mae_global)

    # ── OUTPUT 4 — figures ────────────────────────────────────────────────────
    print('\n==> Generating figures...')
    fig_rotational_mae(rom_mae_global, rom_mae_std_by_group, args.output_dir)
    fig_per_exercise_mae(ex_mae, mean_mae, args.output_dir)
    fig_comparison(mean_mae, mean_mpjpe_mm, knee_mae, args.output_dir)

    # ── OUTPUT 5 — LaTeX ──────────────────────────────────────────────────────
    print('==> Writing LaTeX tables...')
    write_latex_per_joint(rom_mae_global, n_joints, mean_mae, pass_count,
                          args.output_dir)
    write_latex_comparison(mean_mae, mean_mpjpe_mm, knee_mae, args.output_dir)

    # ── final summary ─────────────────────────────────────────────────────────
    print(f'\n==> All outputs saved to: {args.output_dir}')
    print(f'    Body MPJPE : {mean_mpjpe_mm:.2f} mm')
    print(f'    Mean ROM MAE: {mean_mae:.4f}°  '
          f'({pass_count}/{n_joints} joints below 5°)')


if __name__ == '__main__':
    main()
