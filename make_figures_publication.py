"""
Generate two publication-quality figures for the GCADA thesis.

Figure 1 — Per-exercise ROM MAE bar chart with per-subject std error bars.
Figure 2 — Per-joint L vs R asymmetry paired bar chart.

Usage:
    python make_figures_publication.py \
        --checkpoint checkpoint_rehab_baseline_v8/ckpt_best_rehab.pth.tar \
        --cfg w32_adam_lr1e-3.yaml \
        --data_test data/uiprmd_test.npz \
        --output_dir results_comparison/
"""
from __future__ import print_function, absolute_import, division

import argparse
import os
import os.path as path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator

# McGinley et al. (2009) Gait & Posture 29:360-369
CLINICAL_THRESHOLD = 5.0
OVERALL_MEAN       = 4.5891   # degrees — from evaluate_comparison.py

matplotlib.rcParams.update({
    'font.size':        12,
    'axes.linewidth':   1.5,
    'figure.dpi':       300,
    'font.family':      'sans-serif',
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'xtick.major.width':1.2,
    'ytick.major.width':1.2,
})

# ── project imports ──────────────────────────────────────────────────────────
from lib.config import cfg
from common.graph_utils import adj_mx_from_skeleton
from utils.prepare_data_h3wb import Human3WBDataset
import models.graph_hrnet_multi_branch as ghrmb
import models.graph_resnet as GraphRes
import models.graph_hrnet as ghr
from models.graph_sh import GraphSH


EXERCISE_SHORT = [
    'Deep\nSquat',
    'Hurdle\nStep',
    'Inline\nLunge',
    'Side\nLunge',
    'Sit-\nStand',
    'Leg\nRaise',
    'Sho.\nAbd.',
    'Sho.\nExt.',
    'Sho.\nRot.',
    'Sho.\nScap.',
]

EXERCISE_FULL = [
    'Deep Squat', 'Hurdle Step', 'Inline Lunge', 'Side Lunge',
    'Sit to Stand', 'Straight Leg Raise', 'Shoulder Abduction',
    'Shoulder Extension', 'Sho. Int-Ext Rot', 'Shoulder Scaption',
]

JOINT_NAMES_FULL = [
    'Cervical Pitch', 'Trunk Flexion',
    'L Shoulder Flex', 'R Shoulder Flex',
    'L Shoulder Abd',  'R Shoulder Abd',
    'L Hip',           'R Hip',
    'L Knee',          'R Knee',
    'L Ankle',         'R Ankle',
]


# ── model helpers (exact mirror from evaluate_rehab.py) ──────────────────────

class _AngleHead(nn.Module):
    def __init__(self, in_features, hidden, n_joints):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, 64), nn.ReLU(),
            nn.Linear(64, n_joints),
        )

    def forward(self, x):
        return self.net(x)


def detect_head_dims(state):
    return state['net.0.weight'].shape[0], state['net.5.weight'].shape[0]


def build_backbone(args, cfg, adj, device):
    sk = Human3WBDataset('data/h3wb_train.npz', 'data/h3wb_test.npz').skeleton()
    if args.model == 1:
        m = ghrmb.get_pose_net(cfg, True, adj, None, args.gcn, sk.joints_group())
    elif args.model == 2:
        m = GraphRes.get_pose_net(True, adj, None, args.gcn, 50, True)
    elif args.model == 3:
        m = ghr.get_pose_net(cfg, True, adj, None, args.gcn, sk.joints_group())
    elif args.model == 4:
        m = GraphSH(adj, args.hid_dim, sk.joints_group(),
                    num_layers=args.num_layers, p_dropout=None,
                    gcn_type=args.gcn)
    else:
        raise ValueError(f'Unknown model: {args.model}')
    return m.to(device)


class UIRPMDDataset(TensorDataset):
    def __init__(self, npz_path):
        d = np.load(npz_path, allow_pickle=True)
        super().__init__(
            torch.from_numpy(d['poses_2d']).float(),
            torch.from_numpy(d['poses_3d']).float(),
            torch.from_numpy(d['rom_angles']).float(),
        )


# ── inference ────────────────────────────────────────────────────────────────

def run_inference(model, angle_head, loader, n_joints, device):
    pred_list, gt_list = [], []
    model.eval(); angle_head.eval()
    with torch.no_grad():
        for i, (inp2d, _, gt_ang) in enumerate(loader):
            if i % 30 == 0:
                print(f'    batch {i}/{len(loader)}', end='\r', flush=True)
            inp2d  = inp2d.to(device)
            gt_ang = gt_ang.to(device)
            body_3d, _, _, _ = model(inp2d)
            B = body_3d.shape[0]
            pred = angle_head(body_3d.reshape(B, -1))
            pred_list.append(pred.cpu().numpy())
            gt_list.append(gt_ang[:, :n_joints].cpu().numpy())
    print()
    return (np.nan_to_num(np.concatenate(pred_list), nan=0.0),
            np.concatenate(gt_list))


# ── Figure 1 ─────────────────────────────────────────────────────────────────

def make_fig_per_exercise(ex_means, ex_stds, out_dir):
    """
    Per-exercise bar chart with per-subject std, three-colour coding,
    and annotated threshold lines.
    """
    fig, ax = plt.subplots(figsize=(13, 5.5))

    x     = np.arange(10)
    w     = 0.60
    ymax  = 8.0

    # Colour by value
    def bar_color(v):
        if v < CLINICAL_THRESHOLD:
            return '#2ca02c'   # green  — below threshold
        elif v <= 7.0:
            return '#ff7f0e'   # orange — 5–7°
        else:
            return '#d62728'   # red    — above 7°

    colors = [bar_color(v) for v in ex_means]

    bars = ax.bar(x, ex_means, width=w, color=colors, edgecolor='#333333',
                  linewidth=0.9, zorder=3,
                  yerr=ex_stds, capsize=5,
                  error_kw={'elinewidth': 1.8, 'ecolor': '#333333',
                             'capthick': 1.8, 'zorder': 4})

    # Value labels above each bar (or above error cap)
    for i, (bar_, m, s) in enumerate(zip(bars, ex_means, ex_stds)):
        top = m + s + 0.12
        ax.text(bar_.get_x() + bar_.get_width() / 2,
                top, f'{m:.2f}°',
                ha='center', va='bottom', fontsize=9.5,
                fontweight='bold', color='#1a1a1a')

    # ── threshold lines ──────────────────────────────────────────────────────
    ax.axhline(CLINICAL_THRESHOLD, color='#d62728', linestyle='--',
               linewidth=1.8, zorder=2,
               label='Clinical threshold — McGinley (2009): 5°')
    ax.axhline(OVERALL_MEAN, color='#1f77b4', linestyle='-.',
               linewidth=1.6, zorder=2,
               label=f'Overall mean: {OVERALL_MEAN:.2f}°')

    # ── axes decoration ──────────────────────────────────────────────────────
    ax.set_xticks(x)
    ax.set_xticklabels([f'Ex{i+1:02d}\n{EXERCISE_SHORT[i]}'
                         for i in range(10)], fontsize=9.5)
    ax.set_ylabel('Mean ROM MAE (degrees)', fontsize=12, labelpad=8)
    ax.set_ylim(0, ymax)
    ax.yaxis.set_minor_locator(MultipleLocator(0.5))
    ax.tick_params(axis='y', which='minor', length=3, width=0.8)
    ax.grid(axis='y', which='major', linestyle=':', alpha=0.45, zorder=0)
    ax.set_title(
        'GCADA Per-Exercise ROM MAE on UI-PRMD  (n = 2 test subjects)',
        fontsize=13, fontweight='bold', pad=11)

    # ── legend ───────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(facecolor='#2ca02c', edgecolor='#333333',
                       label='< 5°  (clinical pass)'),
        mpatches.Patch(facecolor='#ff7f0e', edgecolor='#333333',
                       label='5–7°  (marginal)'),
        mpatches.Patch(facecolor='#d62728', edgecolor='#333333',
                       label='> 7°  (clinical fail)'),
        plt.Line2D([0], [0], color='#d62728', linestyle='--', linewidth=1.8,
                   label='Clinical threshold — McGinley (2009): 5°'),
        plt.Line2D([0], [0], color='#1f77b4', linestyle='-.', linewidth=1.6,
                   label=f'Overall mean: {OVERALL_MEAN:.2f}°'),
    ]
    ax.legend(handles=legend_handles, fontsize=9, frameon=True,
              framealpha=0.92, edgecolor='#aaaaaa', loc='upper right',
              ncol=1)

    plt.tight_layout(pad=1.2)
    fpath = path.join(out_dir, 'fig_per_exercise_final.png')
    fig.savefig(fpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {fpath}')


# ── Figure 2 ─────────────────────────────────────────────────────────────────

def make_fig_lr_asymmetry(out_dir):
    """
    Paired L/R bar chart showing joint asymmetry.
    Cervical and Trunk use single bars (no laterality).
    Asymmetric joints annotated with 'data asymmetry'.
    """

    # Data from the evaluation results
    joints = ['Cervical\nPitch', 'Trunk\nFlexion',
              'Sho.\nFlex', 'Sho.\nAbd', 'Hip', 'Knee', 'Ankle']

    # (left, right) — None means bilateral value, displayed as single bar
    left_vals  = [3.13,  1.68,  4.84,  4.75,  2.58,  3.80,  4.36]
    right_vals = [3.13,  1.68,  7.33,  9.34,  3.13,  3.77,  6.35]
    bilateral  = [True,  True,  False, False, False, False, False]

    # Pairs where L and R differ by > 2° — flag as asymmetric
    ASYM_THRESHOLD = 2.0
    asymmetric = [abs(r - l) > ASYM_THRESHOLD
                  for l, r, b in zip(left_vals, right_vals, bilateral)]

    n      = len(joints)
    x      = np.arange(n)
    w      = 0.32
    offset = 0.18     # half-gap between paired bars

    fig, ax = plt.subplots(figsize=(12, 5.5))

    BLUE   = '#1f77b4'
    ORANGE = '#ff7f0e'
    GREEN  = '#2ca02c'
    RED    = '#d62728'

    def bar_col(v, side='L'):
        base = BLUE if side == 'L' else ORANGE
        return base

    bars_L = []
    bars_R = []
    for i, (j, lv, rv, bil) in enumerate(
            zip(joints, left_vals, right_vals, bilateral)):
        if bil:
            # Single centred bar for bilateral joints
            b = ax.bar(x[i], lv, width=w * 1.5,
                       color=GREEN if lv < CLINICAL_THRESHOLD else RED,
                       edgecolor='#333333', linewidth=0.9, zorder=3)
            bars_L.append(b[0])
            bars_R.append(None)
        else:
            b_l = ax.bar(x[i] - offset, lv, width=w,
                         color=BLUE if lv < CLINICAL_THRESHOLD else '#6baed6',
                         edgecolor='#333333', linewidth=0.9, zorder=3)
            b_r = ax.bar(x[i] + offset, rv, width=w,
                         color=ORANGE if rv < CLINICAL_THRESHOLD else RED,
                         edgecolor='#333333', linewidth=0.9, zorder=3)
            bars_L.append(b_l[0])
            bars_R.append(b_r[0])

    # ── value labels ─────────────────────────────────────────────────────────
    for i, (lv, rv, bil) in enumerate(zip(left_vals, right_vals, bilateral)):
        if bil:
            ax.text(x[i], lv + 0.15, f'{lv:.2f}°',
                    ha='center', va='bottom', fontsize=9,
                    fontweight='bold', color='#1a1a1a')
        else:
            ax.text(x[i] - offset, lv + 0.15, f'{lv:.2f}°',
                    ha='center', va='bottom', fontsize=9,
                    fontweight='bold', color='#1a1a1a')
            ax.text(x[i] + offset, rv + 0.15, f'{rv:.2f}°',
                    ha='center', va='bottom', fontsize=9,
                    fontweight='bold', color='#1a1a1a')

    # ── "data asymmetry" annotations ─────────────────────────────────────────
    # Annotate R Sho Abd (idx 3) and R Ankle (idx 6)
    annotation_targets = {3: 'data\nasymmetry', 6: 'data\nasymmetry'}
    for idx, label in annotation_targets.items():
        rv  = right_vals[idx]
        xp  = x[idx] + offset
        yp  = rv + 0.25
        ax.annotate(
            label,
            xy=(xp, rv), xytext=(xp + 0.32, rv + 1.1),
            fontsize=8.5, color='#d62728', style='italic',
            ha='left', va='bottom',
            arrowprops=dict(arrowstyle='->', color='#d62728',
                            lw=1.3, connectionstyle='arc3,rad=0.2'),
        )

    # ── threshold line ────────────────────────────────────────────────────────
    ax.axhline(CLINICAL_THRESHOLD, color='#d62728', linestyle='--',
               linewidth=1.8, zorder=2,
               label='Clinical threshold — McGinley (2009): 5°')

    # ── axes decoration ───────────────────────────────────────────────────────
    ax.set_xticks(x)
    ax.set_xticklabels(joints, fontsize=11)
    ax.set_ylabel('ROM MAE (degrees)', fontsize=12, labelpad=8)
    ax.set_ylim(0, 12.0)
    ax.yaxis.set_minor_locator(MultipleLocator(0.5))
    ax.tick_params(axis='y', which='minor', length=3, width=0.8)
    ax.grid(axis='y', which='major', linestyle=':', alpha=0.45, zorder=0)
    ax.set_title(
        'GCADA Per-Joint ROM MAE: Left vs Right Asymmetry  (UI-PRMD Test Set)',
        fontsize=13, fontweight='bold', pad=11)

    # ── legend ────────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(facecolor=BLUE,   edgecolor='#333333',
                       label='Left side  (< 5°)'),
        mpatches.Patch(facecolor='#6baed6', edgecolor='#333333',
                       label='Left side  (≥ 5°)'),
        mpatches.Patch(facecolor=ORANGE, edgecolor='#333333',
                       label='Right side (< 5°)'),
        mpatches.Patch(facecolor=RED,    edgecolor='#333333',
                       label='Right side (≥ 5°)'),
        mpatches.Patch(facecolor=GREEN,  edgecolor='#333333',
                       label='Bilateral  (< 5°)'),
        plt.Line2D([0], [0], color='#d62728', linestyle='--', linewidth=1.8,
                   label='Clinical threshold — McGinley (2009): 5°'),
    ]
    ax.legend(handles=legend_handles, fontsize=9, frameon=True,
              framealpha=0.92, edgecolor='#aaaaaa', loc='upper left',
              ncol=2)

    plt.tight_layout(pad=1.2)
    fpath = path.join(out_dir, 'fig_per_joint_lr.png')
    fig.savefig(fpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {fpath}')


# ── CLI & main ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--cfg',        default='w32_adam_lr1e-3.yaml')
    p.add_argument('--gcn',        default='dc_preagg')
    p.add_argument('--model',      default=1, type=int)
    p.add_argument('--data_test',  default='data/uiprmd_test.npz')
    p.add_argument('--output_dir', default='results_comparison/')
    p.add_argument('--batch_size', default=256, type=int)
    p.add_argument('--device',     default='cuda',
                   choices=['cuda', 'cpu'])
    p.add_argument('--num_workers',default=4, type=int)
    p.add_argument('--hid_dim',    default=64, type=int)
    p.add_argument('--num_layers', default=4, type=int)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
    device = torch.device('cuda:0' if args.device == 'cuda' else 'cpu')

    # ── load checkpoint ──────────────────────────────────────────────────────
    ckpt        = torch.load(args.checkpoint, map_location=device,
                             weights_only=False)
    angle_state = ckpt['angle_head_state_dict']
    hidden, n_joints = detect_head_dims(angle_state)
    print(f'==> Checkpoint: epoch={ckpt.get("epoch","?")}  '
          f'n_joints={n_joints}')

    cfg.merge_from_file(args.cfg)
    sk  = Human3WBDataset('data/h3wb_train.npz',
                          'data/h3wb_test.npz').skeleton()
    adj = adj_mx_from_skeleton(sk).to(device)

    model      = build_backbone(args, cfg, adj, device)
    angle_head = _AngleHead(69, hidden, n_joints).to(device)
    model.load_state_dict(ckpt['state_dict'])
    angle_head.load_state_dict(angle_state)

    # ── load data ────────────────────────────────────────────────────────────
    raw          = np.load(args.data_test, allow_pickle=True)
    exercise_ids = raw['exercise_ids']
    subject_ids  = raw['subject_ids']

    dataset = UIRPMDDataset(args.data_test)
    loader  = DataLoader(dataset, batch_size=args.batch_size,
                         shuffle=False, num_workers=args.num_workers,
                         pin_memory=(args.device == 'cuda'))
    print(f'==> Frames: {len(dataset)}  subjects: {np.unique(subject_ids)}')

    # ── inference ─────────────────────────────────────────────────────────────
    print('==> Running inference for per-subject std...')
    pred_angles, gt_angles = run_inference(
        model, angle_head, loader, n_joints, device)

    # ── per-subject per-exercise MAE → std across subjects ───────────────────
    unique_subs = np.unique(subject_ids)   # [8, 9] (0-indexed)
    unique_exs  = np.unique(exercise_ids)  # [0..9]

    # sub_ex_mae[sub_idx][ex_id] = mean ROM MAE for that subject × exercise
    sub_ex_mae = {}
    for sid in unique_subs:
        sub_ex_mae[sid] = {}
        for eid in unique_exs:
            mask = (subject_ids == sid) & (exercise_ids == eid)
            if not mask.any():
                continue
            sub_ex_mae[sid][eid] = float(
                np.abs(pred_angles[mask] - gt_angles[mask]).mean())

    # For each exercise: mean and std across subjects
    ex_means = []
    ex_stds  = []
    for eid in range(10):
        vals = [sub_ex_mae[sid][eid]
                for sid in unique_subs if eid in sub_ex_mae[sid]]
        ex_means.append(float(np.mean(vals)) if vals else 0.0)
        # Population std; with n=2, std = |v0 - v1| / sqrt(2)
        ex_stds.append(float(np.std(vals))   if len(vals) > 1 else 0.0)

    print('\nPer-subject breakdown:')
    for sid in unique_subs:
        vals_str = '  '.join(
            f'Ex{eid+1:02d}={sub_ex_mae[sid].get(eid,float("nan")):.2f}°'
            for eid in range(10))
        print(f'  S{sid+1:02d}: {vals_str}')

    print('\nExercise means ± std (n=2 subjects):')
    for i, (m, s) in enumerate(zip(ex_means, ex_stds)):
        print(f'  Ex{i+1:02d}: {m:.2f}° ± {s:.2f}°')

    # ── Figure 1 ─────────────────────────────────────────────────────────────
    print('\n==> Making Figure 1 (per-exercise)...')
    make_fig_per_exercise(ex_means, ex_stds, args.output_dir)

    # ── Figure 2 ─────────────────────────────────────────────────────────────
    print('==> Making Figure 2 (L/R asymmetry)...')
    make_fig_lr_asymmetry(args.output_dir)

    print(f'\nDone — saved to {args.output_dir}')


if __name__ == '__main__':
    main()
