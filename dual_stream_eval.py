"""
dual_stream_eval.py — Comprehensive evaluation of a saved DualStreamQualityNet checkpoint.

Usage:
    python dual_stream_eval.py --checkpoint results/dual_random_best_001.pt
    python dual_stream_eval.py --checkpoint results/dual_random_best_001.pt --out_dir results/eval_001/

Metrics computed:
  • Exercise classification   : confusion matrix, per-class P/R/F1, macro F1, top-1 accuracy
  • Validity classification   : confusion matrix, P/R/F1, ROC-AUC
  • Quality score regression  : MAD, RMSE, R², Pearson r, Spearman ρ  (overall + per-exercise)
  • Quality as binary clf     : calibration curve, ROC-AUC at threshold 0.5

Outputs (all saved to --out_dir):
  exercise_cm.png / .csv        — 10×10 exercise confusion matrix
  validity_cm.png / .csv        — 2×2 validity confusion matrix
  quality_scatter.png           — pred vs true scatter coloured by exercise
  quality_error_hist.png        — absolute-error distribution histogram
  roc_quality.png               — ROC curve treating quality > 0.5 as positive
  calibration.png               — reliability diagram (mean predicted vs fraction correct)
  per_exercise_mad.png          — per-exercise MAD bar chart vs paper baseline
  attention_per_exercise.png    — one sample attention trace per exercise
  summary.csv                   — scalar metrics
  per_exercise.csv              — per-exercise regression metrics
  classification_report.txt     — sklearn classification report (exercise + validity)
"""

import argparse
import os
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from dual_stream_quality import (
    J,
    EXERCISE_NAMES,
    PAPER_MAD,
    DualStreamDataset,
    DualStreamQualityNet,
    apply_gmm_scores,
    build_topology_adjacency,
    build_windows,
    fit_gmm_models,
)

warnings.filterwarnings("ignore", category=UserWarning)

EXERCISE_LABELS = [f"Ex{i+1:02d}\n({EXERCISE_NAMES[i]})" for i in range(10)]
VALIDITY_LABELS = ["Valid", "Invalid"]


# ─── Data helpers ─────────────────────────────────────────────────────────────

def _load_quality_labels(d, name):
    if 'quality_labels' in d:
        return d['quality_labels'].astype(np.int32)
    elif 'quality_scores' in d:
        return (d['quality_scores'] >= 0.5).astype(np.int32)
    raise KeyError(f"[{name}] NPZ has neither 'quality_labels' nor 'quality_scores'")


def _load_npz(path, tag):
    d = np.load(path, allow_pickle=True)
    return {
        'poses3d': d['poses_3d'],
        'rom':     d['rom_angles'],
        'ql':      _load_quality_labels(d, tag),
        'ex':      d['exercise_ids'],
        'subj':    d['subject_ids'],
        'gmm':     d['quality_scores'].astype(np.float32) if 'quality_scores' in d
                   else np.zeros(len(d['poses_3d']), dtype=np.float32),
    }


def rebuild_test_split(ckpt_args):
    """
    Reconstruct the same test data that was held out during training.
    Handles both split_mode='random' and split_mode='subject'.
    Returns dict with arrays for the test portion.
    """
    split_mode = ckpt_args.get('split_mode', 'subject')
    tr = _load_npz(ckpt_args['train_npz'], 'train')
    te = _load_npz(ckpt_args['test_npz'],  'test')

    if split_mode == 'random':
        all_p3d  = np.concatenate([tr['poses3d'], te['poses3d']], axis=0)
        all_rom  = np.concatenate([tr['rom'],     te['rom']],     axis=0)
        all_ql   = np.concatenate([tr['ql'],      te['ql']],      axis=0)
        all_ex   = np.concatenate([tr['ex'],      te['ex']],      axis=0)
        all_subj = np.concatenate([tr['subj'],    te['subj']],    axis=0)
        all_gmm  = np.concatenate([tr['gmm'],     te['gmm']],     axis=0)

        seed = ckpt_args.get('seed', 42)
        rng  = np.random.default_rng(seed)
        idx  = rng.permutation(len(all_p3d))
        n_train = int(0.8 * len(idx))
        n_val   = int(0.1 * len(idx))
        test_idx = idx[n_train + n_val:]

        return {
            'poses3d': all_p3d[test_idx],
            'rom':     all_rom[test_idx],
            'ql':      all_ql[test_idx],
            'ex':      all_ex[test_idx],
            'subj':    all_subj[test_idx],
            'gmm':     all_gmm[test_idx],
        }
    else:
        # Subject-level split — the test NPZ is the full test set
        return te


# ─── Inference ────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    ex_preds, ex_trues = [], []
    vc_preds, vc_trues = [], []
    qual_preds, gmm_trues, qual_labels = [], [], []
    attns = []

    for batch in loader:
        joints, rom, gmm_score, ex_label, qual_label, valid_label, _ = [
            b.to(device) for b in batch
        ]
        ex_logits, vc_logits, qual_score, attn = model(joints, rom, ex_label)

        ex_preds.append(ex_logits.argmax(1).cpu().numpy())
        ex_trues.append(ex_label.cpu().numpy())
        vc_preds.append(vc_logits.argmax(1).cpu().numpy())
        vc_trues.append(valid_label.cpu().numpy())
        qual_preds.append(qual_score.cpu().numpy())
        gmm_trues.append(gmm_score.cpu().numpy())
        qual_labels.append(qual_label.cpu().numpy())
        attns.append(attn.cpu().numpy())

    return {
        'ex_pred':    np.concatenate(ex_preds),
        'ex_true':    np.concatenate(ex_trues),
        'vc_pred':    np.concatenate(vc_preds),
        'vc_true':    np.concatenate(vc_trues),
        'qual_pred':  np.concatenate(qual_preds),
        'gmm_true':   np.concatenate(gmm_trues),
        'qual_label': np.concatenate(qual_labels),
        'attn':       np.concatenate(attns),
    }


# ─── Metrics & plots ──────────────────────────────────────────────────────────

def compute_scalar_metrics(res):
    qp, qt, ql = res['qual_pred'], res['gmm_true'], res['qual_label']
    ep, et     = res['ex_pred'],   res['ex_true']
    vp, vt     = res['vc_pred'],   res['vc_true']

    mae  = float(np.mean(np.abs(qp - qt)))
    rmse = float(np.sqrt(np.mean((qp - qt) ** 2)))
    r2   = float(r2_score(qt, qp))
    pr   = float(pearsonr(qt, qp)[0])
    sr   = float(spearmanr(qt, qp)[0])

    ec_acc = float((ep == et).mean())
    vc_acc = float((vp == vt).mean())

    vc_auc = float(roc_auc_score(vt, vp)) if len(np.unique(vt)) > 1 else float('nan')

    # Quality as binary classifier (threshold 0.5)
    qp_bin = (qp >= 0.5).astype(int)
    ql_bin = (ql >= 0.5).astype(int)           # ground truth binary
    q_auc  = float(roc_auc_score(ql_bin, qp)) if len(np.unique(ql_bin)) > 1 else float('nan')

    return {
        'qual_mae': mae, 'qual_rmse': rmse, 'qual_r2': r2,
        'qual_pearson_r': pr, 'qual_spearman_rho': sr,
        'ec_acc': ec_acc, 'vc_acc': vc_acc, 'vc_auc': vc_auc,
        'qual_auc': q_auc,
    }


def compute_per_exercise(res):
    qp, qt = res['qual_pred'], res['gmm_true']
    et     = res['ex_true']
    rows   = []
    for e in range(10):
        mask = et == e
        if mask.sum() == 0:
            rows.append({'ex': e, 'n': 0, 'mad': np.nan, 'rmse': np.nan,
                         'r2': np.nan, 'pearson_r': np.nan})
            continue
        p, t = qp[mask], qt[mask]
        rows.append({
            'ex':       e,
            'n':        int(mask.sum()),
            'mad':      float(np.mean(np.abs(p - t))),
            'rmse':     float(np.sqrt(np.mean((p - t) ** 2))),
            'r2':       float(r2_score(t, p)) if len(t) > 1 else np.nan,
            'pearson_r': float(pearsonr(t, p)[0]) if len(t) > 1 else np.nan,
        })
    return rows


# ─── Plot functions ───────────────────────────────────────────────────────────

def _save(fig, path):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_exercise_cm(res, out_dir):
    cm = confusion_matrix(res['ex_true'], res['ex_pred'], labels=list(range(10)))
    fig, ax = plt.subplots(figsize=(11, 9))
    disp = ConfusionMatrixDisplay(cm, display_labels=[f"Ex{i+1:02d}" for i in range(10)])
    disp.plot(ax=ax, colorbar=True, cmap='Blues', values_format='d')
    ax.set_title('Exercise Classification Confusion Matrix', fontsize=13)
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('True', fontsize=11)
    plt.xticks(rotation=45, ha='right')
    _save(fig, os.path.join(out_dir, 'exercise_cm.png'))

    np.savetxt(
        os.path.join(out_dir, 'exercise_cm.csv'), cm, fmt='%d',
        delimiter=',',
        header=','.join([f"Ex{i+1:02d}" for i in range(10)]),
        comments='',
    )
    print(f"  Saved: {os.path.join(out_dir, 'exercise_cm.csv')}")


def plot_validity_cm(res, out_dir):
    cm = confusion_matrix(res['vc_true'], res['vc_pred'], labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=VALIDITY_LABELS)
    disp.plot(ax=ax, colorbar=False, cmap='Greens', values_format='d')
    ax.set_title('Validity Classification Confusion Matrix', fontsize=12)
    _save(fig, os.path.join(out_dir, 'validity_cm.png'))

    np.savetxt(
        os.path.join(out_dir, 'validity_cm.csv'), cm, fmt='%d',
        delimiter=',', header='Valid,Invalid', comments='',
    )
    print(f"  Saved: {os.path.join(out_dir, 'validity_cm.csv')}")


def plot_quality_scatter(res, out_dir):
    qp, qt, et = res['qual_pred'], res['gmm_true'], res['ex_true']
    cmap = plt.get_cmap('tab10')

    fig, ax = plt.subplots(figsize=(7, 6))
    for e in range(10):
        mask = et == e
        if mask.sum() == 0:
            continue
        ax.scatter(qt[mask], qp[mask], alpha=0.35, s=8,
                   color=cmap(e), label=f"Ex{e+1:02d}")
    lo, hi = min(qt.min(), qp.min()), max(qt.max(), qp.max())
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1, label='Perfect')
    ax.set_xlabel('GMM Quality Score (true)', fontsize=11)
    ax.set_ylabel('Predicted Quality Score', fontsize=11)
    ax.set_title('Quality Score: Predicted vs True', fontsize=12)
    ax.legend(loc='upper left', fontsize=7, ncol=2)
    r = pearsonr(qt, qp)[0]
    ax.text(0.97, 0.03, f"r = {r:.3f}", transform=ax.transAxes,
            ha='right', va='bottom', fontsize=10)
    _save(fig, os.path.join(out_dir, 'quality_scatter.png'))


def plot_error_histogram(res, out_dir):
    err = np.abs(res['qual_pred'] - res['gmm_true'])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(err, bins=60, color='steelblue', edgecolor='white', linewidth=0.4)
    ax.axvline(err.mean(), color='red', lw=1.5, linestyle='--', label=f'Mean = {err.mean():.4f}')
    ax.axvline(np.median(err), color='orange', lw=1.5, linestyle=':', label=f'Median = {np.median(err):.4f}')
    ax.set_xlabel('Absolute Error', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Quality Score Absolute Error Distribution', fontsize=12)
    ax.legend()
    _save(fig, os.path.join(out_dir, 'quality_error_hist.png'))


def plot_roc_quality(res, out_dir):
    ql_bin = (res['qual_label'] >= 0.5).astype(int)
    if len(np.unique(ql_bin)) < 2:
        print("  [skip] ROC: only one class in quality labels")
        return
    fpr, tpr, _ = roc_curve(ql_bin, res['qual_pred'])
    auc = roc_auc_score(ql_bin, res['qual_pred'])

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f'AUC = {auc:.3f}')
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('ROC Curve — Quality Score (threshold 0.5)', fontsize=12)
    ax.legend(loc='lower right')
    _save(fig, os.path.join(out_dir, 'roc_quality.png'))


def plot_calibration(res, out_dir):
    ql_bin = (res['qual_label'] >= 0.5).astype(int)
    if len(np.unique(ql_bin)) < 2:
        print("  [skip] Calibration: only one class in quality labels")
        return
    try:
        frac_pos, mean_pred = calibration_curve(ql_bin, res['qual_pred'], n_bins=10)
    except Exception as e:
        print(f"  [skip] Calibration: {e}")
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(mean_pred, frac_pos, 'o-', label='Model')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfect')
    ax.set_xlabel('Mean Predicted Probability', fontsize=11)
    ax.set_ylabel('Fraction of Positives', fontsize=11)
    ax.set_title('Calibration Curve (Quality Score)', fontsize=12)
    ax.legend()
    _save(fig, os.path.join(out_dir, 'calibration.png'))


def plot_per_exercise_mad(per_ex_rows, out_dir):
    names = [EXERCISE_NAMES[r['ex']] for r in per_ex_rows]
    mads  = [r['mad'] for r in per_ex_rows]
    paper = PAPER_MAD

    x = np.arange(len(names))
    w = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))
    bars1 = ax.bar(x - w / 2, mads,  w, label='Dual-Stream (ours)', color='steelblue')
    bars2 = ax.bar(x + w / 2, paper, w, label='Paper baseline',      color='salmon',    alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Ex{r['ex']+1:02d}\n({n})" for r, n in zip(per_ex_rows, names)],
                       fontsize=9)
    ax.set_ylabel('MAD', fontsize=11)
    ax.set_title('Per-Exercise Quality Score MAD vs Paper Baseline', fontsize=12)
    ax.legend()

    for bar in bars1:
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.0003, f'{h:.4f}',
                    ha='center', va='bottom', fontsize=7)
    _save(fig, os.path.join(out_dir, 'per_exercise_mad.png'))


def plot_attention_per_exercise(model, test_dataset, device, out_dir, windows_per_row=3):
    """Plot attention weight traces, one row per exercise."""
    from collections import defaultdict

    model.eval()
    ex_to_indices = defaultdict(list)
    for i, w in enumerate(test_dataset.windows):
        ex_to_indices[w['exercise']].append(i)

    fig, axes = plt.subplots(10, windows_per_row,
                             figsize=(windows_per_row * 4, 10 * 2.2),
                             sharey=False)

    with torch.no_grad():
        for e in range(10):
            idxs = ex_to_indices[e][:windows_per_row]
            for col, idx in enumerate(idxs):
                ax = axes[e][col]
                joints, rom, gmm_score, ex_label, ql, vl, _ = test_dataset[idx]
                joints   = joints.unsqueeze(0).to(device)
                rom      = rom.unsqueeze(0).to(device)
                ex_label = ex_label.unsqueeze(0).to(device)
                _, _, qpred, attn = model(joints, rom, ex_label)
                ax.plot(attn[0].cpu().numpy(), linewidth=0.9)
                ax.set_title(f"Ex{e+1:02d} q_pred={qpred.item():.2f} "
                             f"q_true={gmm_score.item():.2f}", fontsize=7)
                ax.set_ylim(0, None)
                ax.tick_params(labelsize=6)
            # blank unused cols
            for col in range(len(idxs), windows_per_row):
                axes[e][col].axis('off')
            axes[e][0].set_ylabel(f"Ex{e+1:02d}\n({EXERCISE_NAMES[e]})", fontsize=8)

    fig.suptitle('ROM-Stream Attention Weights per Exercise', fontsize=12, y=1.01)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, 'attention_per_exercise.png'))


# ─── Save text reports ────────────────────────────────────────────────────────

def save_classification_report(res, out_dir):
    path = os.path.join(out_dir, 'classification_report.txt')
    lines = []

    lines.append("=" * 60)
    lines.append("EXERCISE CLASSIFICATION REPORT")
    lines.append("=" * 60)
    lines.append(classification_report(
        res['ex_true'], res['ex_pred'],
        target_names=[f"Ex{i+1:02d}({EXERCISE_NAMES[i]})" for i in range(10)],
        digits=4,
    ))

    lines.append("=" * 60)
    lines.append("VALIDITY CLASSIFICATION REPORT")
    lines.append("=" * 60)
    lines.append(classification_report(
        res['vc_true'], res['vc_pred'],
        target_names=VALIDITY_LABELS,
        digits=4,
    ))

    text = '\n'.join(lines)
    print(text)
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    with open(path, 'w') as f:
        f.write(text)
    print(f"  Saved: {path}")


def save_summary_csv(scalars, out_dir):
    path = os.path.join(out_dir, 'summary.csv')
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    with open(path, 'w') as f:
        f.write('metric,value\n')
        for k, v in scalars.items():
            f.write(f'{k},{v:.6f}\n')
    print(f"  Saved: {path}")


def save_per_exercise_csv(rows, out_dir):
    path = os.path.join(out_dir, 'per_exercise.csv')
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    with open(path, 'w') as f:
        f.write('exercise,name,n_windows,mad,rmse,r2,pearson_r,paper_mad,delta_mad\n')
        for r in rows:
            e = r['ex']
            f.write(f"Ex{e+1:02d},{EXERCISE_NAMES[e]},{r['n']},"
                    f"{r['mad']:.6f},{r['rmse']:.6f},{r['r2']:.6f},{r['pearson_r']:.6f},"
                    f"{PAPER_MAD[e]:.4f},{r['mad']-PAPER_MAD[e]:+.6f}\n")
    print(f"  Saved: {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Comprehensive eval of DualStreamQualityNet checkpoint')
    p.add_argument('--checkpoint',  required=True,
                   help='Path to .pt checkpoint (e.g. results/dual_random_best_001.pt)')
    p.add_argument('--out_dir',     default='results/eval',
                   help='Directory to write all outputs (default: results/eval)')
    p.add_argument('--batch_size',  type=int, default=64)
    p.add_argument('--num_workers', type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Load checkpoint ──────────────────────────────────────────────────────
    print(f"\nLoading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = ckpt['args']
    rom_mean  = ckpt['rom_mean']   # (10, 12)
    rom_std   = ckpt['rom_std']    # (10, 12)

    print(f"  Trained for {ckpt['epoch']} epochs  |  val_MAD = {float(ckpt['val_mad']):.6f}")
    print(f"  split_mode = {ckpt_args.get('split_mode','subject')}  |  "
          f"seed = {ckpt_args.get('seed', 42)}")

    # ── Rebuild test split ───────────────────────────────────────────────────
    print("\nRebuilding test split...")
    test_data = rebuild_test_split(ckpt_args)
    n_test = len(test_data['poses3d'])
    print(f"  Test frames: {n_test}")

    # ── Build test windows ───────────────────────────────────────────────────
    window_size = ckpt_args.get('window_size', 100)
    stride      = ckpt_args.get('stride', 50)

    print("Building test windows...")
    test_windows = build_windows(
        test_data['poses3d'], test_data['rom'],
        test_data['gmm'],     test_data['ql'],
        test_data['ex'],      test_data['subj'],
        window_size, stride, min_len=50,
        rom_mean=rom_mean, rom_std=rom_std,
        augment=False, generate_invalid=False,
    )
    print(f"  Test windows: {len(test_windows)}")

    test_ds     = DualStreamDataset(test_windows)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers,
                             pin_memory=(device.type == 'cuda'))

    # ── Build model ──────────────────────────────────────────────────────────
    print("\nBuilding model...")
    hidden_dim = ckpt_args.get('hidden_dim', 64)
    M          = ckpt_args.get('M', 100)
    A_topology = build_topology_adjacency(J)

    model = DualStreamQualityNet(
        hidden_dim=hidden_dim, M=M, n_joints=J,
        n_exercises=10, A_topology=A_topology,
        rom_guided_inits=None,      # overwritten by state_dict below
    ).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    # ── Run inference ────────────────────────────────────────────────────────
    print("\nRunning inference on test set...")
    res = run_inference(model, test_loader, device)
    print(f"  Samples: {len(res['ex_true'])}")

    # ── Scalar metrics ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SCALAR METRICS")
    print("=" * 60)
    scalars = compute_scalar_metrics(res)
    for k, v in scalars.items():
        print(f"  {k:<25} {v:.6f}")

    # ── Per-exercise metrics ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PER-EXERCISE QUALITY METRICS")
    print("=" * 60)
    per_ex = compute_per_exercise(res)
    print(f"{'Exercise':<12} {'N':>6}  {'MAD':>8}  {'RMSE':>8}  {'R²':>7}  "
          f"{'r':>7}  {'Paper':>7}  {'Δ MAD':>8}")
    print("-" * 72)
    for r in per_ex:
        e = r['ex']
        print(f"Ex{e+1:02d} ({EXERCISE_NAMES[e]:<4}) {r['n']:>6}  "
              f"{r['mad']:>8.4f}  {r['rmse']:>8.4f}  {r['r2']:>7.4f}  "
              f"{r['pearson_r']:>7.4f}  {PAPER_MAD[e]:>7.4f}  "
              f"{r['mad']-PAPER_MAD[e]:>+8.4f}")
    avg_mad = np.nanmean([r['mad'] for r in per_ex])
    print("-" * 72)
    print(f"{'AVERAGE':<12} {'':>6}  {avg_mad:>8.4f}")

    # ── Save text reports ────────────────────────────────────────────────────
    print("\nSaving reports...")
    save_classification_report(res, args.out_dir)
    save_summary_csv(scalars, args.out_dir)
    save_per_exercise_csv(per_ex, args.out_dir)

    # ── Plots ────────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_exercise_cm(res, args.out_dir)
    plot_validity_cm(res, args.out_dir)
    plot_quality_scatter(res, args.out_dir)
    plot_error_histogram(res, args.out_dir)
    plot_roc_quality(res, args.out_dir)
    plot_calibration(res, args.out_dir)
    plot_per_exercise_mad(per_ex, args.out_dir)
    plot_attention_per_exercise(model, test_ds, device, args.out_dir)

    print(f"\nAll outputs written to: {args.out_dir}/")
    print("Done.")


if __name__ == '__main__':
    main()
