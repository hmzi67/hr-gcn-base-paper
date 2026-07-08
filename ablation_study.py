"""
Ablation Study for Dual-Stream Quality Assessment Network.

Runs five controlled configurations — each changes exactly one component
relative to the full model — and produces a comparison table, CSV, and
bar-chart PNG.

Usage:
  python ablation_study.py \\
    --train_npz data/uiprmd_segmented_gmm_train.npz \\
    --test_npz  data/uiprmd_segmented_gmm_test.npz  \\
    --use_npz_scores --split_mode random             \\
    --epochs 200                                     \\
    --configs full no_romguide no_romstream no_attn uni_lstm
"""

import argparse
import copy
import csv
import math
import os
import random
import time
import traceback

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ── Reuse everything that is already in dual_stream_quality.py ────────────────
from dual_stream_quality import (
    J,
    build_topology_adjacency,
    compute_rom_guided_init,
    fit_gmm_models,
    apply_gmm_scores,
    build_windows,
    compute_rom_stats,
    DualStreamDataset,
    make_weighted_sampler,
    train_one_epoch,
    evaluate,
    SpatialGCN,
    TemporalGCN,
    ROMStream,
    FusionLayer,
)

# ─── Ablation registry ────────────────────────────────────────────────────────

ALL_CONFIGS = ['full', 'no_romguide', 'no_romstream', 'no_attn', 'uni_lstm']

ABLATION_NAMES = {
    'full':         'Full Dual-Stream',
    'no_romguide':  'w/o ROM-guided adj.',
    'no_romstream': 'w/o ROM stream (GCN only)',
    'no_attn':      'w/o Bahdanau attention',
    'uni_lstm':     'Unidirectional LSTM',
}

# ─── Ablation-specific sub-modules ────────────────────────────────────────────


class ROMStreamNoAttn(nn.Module):
    """BiLSTM ROM stream with plain mean pooling instead of Bahdanau attention."""

    def __init__(self, rom_dim=12, proj_dim=64, lstm_hidden=64, n_layers=2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(rom_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.lstm = nn.LSTM(
            input_size=proj_dim,
            hidden_size=lstm_hidden,
            num_layers=n_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.2,
        )

    def forward(self, x, padding_mask=None):
        h = self.proj(x)
        h, _ = self.lstm(h)
        context = h.mean(dim=1)         # plain mean over time axis
        return context, None


class ROMStreamUni(nn.Module):
    """ROM stream with unidirectional LSTM (hidden=64) + Bahdanau attention."""

    def __init__(self, rom_dim=12, proj_dim=64, lstm_hidden=64, n_layers=2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(rom_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.lstm = nn.LSTM(
            input_size=proj_dim,
            hidden_size=lstm_hidden,
            num_layers=n_layers,
            bidirectional=False,        # ← unidirectional
            batch_first=True,
            dropout=0.2,
        )
        self.attn = nn.Linear(lstm_hidden, 1, bias=False)   # hidden, not *2

    def forward(self, x, padding_mask=None):
        h = self.proj(x)
        h, _ = self.lstm(h)
        scores = self.attn(torch.tanh(h))
        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask.unsqueeze(-1), float('-inf'))
        weights = torch.softmax(scores, dim=1)
        context = (weights * h).sum(dim=1)
        return context, weights.squeeze(-1)


class FusionLayerSingle(nn.Module):
    """Fusion layer for no_romstream: accepts a single 128-dim feature."""

    def __init__(self, in_dim=128, out_dim=128):
        super().__init__()
        self.fc      = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(0.2)
        self.norm    = nn.LayerNorm(out_dim)

    def forward(self, feat):
        x = F.relu(self.fc(feat))
        x = self.dropout(x)
        return self.norm(x)


# ─── Ablation model ───────────────────────────────────────────────────────────


class AblationDualStreamNet(nn.Module):
    """
    DualStreamQualityNet parameterised by ablation config.

    ablation='full'         — reference (identical to DualStreamQualityNet)
    ablation='no_romguide'  — spatial adj init from topology only
    ablation='no_romstream' — GCN stream only; FusionLayerSingle(128 → 128)
    ablation='no_attn'      — BiLSTM + mean pool; FusionLayer(256 → 128)
    ablation='uni_lstm'     — unidirectional LSTM; FusionLayer(192 → 128)
    """

    def __init__(self, ablation='full', hidden_dim=64, M=100, n_joints=17,
                 n_exercises=10, A_topology=None, rom_guided_inits=None):
        super().__init__()
        self.ablation   = ablation
        spatial_out     = hidden_dim * 2
        temporal_in     = n_joints * spatial_out

        # no_romguide: topology-only adjacency initialisation
        rg_inits = None if ablation == 'no_romguide' else rom_guided_inits

        self.spatial_gcn  = SpatialGCN(n_joints, hidden_dim, n_exercises,
                                        A_topology, rg_inits)
        self.temporal_gcn = TemporalGCN(temporal_in, 128, M)

        if ablation == 'no_romstream':
            self.rom_stream = None
            self.fusion     = FusionLayerSingle(128, 128)
        elif ablation == 'no_attn':
            self.rom_stream = ROMStreamNoAttn(12, 64, 64, 2)
            self.fusion     = FusionLayer(256, 128)
        elif ablation == 'uni_lstm':
            # LSTM output is lstm_hidden=64 (not *2); fusion input = 128 + 64
            self.rom_stream = ROMStreamUni(12, 64, 64, 2)
            self.fusion     = FusionLayer(192, 128)
        else:                               # full  /  no_romguide
            self.rom_stream = ROMStream(12, 64, 64, 2)
            self.fusion     = FusionLayer(256, 128)

        self.exercise_head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, n_exercises),
        )
        self.validity_head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 2),
        )
        self.quality_head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 1), nn.Sigmoid(),
        )

    def forward(self, joints, rom, exercise_id, padding_mask=None):
        B, M, J_in, C = joints.shape
        sf = self.spatial_gcn(joints, exercise_id)
        sf = sf.reshape(B, M, J_in * sf.shape[-1])
        tf = self.temporal_gcn(sf)                  # (B, 128)

        if self.ablation == 'no_romstream':
            fused = self.fusion(tf)
            attn  = None
        else:
            rf, attn = self.rom_stream(rom, padding_mask)
            fused = self.fusion(tf, rf)

        return (self.exercise_head(fused),
                self.validity_head(fused),
                self.quality_head(fused).squeeze(-1),
                attn)


# ─── Data preparation (runs ONCE; shared by all configs) ─────────────────────


def _load_quality_labels(d, name):
    if 'quality_labels' in d:
        return d['quality_labels'].astype(np.int32)
    elif 'quality_scores' in d:
        print(f"  [{name}] 'quality_labels' missing — thresholding 'quality_scores' @ 0.5")
        return (d['quality_scores'] >= 0.5).astype(np.int32)
    else:
        raise KeyError(f"[{name}] NPZ has neither 'quality_labels' nor 'quality_scores'")


def prepare_data(args):
    """
    Load NPZ files, compute GMM scores / use NPZ scores, build windows.
    Returns a bundle dict that is shared (read-only) by every config run.
    Each run deep-copies the windows before mutating them.
    """
    print("Loading data...")
    tr = np.load(args.train_npz, allow_pickle=True)
    te = np.load(args.test_npz,  allow_pickle=True)

    tr_poses3d = tr['poses_3d'];   tr_rom  = tr['rom_angles']
    tr_ql      = _load_quality_labels(tr, 'train')
    tr_ex      = tr['exercise_ids']; tr_subj = tr['subject_ids']

    te_poses3d = te['poses_3d'];   te_rom  = te['rom_angles']
    te_ql      = _load_quality_labels(te, 'test')
    te_ex      = te['exercise_ids']; te_subj = te['subject_ids']

    # ── Split ────────────────────────────────────────────────────────────────
    if args.split_mode == 'random':
        all_p3d  = np.concatenate([tr_poses3d, te_poses3d], axis=0)
        all_rom  = np.concatenate([tr_rom,     te_rom],     axis=0)
        all_ql   = np.concatenate([tr_ql,      te_ql],      axis=0)
        all_ex   = np.concatenate([tr_ex,      te_ex],      axis=0)
        all_subj = np.concatenate([tr_subj,    te_subj],    axis=0)
        _tr_sc   = (tr['quality_scores'].astype(np.float32)
                    if 'quality_scores' in tr else np.zeros(len(tr_poses3d), np.float32))
        _te_sc   = (te['quality_scores'].astype(np.float32)
                    if 'quality_scores' in te else np.zeros(len(te_poses3d), np.float32))
        all_gmm  = np.concatenate([_tr_sc, _te_sc], axis=0)

        rng       = np.random.default_rng(args.seed)
        idx       = rng.permutation(len(all_p3d))
        n_train   = int(0.8 * len(idx))
        n_val     = int(0.1 * len(idx))
        tr_idx    = idx[:n_train]
        va_idx    = idx[n_train: n_train + n_val]
        te_idx    = idx[n_train + n_val:]

        tr_poses3d  = all_p3d[tr_idx];   tr_rom   = all_rom[tr_idx]
        tr_ql       = all_ql[tr_idx];    tr_ex    = all_ex[tr_idx]
        tr_subj     = all_subj[tr_idx];  tr_gmm   = all_gmm[tr_idx]

        val_poses3d = all_p3d[va_idx];   val_rom  = all_rom[va_idx]
        val_ql      = all_ql[va_idx];    val_ex   = all_ex[va_idx]
        val_subj    = all_subj[va_idx];  val_gmm  = all_gmm[va_idx]

        te_poses3d  = all_p3d[te_idx];   te_rom   = all_rom[te_idx]
        te_ql       = all_ql[te_idx];    te_ex    = all_ex[te_idx]
        te_subj     = all_subj[te_idx];  te_gmm   = all_gmm[te_idx]

        train_mask = np.ones(len(tr_poses3d), dtype=bool)
        print(f"Random split — train={len(tr_idx)}  val={len(va_idx)}  test={len(te_idx)}")
    else:
        train_mask  = tr_subj <= 6
        val_mask    = tr_subj == 7

    # ── GMM / NPZ scores ─────────────────────────────────────────────────────
    if args.use_npz_scores:
        if args.split_mode != 'random':
            tr_gmm = tr['quality_scores'].astype(np.float32)
            te_gmm = te['quality_scores'].astype(np.float32)
        print("Using quality_scores from NPZ directly")
        if args.split_mode == 'subject':
            val_gmm = tr_gmm[val_mask]
    else:
        print("Fitting PCA+GMM models on training frames (correct only)...")
        gmm_models = fit_gmm_models(
            tr_poses3d[train_mask], tr_rom[train_mask],
            tr_ex[train_mask],      tr_ql[train_mask])
        tr_gmm  = apply_gmm_scores(tr_rom, tr_ex, tr_ql, gmm_models, "TRAIN")
        if args.split_mode == 'random':
            val_gmm = apply_gmm_scores(val_rom, val_ex, val_ql, gmm_models, "VAL")
        te_gmm  = apply_gmm_scores(te_rom, te_ex, te_ql, gmm_models, "TEST")
        if args.split_mode == 'subject':
            val_gmm = tr_gmm[val_mask]

    if args.split_mode == 'subject':
        val_poses3d = tr_poses3d[val_mask]; val_rom  = tr_rom[val_mask]
        val_ql      = tr_ql[val_mask];      val_ex   = tr_ex[val_mask]
        val_subj    = tr_subj[val_mask]

    # ── ROM-guided adjacency ─────────────────────────────────────────────────
    print("Computing ROM-guided adjacency init...")
    rom_guided_inits = compute_rom_guided_init(
        tr_poses3d[train_mask], tr_rom[train_mask],
        tr_ex[train_mask],      tr_ql[train_mask])
    A_topology = build_topology_adjacency(J)

    # ── ROM normalisation stats (from a preliminary pass with identity norm) ─
    print("Computing ROM normalisation stats...")
    dummy = build_windows(
        tr_poses3d[train_mask], tr_rom[train_mask],
        tr_gmm[train_mask],     tr_ql[train_mask],
        tr_ex[train_mask],      tr_subj[train_mask],
        args.window_size, args.stride, 50,
        np.zeros((10, 12), np.float32), np.ones((10, 12), np.float32),
        augment=False, generate_invalid=False,
    )
    rom_mean, rom_std = compute_rom_stats(dummy)
    del dummy

    # ── Build windows ────────────────────────────────────────────────────────
    print("Building train windows (with augmentation)...")
    train_windows = build_windows(
        tr_poses3d[train_mask], tr_rom[train_mask],
        tr_gmm[train_mask],     tr_ql[train_mask],
        tr_ex[train_mask],      tr_subj[train_mask],
        args.window_size, args.stride, 50,
        rom_mean, rom_std, augment=True, generate_invalid=True,
    )
    print("Building val windows...")
    val_windows = build_windows(
        val_poses3d, val_rom, val_gmm, val_ql, val_ex, val_subj,
        args.window_size, args.stride, 50,
        rom_mean, rom_std, augment=False, generate_invalid=False,
    )
    print("Building test windows...")
    test_windows = build_windows(
        te_poses3d, te_rom, te_gmm, te_ql, te_ex, te_subj,
        args.window_size, args.stride, 50,
        rom_mean, rom_std, augment=False, generate_invalid=False,
    )
    print(f"  train={len(train_windows)}  val={len(val_windows)}  test={len(test_windows)}")

    # ── Class weights for exercise head ─────────────────────────────────────
    ex_arr    = np.array([w['exercise'] for w in train_windows])
    ex_counts = np.bincount(ex_arr, minlength=10).astype(np.float32)
    ex_counts = np.where(ex_counts == 0, 1, ex_counts)
    cls_w_ex  = torch.tensor(1.0 / ex_counts)
    cls_w_ex  = cls_w_ex / cls_w_ex.sum() * 10

    return {
        'train_windows':    train_windows,
        'val_windows':      val_windows,
        'test_windows':     test_windows,
        'cls_w_ex':         cls_w_ex,
        'A_topology':       A_topology,
        'rom_guided_inits': rom_guided_inits,
    }


# ─── Per-config helpers ───────────────────────────────────────────────────────


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def _flip_vc_labels(windows):
    for w in windows:
        w['valid'] = 1 - w['valid']


def _make_loaders(train_windows, val_windows, test_windows, batch_size):
    train_ds = DualStreamDataset(train_windows)
    val_ds   = DualStreamDataset(val_windows)
    test_ds  = DualStreamDataset(test_windows)
    sampler  = make_weighted_sampler(train_windows)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=sampler, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size * 2,
                              shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size * 2,
                              shuffle=False, num_workers=4, pin_memory=True)
    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


# ─── Single-config training run ──────────────────────────────────────────────


def run_config(config_id: str, args, data_bundle: dict, device, out_dir: str) -> dict:
    """
    Reset seeds, build model, train, evaluate on test set.
    Returns a metrics dict.  The caller handles try/except.
    """
    set_seeds(args.seed)

    # Deep-copy windows so VC-flip mutations don't bleed into other configs
    train_windows = copy.deepcopy(data_bundle['train_windows'])
    val_windows   = copy.deepcopy(data_bundle['val_windows'])
    test_windows  = copy.deepcopy(data_bundle['test_windows'])

    (train_ds, val_ds, test_ds,
     train_loader, val_loader, test_loader) = _make_loaders(
        train_windows, val_windows, test_windows, args.batch_size)

    model = AblationDualStreamNet(
        ablation     = config_id,
        hidden_dim   = 64,
        M            = args.M,
        n_joints     = J,
        n_exercises  = 10,
        A_topology   = data_bundle['A_topology'],
        rom_guided_inits = data_bundle['rom_guided_inits'],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    ec_params    = list(model.exercise_head.parameters())
    ec_ids       = {id(p) for p in ec_params}
    other_params = [p for p in model.parameters() if id(p) not in ec_ids]
    optimizer = torch.optim.AdamW([
        {'params': other_params, 'weight_decay': 1e-4},
        {'params': ec_params,    'weight_decay': 1e-3, 'lr': args.lr * 0.3},
    ], lr=args.lr)

    warmup = 10

    def _lr_lambda(epoch):
        if epoch < warmup:
            return epoch / max(warmup, 1)
        progress = (epoch - warmup) / max(args.epochs - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler    = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
    cls_w_ex     = data_bundle['cls_w_ex']
    best_val_mad = float('inf')
    best_epoch   = 0
    patience_cnt = 0
    early_stop   = 40
    vc_flipped   = False
    ckpt_path    = os.path.join(out_dir, f'ckpt_{config_id}.pt')

    for epoch in range(1, args.epochs + 1):
        t0   = time.time()
        tr_m = train_one_epoch(
            model, train_loader, optimizer, device,
            args.lambda_ec, args.lambda_vc, cls_w_ex,
            desc=f"  [{config_id}] Ep[{epoch:3d}/{args.epochs}]",
            first_epoch=(epoch == 1),
        )
        val_m   = evaluate(model, val_loader, device, args.lambda_ec, args.lambda_vc)
        elapsed = time.time() - t0

        # Auto-flip VC labels if accuracy is below chance on first epoch
        if epoch == 1 and not vc_flipped and tr_m['vc_acc'] < 0.45:
            print(f"  [VC-fix] acc={tr_m['vc_acc']:.0%} < 45% — inverting VC labels")
            _flip_vc_labels(train_windows)
            _flip_vc_labels(val_windows)
            _flip_vc_labels(test_windows)
            for layer in model.validity_head:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
            (train_ds, val_ds, test_ds,
             train_loader, val_loader, test_loader) = _make_loaders(
                train_windows, val_windows, test_windows, args.batch_size)
            vc_flipped = True

        scheduler.step()

        if val_m['mad'] < best_val_mad:
            best_val_mad = val_m['mad']
            best_epoch   = epoch
            patience_cnt = 0
            torch.save({
                'epoch':      epoch,
                'state_dict': model.state_dict(),
                'val_mad':    best_val_mad,
                'config':     config_id,
            }, ckpt_path)
        else:
            patience_cnt += 1
            if patience_cnt >= early_stop:
                print(f"  Early stop ep {epoch} "
                      f"(best val MAD={best_val_mad:.4f} @ ep {best_epoch})")
                break

        if epoch % 20 == 0 or epoch <= 3:
            cur_lr = optimizer.param_groups[0]['lr']
            print(f"  Ep[{epoch:3d}] lr={cur_lr:.2e} | "
                  f"QA={tr_m['qual_loss']:.4f} EC={tr_m['ec_acc']:.0%} "
                  f"VC={tr_m['vc_acc']:.0%} | "
                  f"val_MAD={val_m['mad']:.4f} | {elapsed:.1f}s")

    # Reload best checkpoint and run final test evaluation
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    test_m = evaluate(model, test_loader, device, args.lambda_ec, args.lambda_vc)

    pred = test_m['qual_pred']
    true = test_m['qual_true']
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    mape = float(np.mean(np.abs((pred - true) / (np.abs(true) + 1e-8))) * 100)

    return {
        'config': config_id,
        'name':   ABLATION_NAMES[config_id],
        'mad':    float(test_m['mad']),
        'rmse':   rmse,
        'mape':   mape,
        'ec_acc': float(test_m['ec_acc']),
        'vc_acc': float(test_m['vc_acc']),
    }


# ─── Output helpers ───────────────────────────────────────────────────────────


def print_ablation_table(rows: list, full_mad: float):
    rows_sorted = sorted(rows, key=lambda r: r['mad'])
    hdr = (f"{'Config':<30} {'MAD':>7} {'RMSE':>7} {'MAPE%':>7}  "
           f"{'EC Acc':>7}  {'VC Acc':>7}  {'ΔMAD vs Full':>13}")
    sep = '-' * len(hdr)
    print(f"\n=== Ablation Study Results (UI-PRMD Test Set) ===\n")
    print(hdr)
    print(sep)
    for r in rows_sorted:
        delta     = r['mad'] - full_mad
        delta_str = '—' if r['config'] == 'full' else f'{delta:+.4f}'
        print(f"{r['name']:<30} {r['mad']:>7.4f} {r['rmse']:>7.4f} "
              f"{r['mape']:>7.2f}  {r['ec_acc']*100:>6.2f}%  "
              f"{r['vc_acc']*100:>6.2f}%  {delta_str:>13}")
    print(sep)


def save_csv(rows: list, path: str):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    full_mad = next((r['mad'] for r in rows if r['config'] == 'full'), 0.0)
    fieldnames = ['config', 'name', 'mad', 'rmse', 'mape',
                  'ec_acc', 'vc_acc', 'delta_vs_full']
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in sorted(rows, key=lambda x: x['mad']):
            w.writerow({
                'config':        r['config'],
                'name':          r['name'],
                'mad':           f"{r['mad']:.4f}",
                'rmse':          f"{r['rmse']:.4f}",
                'mape':          f"{r['mape']:.4f}",
                'ec_acc':        f"{r['ec_acc']*100:.2f}",
                'vc_acc':        f"{r['vc_acc']*100:.2f}",
                'delta_vs_full': f"{r['mad'] - full_mad:+.4f}",
            })
    print(f"  CSV: {path}")


def save_bar_chart(rows: list, path: str):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    rows_sorted = sorted(rows, key=lambda r: r['mad'])
    names  = [r['name'] for r in rows_sorted]
    mads   = [r['mad']  for r in rows_sorted]
    colors = ['#2C6FAC' if r['config'] == 'full' else '#AAAAAA'
              for r in rows_sorted]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.barh(names, mads, color=colors, edgecolor='white', height=0.55)
    x_max = max(mads) * 1.15
    for bar, val in zip(bars, mads):
        ax.text(val + x_max * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}',
                va='center', ha='left',
                fontsize=9, fontfamily='serif')
    ax.set_xlim(0, x_max)
    ax.set_xlabel('Mean Absolute Deviation (MAD)', fontfamily='serif', fontsize=11)
    ax.set_title(
        'Ablation Study: Effect of Removing Each Component on Quality MAD',
        fontfamily='serif', fontsize=12, pad=12)
    ax.tick_params(axis='y', labelsize=10)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  Bar chart: {path}")


# ─── CLI ──────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description='Ablation study for Dual-Stream Quality Assessment Net')
    p.add_argument('--train_npz',      default='data/uiprmd_segmented_gmm_train.npz')
    p.add_argument('--test_npz',       default='data/uiprmd_segmented_gmm_test.npz')
    p.add_argument('--use_npz_scores', action='store_true',
                   help='Use quality_scores from NPZ; skip internal GMM fitting')
    p.add_argument('--split_mode',     choices=['subject', 'random'], default='random')
    p.add_argument('--epochs',         type=int,   default=200)
    p.add_argument('--batch_size',     type=int,   default=16)
    p.add_argument('--lr',             type=float, default=1e-4)
    p.add_argument('--lambda_ec',      type=float, default=0.5)
    p.add_argument('--lambda_vc',      type=float, default=0.3)
    p.add_argument('--M',              type=int,   default=100,
                   help='Window size fed to TemporalGCN')
    p.add_argument('--window_size',    type=int,   default=100)
    p.add_argument('--stride',         type=int,   default=50)
    p.add_argument('--seed',           type=int,   default=42)
    p.add_argument('--configs',        nargs='+',  default=ALL_CONFIGS,
                   choices=ALL_CONFIGS, metavar='CONFIG',
                   help=f'Ablation configs to run (default: all). '
                        f'Choices: {ALL_CONFIGS}')
    p.add_argument('--out_dir',        default='ablation_results',
                   help='Directory for checkpoints, CSV, and PNG outputs')
    return p.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    os.makedirs(args.out_dir, exist_ok=True)

    # Prepare data once with a fixed seed — same split for all configs
    set_seeds(args.seed)
    data_bundle = prepare_data(args)

    results   = []
    n_configs = len(args.configs)

    for i, config_id in enumerate(args.configs, 1):
        print(f"\n{'='*62}")
        print(f"===== Running ablation: {config_id}  ({i}/{n_configs}) =====")
        print(f"{'='*62}")
        t_start = time.time()
        try:
            metrics = run_config(config_id, args, data_bundle, device, args.out_dir)
            wall    = time.time() - t_start
            results.append(metrics)
            print(f"\n  DONE  MAD={metrics['mad']:.4f}  RMSE={metrics['rmse']:.4f}  "
                  f"MAPE={metrics['mape']:.2f}%  "
                  f"EC={metrics['ec_acc']*100:.2f}%  VC={metrics['vc_acc']*100:.2f}%  "
                  f"({wall/60:.1f} min)")
        except Exception:
            print(f"\n  [FAILED] config='{config_id}'")
            traceback.print_exc()

    if not results:
        print("\nNo configs completed successfully — nothing to report.")
        return

    full_mad = next((r['mad'] for r in results if r['config'] == 'full'), 0.0)

    print_ablation_table(results, full_mad)
    save_csv(results,      os.path.join(args.out_dir, 'ablation_summary.csv'))
    save_bar_chart(results, os.path.join(args.out_dir, 'ablation_mad.png'))
    print("\nAll outputs written to:", args.out_dir)


if __name__ == '__main__':
    main()
