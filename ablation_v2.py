"""
ablation_v2.py — Rigorous component-wise ablation for the Dual-Stream Quality model.

Runs on the EXACT settings of the reported full-model run:
    segmented GMM NPZs, --use_npz_scores, --split_mode random,
    epochs=200, batch=16, lr=1e-4, hidden=64, M=100, window=100, stride=50,
    lambda_ec=0.5, lambda_vc=0.3.

Why this differs from the first ablation attempt (and why those numbers looked backwards):
  1. Reports TEST-set MAD/RMSE/MAPE (the thesis metric) for every config,
     NOT validation MAD. The val split is only ~726 windows -> noisy & optimistically
     biased because the checkpoint is chosen on best val MAD.
  2. Averages each config over multiple model seeds (mean +/- std) so a 0.001-0.002
     difference is not mistaken for a real effect.
  3. Adds the ablations that were missing: no ROM stream (GCN-only),
     no temporal GCN, and multi-task decomposition (Q+E only, Q+V only).

Data (random 80/10/10 split) is built ONCE with a fixed split seed so every
config/seed sees identical windows; only model init / sampler / dropout vary.
This isolates the architecture change, which is exactly what an ablation should do.

The ONLY file this depends on is dual_stream_quality.py (kept UNTOUCHED); it is
imported as `D` and its data pipeline + SpatialGCN/TemporalGCN + train/eval loops
are reused verbatim.

------------------------------------------------------------------------------
USAGE
------------------------------------------------------------------------------
Defaults reproduce the reported run (3 seeds, 200 epochs, all 9 configs):
    python ablation_v2.py

Cut compute (single seed is fine for a first look; final numbers want >=3 seeds):
    python ablation_v2.py --seeds 42 --epochs 120
    python ablation_v2.py --seeds 42 1 7 --only full no_rom_stream single_task

------------------------------------------------------------------------------
TIME BUDGET  (read before launching)
------------------------------------------------------------------------------
One 200-epoch run ~= 2.5-3 h on your RTX 3090 (early stopping usually cuts it
shorter). So:
    9 configs x 1 seed  ~= 22-27 h
    9 configs x 3 seeds ~= 65-80 h
Practical plan: run all 9 configs with 1 seed first to see the ranking, then
re-run the 3-4 configs that matter with 3 seeds for the thesis table. Or drop
--epochs to ~120 (they nearly all early-stop before that anyway).
"""

import argparse
import copy
import csv
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import dual_stream_quality as D

# ─── Hardcoded settings (match the reported full-model command) ───────────────

TRAIN_NPZ   = 'data/uiprmd_segmented_gmm_train.npz'
TEST_NPZ    = 'data/uiprmd_segmented_gmm_test.npz'
BATCH_SIZE  = 16
LR          = 1e-4
HIDDEN_DIM  = 64
M           = 100
WINDOW_SIZE = 100
STRIDE      = 50
N_EXERCISES = 10
MIN_LEN     = 50

WARMUP_EPOCHS = 10
EARLY_STOP    = 40          # patience (epochs), matches main()
SPLIT_SEED    = 42          # fixed data split (matches --seed 42 in the full run)

DEFAULT_SEEDS  = [42, 1, 7]
DEFAULT_EPOCHS = 200

OUT_DIR   = 'results/ablation_v2'
TABLE_CSV = os.path.join(OUT_DIR, 'full_ablation_table.csv')

# stem, human label, and the ablation switches.
# Multi-task is controlled by lambda_ec / lambda_vc, NOT by removing heads.
CONFIGS = [
    dict(stem="full",          label="Full model",
         rom_guided=True,  rom_stream=True,  temporal=True,  attention=True,  bidirectional=True,  lambda_ec=0.5, lambda_vc=0.3),
    dict(stem="no_rom_init",   label="No ROM-guided adjacency init",
         rom_guided=False, rom_stream=True,  temporal=True,  attention=True,  bidirectional=True,  lambda_ec=0.5, lambda_vc=0.3),
    dict(stem="no_rom_stream", label="No ROM stream (GCN only)",
         rom_guided=True,  rom_stream=False, temporal=True,  attention=True,  bidirectional=True,  lambda_ec=0.5, lambda_vc=0.3),
    dict(stem="no_temporal",   label="No temporal GCN",
         rom_guided=True,  rom_stream=True,  temporal=False, attention=True,  bidirectional=True,  lambda_ec=0.5, lambda_vc=0.3),
    dict(stem="no_attention",  label="No attention (mean pool)",
         rom_guided=True,  rom_stream=True,  temporal=True,  attention=False, bidirectional=True,  lambda_ec=0.5, lambda_vc=0.3),
    dict(stem="uni_lstm",      label="Unidirectional LSTM",
         rom_guided=True,  rom_stream=True,  temporal=True,  attention=True,  bidirectional=False, lambda_ec=0.5, lambda_vc=0.3),
    dict(stem="single_task",   label="Single-task (quality only)",
         rom_guided=True,  rom_stream=True,  temporal=True,  attention=True,  bidirectional=True,  lambda_ec=0.0, lambda_vc=0.0),
    dict(stem="no_validity",   label="Multi-task w/o validity (Q+E)",
         rom_guided=True,  rom_stream=True,  temporal=True,  attention=True,  bidirectional=True,  lambda_ec=0.5, lambda_vc=0.0),
    dict(stem="no_exercise",   label="Multi-task w/o exercise (Q+V)",
         rom_guided=True,  rom_stream=True,  temporal=True,  attention=True,  bidirectional=True,  lambda_ec=0.0, lambda_vc=0.3),
]


# ─── Ablation-aware building blocks (reuse D.SpatialGCN / D.TemporalGCN) ───────


class AblationROMStream(nn.Module):
    """ROM stream with attention on/off and uni/bi-directional LSTM.
    attention=True + bidirectional=True == dual_stream_quality.ROMStream."""

    def __init__(self, rom_dim=12, proj_dim=64, lstm_hidden=64, n_layers=2,
                 use_attention=True, bidirectional=True):
        super().__init__()
        self.use_attention = use_attention
        self.bidirectional = bidirectional
        self.out_dim = lstm_hidden * 2 if bidirectional else lstm_hidden
        self.proj = nn.Sequential(
            nn.Linear(rom_dim, proj_dim), nn.ReLU(), nn.Dropout(0.1),
        )
        self.lstm = nn.LSTM(
            input_size=proj_dim, hidden_size=lstm_hidden, num_layers=n_layers,
            bidirectional=bidirectional, batch_first=True, dropout=0.2,
        )
        self.attn = nn.Linear(self.out_dim, 1, bias=False) if use_attention else None

    def forward(self, x, padding_mask=None):
        h = self.proj(x)
        h, _ = self.lstm(h)
        if self.use_attention:
            scores = self.attn(torch.tanh(h))
            if padding_mask is not None:
                scores = scores.masked_fill(padding_mask.unsqueeze(-1), float('-inf'))
            weights = torch.softmax(scores, dim=1)
            context = (weights * h).sum(dim=1)
            return context, weights.squeeze(-1)
        return h.mean(dim=1), None            # ablation: mean pool, no weights


class AblationNet(nn.Module):
    """Dual-Stream net with ablation knobs. Reuses D.SpatialGCN and D.TemporalGCN.
      rom_guided=False -> topology-only spatial adjacency (pass rom_guided_inits=None)
      rom_stream=False -> drop the whole ROM branch (GCN-only)
      temporal=False   -> replace the Temporal GCN with mean-pool over frames+joints
    """

    def __init__(self, hidden_dim, M, n_joints, n_exercises, A_topology,
                 rom_guided_inits, *, rom_stream=True, temporal=True,
                 attention=True, bidirectional=True):
        super().__init__()
        self.use_rom_stream = rom_stream
        self.use_temporal = temporal
        spatial_out = hidden_dim * 2               # 128
        temporal_in = n_joints * spatial_out

        self.spatial_gcn = D.SpatialGCN(n_joints, hidden_dim, n_exercises,
                                        A_topology, rom_guided_inits)

        if temporal:
            self.temporal_gcn = D.TemporalGCN(temporal_in, 128, M)
            gcn_out = 128
        else:
            # No temporal modelling: pool spatial features over frames + joints.
            self.spatial_proj = nn.Sequential(
                nn.Linear(spatial_out, 128), nn.ReLU(), nn.LayerNorm(128),
            )
            gcn_out = 128

        if rom_stream:
            self.rom_stream = AblationROMStream(
                12, 64, 64, 2, use_attention=attention, bidirectional=bidirectional)
            rom_out = self.rom_stream.out_dim
        else:
            self.rom_stream = None
            rom_out = 0

        # Inline fusion (identical to D.FusionLayer when both streams are present,
        # and works for the single-stream GCN-only case too).
        fusion_in = gcn_out + rom_out              # 256 / 192 / 128
        self.fusion_fc = nn.Linear(fusion_in, 128)
        self.fusion_drop = nn.Dropout(0.2)
        self.fusion_norm = nn.LayerNorm(128)

        self.exercise_head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.4), nn.Linear(64, n_exercises))
        self.validity_head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 2))
        self.quality_head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, joints, rom, exercise_id, padding_mask=None):
        B, M_, Jn, C = joints.shape
        sf = self.spatial_gcn(joints, exercise_id)          # (B, M, J, spatial_out)
        if self.use_temporal:
            sf = sf.reshape(B, M_, Jn * sf.shape[-1])
            tf = self.temporal_gcn(sf)                      # (B, 128)
        else:
            tf = self.spatial_proj(sf.mean(dim=(1, 2)))     # (B, 128)

        if self.use_rom_stream:
            rf, attn = self.rom_stream(rom, padding_mask)
            x = torch.cat([tf, rf], dim=-1)
        else:
            attn = None
            x = tf

        f = F.relu(self.fusion_fc(x))
        f = self.fusion_drop(f)
        f = self.fusion_norm(f)
        return (self.exercise_head(f), self.validity_head(f),
                self.quality_head(f).squeeze(-1), attn)


# ─── Shared data (built ONCE; fixed split so every config/seed sees same windows) ─


def build_shared_data():
    random.seed(SPLIT_SEED)
    np.random.seed(SPLIT_SEED)
    torch.manual_seed(SPLIT_SEED)

    print("Loading data...")
    tr = np.load(TRAIN_NPZ, allow_pickle=True)
    te = np.load(TEST_NPZ,  allow_pickle=True)

    def _labels(d, name):
        if 'quality_labels' in d:
            return d['quality_labels'].astype(np.int32)
        if 'quality_scores' in d:
            print(f"  [{name}] thresholding 'quality_scores' at 0.5")
            return (d['quality_scores'] >= 0.5).astype(np.int32)
        raise KeyError(f"[{name}] NPZ has neither 'quality_labels' nor 'quality_scores'")

    tr_poses3d, tr_rom, tr_ql = tr['poses_3d'], tr['rom_angles'], _labels(tr, 'train')
    tr_ex, tr_subj            = tr['exercise_ids'], tr['subject_ids']
    te_poses3d, te_rom, te_ql = te['poses_3d'], te['rom_angles'], _labels(te, 'test')
    te_ex, te_subj            = te['exercise_ids'], te['subject_ids']

    # Random 80/10/10 split (matches --split_mode random)
    all_poses3d = np.concatenate([tr_poses3d, te_poses3d], axis=0)
    all_rom     = np.concatenate([tr_rom,     te_rom],     axis=0)
    all_ql      = np.concatenate([tr_ql,      te_ql],      axis=0)
    all_ex      = np.concatenate([tr_ex,      te_ex],      axis=0)
    all_subj    = np.concatenate([tr_subj,    te_subj],    axis=0)
    _tr_sc = tr['quality_scores'].astype(np.float32) if 'quality_scores' in tr else np.zeros(len(tr_poses3d), np.float32)
    _te_sc = te['quality_scores'].astype(np.float32) if 'quality_scores' in te else np.zeros(len(te_poses3d), np.float32)
    all_gmm = np.concatenate([_tr_sc, _te_sc], axis=0)

    rng = np.random.default_rng(SPLIT_SEED)
    idx = rng.permutation(len(all_poses3d))
    n_train = int(0.8 * len(idx))
    n_val   = int(0.1 * len(idx))
    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train + n_val]
    test_idx  = idx[n_train + n_val:]

    tr_poses3d, tr_rom, tr_ql, tr_ex, tr_subj, tr_gmm = (
        all_poses3d[train_idx], all_rom[train_idx], all_ql[train_idx],
        all_ex[train_idx], all_subj[train_idx], all_gmm[train_idx])
    val_poses3d, val_rom, val_ql, val_ex, val_subj, val_gmm = (
        all_poses3d[val_idx], all_rom[val_idx], all_ql[val_idx],
        all_ex[val_idx], all_subj[val_idx], all_gmm[val_idx])
    te_poses3d, te_rom, te_ql, te_ex, te_subj, te_gmm = (
        all_poses3d[test_idx], all_rom[test_idx], all_ql[test_idx],
        all_ex[test_idx], all_subj[test_idx], all_gmm[test_idx])
    train_mask = np.ones(len(tr_poses3d), dtype=bool)
    print(f"Random split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
    print("Using quality_scores from NPZ directly")
    print(f"  Train range [{tr_gmm.min():.3f}, {tr_gmm.max():.3f}] | "
          f"Test range [{te_gmm.min():.3f}, {te_gmm.max():.3f}]")

    print("Computing ROM-guided adjacency init...")
    rom_guided_inits = D.compute_rom_guided_init(
        tr_poses3d[train_mask], tr_rom[train_mask],
        tr_ex[train_mask], tr_ql[train_mask], n_exercises=N_EXERCISES)

    A_topology = D.build_topology_adjacency(D.J)

    print("Computing ROM normalization stats...")
    dummy = D.build_windows(
        tr_poses3d[train_mask], tr_rom[train_mask], tr_gmm[train_mask],
        tr_ql[train_mask], tr_ex[train_mask], tr_subj[train_mask],
        WINDOW_SIZE, STRIDE, min_len=MIN_LEN,
        rom_mean=np.zeros((10, 12), np.float32), rom_std=np.ones((10, 12), np.float32),
        augment=False, generate_invalid=False)
    rom_mean, rom_std = D.compute_rom_stats(dummy)
    del dummy

    print("Building train windows...")
    train_windows = D.build_windows(
        tr_poses3d[train_mask], tr_rom[train_mask], tr_gmm[train_mask],
        tr_ql[train_mask], tr_ex[train_mask], tr_subj[train_mask],
        WINDOW_SIZE, STRIDE, min_len=MIN_LEN,
        rom_mean=rom_mean, rom_std=rom_std, augment=True, generate_invalid=True)
    print("Building val windows...")
    val_windows = D.build_windows(
        val_poses3d, val_rom, val_gmm, val_ql, val_ex, val_subj,
        WINDOW_SIZE, STRIDE, min_len=MIN_LEN,
        rom_mean=rom_mean, rom_std=rom_std, augment=False, generate_invalid=False)
    print("Building test windows...")
    test_windows = D.build_windows(
        te_poses3d, te_rom, te_gmm, te_ql, te_ex, te_subj,
        WINDOW_SIZE, STRIDE, min_len=MIN_LEN,
        rom_mean=rom_mean, rom_std=rom_std, augment=False, generate_invalid=False)
    print(f"  Train windows: {len(train_windows)} | "
          f"Val: {len(val_windows)} | Test: {len(test_windows)}")

    return {
        'train_windows': train_windows,
        'val_windows':   val_windows,
        'test_windows':  test_windows,
        'rom_guided_inits': rom_guided_inits,
        'A_topology': A_topology,
        'rom_mean': rom_mean,
        'rom_std':  rom_std,
    }


# ─── One (config, seed) training run -> TEST metrics ─────────────────────────


def run_single(cfg, seed, sd, device, epochs):
    label, stem = cfg['label'], cfg['stem']
    lambda_ec, lambda_vc = cfg['lambda_ec'], cfg['lambda_vc']

    print("\n" + "=" * 80)
    print(f"  CONFIG: {label}  |  seed={seed}")
    print(f"  rom_guided={cfg['rom_guided']} rom_stream={cfg['rom_stream']} "
          f"temporal={cfg['temporal']} attention={cfg['attention']} "
          f"bidirectional={cfg['bidirectional']} | lambda_ec={lambda_ec} lambda_vc={lambda_vc}")
    print("=" * 80 + "\n")

    # Reset RNG to this seed so init / sampler / dropout are deterministic per seed.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    save_model      = os.path.join(OUT_DIR, f'{stem}_seed{seed}.pt')
    save_csv        = os.path.join(OUT_DIR, f'{stem}_seed{seed}.csv')
    save_per_ex_csv = os.path.join(OUT_DIR, f'{stem}_seed{seed}_per_ex.csv')

    # Fresh copies (VC auto-flip mutates the window dicts in place).
    train_windows = copy.deepcopy(sd['train_windows'])
    val_windows   = copy.deepcopy(sd['val_windows'])
    test_windows  = copy.deepcopy(sd['test_windows'])

    train_ds = D.DualStreamDataset(train_windows)
    val_ds   = D.DualStreamDataset(val_windows)
    test_ds  = D.DualStreamDataset(test_windows)

    sampler      = D.make_weighted_sampler(train_windows)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                              num_workers=4, pin_memory=True)

    ex_arr    = np.array([w['exercise'] for w in train_windows])
    ex_counts = np.bincount(ex_arr, minlength=N_EXERCISES).astype(np.float32)
    ex_counts = np.where(ex_counts == 0, 1, ex_counts)
    cls_w_ex  = torch.tensor(1.0 / ex_counts[:N_EXERCISES])
    cls_w_ex  = cls_w_ex / cls_w_ex.sum() * N_EXERCISES

    rom_inits = sd['rom_guided_inits'] if cfg['rom_guided'] else None
    model = AblationNet(
        hidden_dim=HIDDEN_DIM, M=M, n_joints=D.J, n_exercises=N_EXERCISES,
        A_topology=sd['A_topology'], rom_guided_inits=rom_inits,
        rom_stream=cfg['rom_stream'], temporal=cfg['temporal'],
        attention=cfg['attention'], bidirectional=cfg['bidirectional'],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    ec_params    = list(model.exercise_head.parameters())
    ec_param_ids = set(id(p) for p in ec_params)
    other_params = [p for p in model.parameters() if id(p) not in ec_param_ids]
    optimizer = torch.optim.AdamW([
        {'params': other_params, 'weight_decay': 1e-4},
        {'params': ec_params,    'weight_decay': 1e-3, 'lr': LR * 0.3},
    ], lr=LR)

    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return epoch / max(WARMUP_EPOCHS, 1)
        progress = (epoch - WARMUP_EPOCHS) / max(epochs - WARMUP_EPOCHS, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_mad = float('inf')
    best_epoch = 0
    patience_cnt = 0
    vc_flipped = False

    print(f"Training up to {epochs} epochs (early stop patience={EARLY_STOP})...\n")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_m = D.train_one_epoch(
            model, train_loader, optimizer, device, lambda_ec, lambda_vc,
            cls_w_ex, desc=f"Ep[{epoch:3d}/{epochs}]", first_epoch=(epoch == 1))
        val_m = D.evaluate(model, val_loader, device, lambda_ec, lambda_vc)
        elapsed = time.time() - t0

        # VC auto-flip only makes sense when the VC head is actually being trained.
        if lambda_vc > 0 and epoch == 1 and not vc_flipped and tr_m['vc_acc'] < 0.45:
            print(f"  [VC] Train VC acc={tr_m['vc_acc']:.0%} < 45% - inverting VC labels")
            D._flip_vc_labels(train_windows)
            D._flip_vc_labels(val_windows)
            D._flip_vc_labels(test_windows)
            for layer in model.validity_head:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
            train_ds = D.DualStreamDataset(train_windows)
            val_ds   = D.DualStreamDataset(val_windows)
            test_ds  = D.DualStreamDataset(test_windows)
            sampler      = D.make_weighted_sampler(train_windows)
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                                      num_workers=4, pin_memory=True)
            val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                                      num_workers=4, pin_memory=True)
            test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                                      num_workers=4, pin_memory=True)
            vc_flipped = True

        scheduler.step()

        if val_m['mad'] < best_val_mad:
            best_val_mad = val_m['mad']
            best_epoch = epoch
            patience_cnt = 0
            torch.save({
                'epoch': epoch, 'state_dict': model.state_dict(),
                'val_mad': best_val_mad, 'config': label, 'seed': seed,
                'switches': {k: cfg[k] for k in
                             ('rom_guided', 'rom_stream', 'temporal',
                              'attention', 'bidirectional', 'lambda_ec', 'lambda_vc')},
                'rom_mean': sd['rom_mean'], 'rom_std': sd['rom_std'],
            }, save_model)
        else:
            patience_cnt += 1
            if patience_cnt >= EARLY_STOP:
                print(f"Early stopping at epoch {epoch} "
                      f"(best val MAD={best_val_mad:.4f} at ep {best_epoch})")
                break

        if epoch % 10 == 0 or epoch <= 3:
            cur_lr = optimizer.param_groups[0]['lr']
            print(f"Ep[{epoch:3d}] LR={cur_lr:.2e} | L={tr_m['loss']:.4f} "
                  f"QA={tr_m['qual_loss']:.4f} EC={tr_m['ec_acc']:.0%} "
                  f"VC={tr_m['vc_acc']:.0%} | val_MAD={val_m['mad']:.4f} "
                  f"val_EC={val_m['ec_acc']:.0%} | {elapsed:.1f}s")

    # ── TEST evaluation on the best-val checkpoint ───────────────────────────
    print(f"\nLoading best checkpoint (epoch {best_epoch}, val MAD={best_val_mad:.4f})...")
    ckpt = torch.load(save_model, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['state_dict'])

    test_m = D.evaluate(model, test_loader, device, lambda_ec, lambda_vc)
    D.print_results_table(test_m, save_csv, save_per_ex_csv)   # writes avg_mad/rmse/mape/ec/vc

    summ = _read_summary(save_csv) or {}
    return {
        'seed': seed,
        'best_epoch': best_epoch,
        'best_val_mad': best_val_mad,
        'test_mad':  summ.get('avg_mad'),
        'test_rmse': summ.get('avg_rmse'),
        'test_mape': summ.get('avg_mape'),
        'ec_acc':    summ.get('ec_acc'),
        'vc_acc':    summ.get('vc_acc'),
        'params':    n_params,
    }


def _read_summary(path):
    if not os.path.isfile(path):
        return None
    out = {}
    with open(path) as f:
        for row in csv.reader(f):
            if len(row) == 2 and row[0] != 'metric':
                try:
                    out[row[0]] = float(row[1])
                except ValueError:
                    pass
    return out


# ─── Aggregate across seeds -> final table ───────────────────────────────────


def _mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.std(vals))


def aggregate_and_report(configs, results, seeds):
    os.makedirs(OUT_DIR, exist_ok=True)

    agg = {}
    for cfg in configs:
        runs = results.get(cfg['stem'], [])
        mad_m,  mad_s  = _mean_std([r['test_mad']  for r in runs])
        rmse_m, rmse_s = _mean_std([r['test_rmse'] for r in runs])
        mape_m, mape_s = _mean_std([r['test_mape'] for r in runs])
        ec_m,   ec_s   = _mean_std([r['ec_acc']    for r in runs])
        vc_m,   vc_s   = _mean_std([r['vc_acc']    for r in runs])
        ep_m,   _      = _mean_std([r['best_epoch'] for r in runs])
        agg[cfg['stem']] = dict(
            label=cfg['label'], n=len(runs), params=runs[0]['params'] if runs else None,
            mad_m=mad_m, mad_s=mad_s, rmse_m=rmse_m, rmse_s=rmse_s,
            mape_m=mape_m, mape_s=mape_s, ec_m=ec_m, vc_m=vc_m, ep_m=ep_m,
            ec_trained=cfg['lambda_ec'] > 0, vc_trained=cfg['lambda_vc'] > 0)

    full_mad = agg.get('full', {}).get('mad_m')

    # CSV
    with open(TABLE_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['config', 'n_seeds', 'params', 'test_mad_mean', 'test_mad_std',
                    'test_rmse_mean', 'test_mape_mean', 'ec_acc_mean', 'vc_acc_mean',
                    'mean_best_epoch', 'delta_mad_vs_full'])
        for cfg in configs:
            a = agg[cfg['stem']]
            delta = (a['mad_m'] - full_mad) if (a['mad_m'] is not None and full_mad is not None) else None
            w.writerow([
                a['label'], a['n'], a['params'],
                _fmt(a['mad_m'], 4), _fmt(a['mad_s'], 4),
                _fmt(a['rmse_m'], 4), _fmt(a['mape_m'], 3),
                (_fmt(a['ec_m'], 2) if a['ec_trained'] else 'untrained'),
                (_fmt(a['vc_m'], 2) if a['vc_trained'] else 'untrained'),
                _fmt(a['ep_m'], 1), _fmt(delta, 4, sign=True),
            ])

    # Console
    SEP = "-" * 108
    print("\n\n=== Component-Wise Ablation (UI-PRMD TEST set, mean over "
          f"{len(seeds)} seed{'s' if len(seeds) != 1 else ''}: {seeds}) ===\n")
    print(SEP)
    print(f"{'Config':<34} {'Params':>9} {'Test MAD':>16} {'RMSE':>8} {'MAPE':>7} "
          f"{'EC%':>7} {'VC%':>7} {'dMAD':>9}")
    print(SEP)
    for cfg in configs:
        a = agg[cfg['stem']]
        mad = (f"{a['mad_m']:.4f}±{a['mad_s']:.4f}" if a['mad_m'] is not None else "   --   ")
        rmse = _cell(a['rmse_m'], '{:8.4f}')
        mape = _cell(a['mape_m'], '{:7.2f}')
        ec = "untr." if not a['ec_trained'] else _cell(a['ec_m'], '{:7.2f}')
        vc = "untr." if not a['vc_trained'] else _cell(a['vc_m'], '{:7.2f}')
        prm = f"{a['params']:,}" if a['params'] else "--"
        delta = (a['mad_m'] - full_mad) if (a['mad_m'] is not None and full_mad is not None) else None
        dcell = "  (ref)" if cfg['stem'] == 'full' else _cell(delta, '{:+9.4f}')
        print(f"{a['label']:<34} {prm:>9} {mad:>16} {rmse} {mape} {ec:>7} {vc:>7} {dcell}")
    print(SEP)
    print("\nNotes:")
    print("  * Test MAD is the thesis metric (val MAD is noisy on ~726 windows -> do not report it).")
    print("  * dMAD > 0 means removing that component HURTS (higher error). dMAD < 0 means it helps.")
    print("  * 'untr.' = that head's loss weight is 0, so its accuracy is untrained/meaningless.")
    print("  * If a removal shows dMAD <= its own std, treat it as within noise, not a real effect.")
    print(f"\nSaved: {TABLE_CSV}\n")


def _fmt(x, nd, sign=False):
    if x is None:
        return ''
    return (f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}")


def _cell(x, spec):
    return spec.format(x) if x is not None else "   --  "


# ─── Main ─────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="Multi-seed test-metric ablation study.")
    p.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS,
                   help='Model init seeds to average over (default: 42 1 7).')
    p.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS,
                   help='Max epochs per run (early stopping still applies).')
    p.add_argument('--only', type=str, nargs='+', default=None,
                   help="Run only these config stems (e.g. full no_rom_stream single_task).")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    os.makedirs(OUT_DIR, exist_ok=True)

    configs = CONFIGS
    if args.only:
        wanted = set(args.only)
        configs = [c for c in CONFIGS if c['stem'] in wanted]
        # keep 'full' so deltas have a reference
        if 'full' not in wanted and any(c['stem'] == 'full' for c in CONFIGS):
            configs = [c for c in CONFIGS if c['stem'] == 'full'] + configs
        if not configs:
            raise SystemExit(f"No matching configs in --only {args.only}")

    sd = build_shared_data()

    results = {}
    for cfg in configs:
        results[cfg['stem']] = []
        for seed in args.seeds:
            results[cfg['stem']].append(run_single(cfg, seed, sd, device, args.epochs))

    aggregate_and_report(configs, results, args.seeds)
    print("All ablation runs complete.")


if __name__ == '__main__':
    main()