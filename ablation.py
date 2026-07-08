"""
Component-Wise Ablation Study for the Dual-Stream Quality model (UI-PRMD).

Self-contained driver. The ONLY command you ever need to run is:

    python ablation.py

It reuses the data pipeline and model building blocks from
dual_stream_quality.py (which is left completely untouched) and applies
each ablation purely at the Python level — no CLI flags, no argparse.

All paths and hyperparameters are hardcoded below as constants, pulled
to exactly match the original best checkpoint
(results/dual_stream_improve/dual_segmented_best.pt → MAD 0.0077). The
data pipeline (split, GMM scores, windows, augmentation) is built ONCE
and shared across every configuration so the only thing that varies is
the architecture / loss being ablated. The same seed is reset at the
start of every run, so differences are attributable to the change, not
to random variation.

Configurations:
    A. full_model          — sanity check (should reproduce ~0.0077)
    B. no_rom_guided_init   — topology-only spatial adjacency (no ROM blend)
    C. no_attention         — mean pooling instead of Bahdanau attention
    D. single_task          — quality head only (lambda_ec=0, lambda_vc=0)
    E. unidirectional_lstm  — unidirectional LSTM instead of BiLSTM
"""

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

# ─── Hardcoded constants (match dual_segmented_best.pt['args']) ───────────────

TRAIN_NPZ   = 'data/uiprmd_segmented_gmm_train.npz'
TEST_NPZ    = 'data/uiprmd_segmented_gmm_test.npz'
EPOCHS      = 200
BATCH_SIZE  = 16
LR          = 1e-4
HIDDEN_DIM  = 64
M           = 100
WINDOW_SIZE = 100
STRIDE      = 50
LAMBDA_EC   = 0.5
LAMBDA_VC   = 0.3
SEED        = 42
N_EXERCISES = 10
SPLIT_MODE  = 'random'      # original used a random 80/10/10 split
USE_NPZ_SCORES = True       # original used quality_scores from the NPZ directly

WARMUP_EPOCHS = 10          # matches main() in dual_stream_quality.py
EARLY_STOP    = 40          # patience (epochs) — matches main()
MIN_LEN       = 50

OUT_DIR        = 'results/ablation'
BEST_CKPT_PATH = 'results/dual_stream_improve/dual_segmented_best.pt'
TABLE_CSV      = os.path.join(OUT_DIR, 'full_ablation_table.csv')

# config label → (file stem, switches: rom_guided, attention, bidirectional, multi_task)
CONFIGS = [
    ("Full",                "full_model",   dict(use_rom_guided=True,  use_attention=True,  bidirectional=True,  single_task=False)),
    ("No ROM-guided init",  "no_rom_init",  dict(use_rom_guided=False, use_attention=True,  bidirectional=True,  single_task=False)),
    ("No attention",        "no_attention", dict(use_rom_guided=True,  use_attention=False, bidirectional=True,  single_task=False)),
    ("Single-task",         "single_task",  dict(use_rom_guided=True,  use_attention=True,  bidirectional=True,  single_task=True)),
    ("Unidirectional LSTM", "uni_lstm",     dict(use_rom_guided=True,  use_attention=True,  bidirectional=False, single_task=False)),
]


# ─── Ablation-aware model variants (reuse base blocks from D) ─────────────────


class AblationROMStream(nn.Module):
    """ROMStream variant supporting attention on/off and uni/bi-directional LSTM.

    With use_attention=True and bidirectional=True it is functionally identical
    to dual_stream_quality.ROMStream.
    """

    def __init__(self, rom_dim=12, proj_dim=64, lstm_hidden=64, n_layers=2,
                 use_attention=True, bidirectional=True):
        super().__init__()
        self.use_attention = use_attention
        self.bidirectional = bidirectional
        self.out_dim = lstm_hidden * 2 if bidirectional else lstm_hidden
        self.proj = nn.Sequential(
            nn.Linear(rom_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.lstm = nn.LSTM(
            input_size=proj_dim,
            hidden_size=lstm_hidden,
            num_layers=n_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=0.2,
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
        # Ablation: mean pool over time, no attention weights
        return h.mean(dim=1), None


class AblationNet(nn.Module):
    """DualStreamQualityNet with ablation knobs. Reuses SpatialGCN / TemporalGCN
    / FusionLayer from dual_stream_quality unchanged. Passing rom_guided_inits=None
    yields the topology-only spatial init (the --no_rom_guided_init ablation)."""

    def __init__(self, hidden_dim, M, n_joints, n_exercises, A_topology,
                 rom_guided_inits, use_attention=True, bidirectional=True):
        super().__init__()
        spatial_out = hidden_dim * 2
        temporal_in = n_joints * spatial_out

        self.spatial_gcn  = D.SpatialGCN(n_joints, hidden_dim, n_exercises,
                                          A_topology, rom_guided_inits)
        self.temporal_gcn = D.TemporalGCN(temporal_in, 128, M)
        self.rom_stream   = AblationROMStream(12, 64, 64, 2,
                                              use_attention=use_attention,
                                              bidirectional=bidirectional)
        fusion_in = 128 + self.rom_stream.out_dim   # 256 (bi) or 192 (uni)
        self.fusion       = D.FusionLayer(fusion_in, 128)

        self.exercise_head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.4),
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
        B, M_, Jn, C = joints.shape
        sf = self.spatial_gcn(joints, exercise_id)
        sf = sf.reshape(B, M_, Jn * sf.shape[-1])
        tf = self.temporal_gcn(sf)
        rf, attn = self.rom_stream(rom, padding_mask)
        fused = self.fusion(tf, rf)
        return (self.exercise_head(fused),
                self.validity_head(fused),
                self.quality_head(fused).squeeze(-1),
                attn)


# ─── Shared data pipeline (built once, reused by every config) ────────────────


def build_shared_data(device):
    """Replicates the data half of dual_stream_quality.main() for the exact
    SPLIT_MODE='random' + USE_NPZ_SCORES=True configuration of the best
    checkpoint. Returns a dict with windows + adjacency + stats."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

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

    # ── Random 80/10/10 split (SPLIT_MODE == 'random') ───────────────────────
    all_poses3d = np.concatenate([tr_poses3d, te_poses3d], axis=0)
    all_rom     = np.concatenate([tr_rom,     te_rom],     axis=0)
    all_ql      = np.concatenate([tr_ql,      te_ql],      axis=0)
    all_ex      = np.concatenate([tr_ex,      te_ex],      axis=0)
    all_subj    = np.concatenate([tr_subj,    te_subj],    axis=0)
    _tr_sc = tr['quality_scores'].astype(np.float32) if 'quality_scores' in tr else np.zeros(len(tr_poses3d), np.float32)
    _te_sc = te['quality_scores'].astype(np.float32) if 'quality_scores' in te else np.zeros(len(te_poses3d), np.float32)
    all_gmm = np.concatenate([_tr_sc, _te_sc], axis=0)

    rng = np.random.default_rng(SEED)
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

    # USE_NPZ_SCORES=True → tr_gmm/val_gmm/te_gmm already taken from quality_scores.
    print('Using quality_scores from NPZ directly')
    print(f'Train: range [{tr_gmm.min():.3f}, {tr_gmm.max():.3f}]')
    print(f'Test:  range [{te_gmm.min():.3f}, {te_gmm.max():.3f}]')

    # ── ROM-guided adjacency (train only) ────────────────────────────────────
    print("Computing ROM-guided adjacency init...")
    rom_guided_inits = D.compute_rom_guided_init(
        tr_poses3d[train_mask], tr_rom[train_mask],
        tr_ex[train_mask], tr_ql[train_mask], n_exercises=N_EXERCISES)

    A_topology = D.build_topology_adjacency(D.J)

    # ── ROM normalization stats from train windows ───────────────────────────
    print("Computing ROM normalization stats...")
    dummy = D.build_windows(
        tr_poses3d[train_mask], tr_rom[train_mask], tr_gmm[train_mask],
        tr_ql[train_mask], tr_ex[train_mask], tr_subj[train_mask],
        WINDOW_SIZE, STRIDE, min_len=MIN_LEN,
        rom_mean=np.zeros((10, 12), np.float32), rom_std=np.ones((10, 12), np.float32),
        augment=False, generate_invalid=False)
    rom_mean, rom_std = D.compute_rom_stats(dummy)
    del dummy

    # ── Build windows ONCE (shared across all configs) ───────────────────────
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
    print(f"  Train windows: {len(train_windows)}")
    print(f"  Val   windows: {len(val_windows)}")
    print(f"  Test  windows: {len(test_windows)}")

    return {
        'train_windows': train_windows,
        'val_windows':   val_windows,
        'test_windows':  test_windows,
        'rom_guided_inits': rom_guided_inits,
        'A_topology': A_topology,
        'rom_mean': rom_mean,
        'rom_std':  rom_std,
    }


# ─── Per-config training run ──────────────────────────────────────────────────


def run_config(label, stem, sd, device, *,
               use_rom_guided, use_attention, bidirectional, single_task):
    print("\n" + "=" * 78)
    print(f"  CONFIG: {label}  (rom_guided={use_rom_guided}, attention={use_attention}, "
          f"bidirectional={bidirectional}, multi_task={not single_task})")
    print("=" * 78 + "\n")

    # Reset seed so every config sees identical init/sampler/dropout randomness.
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    lambda_ec = 0.0 if single_task else LAMBDA_EC
    lambda_vc = 0.0 if single_task else LAMBDA_VC
    if single_task:
        print("[ablation] single_task: lambda_ec=0.0, lambda_vc=0.0 "
              "(EC/VC heads present but receive no gradient -> untrained)")

    save_model      = os.path.join(OUT_DIR, f'{stem}.pt')
    save_csv        = os.path.join(OUT_DIR, f'{stem}.csv')
    save_per_ex_csv = os.path.join(OUT_DIR, f'{stem}_per_ex.csv')

    # Fresh, isolated copies of the shared windows (VC auto-flip mutates them).
    train_windows = copy.deepcopy(sd['train_windows'])
    val_windows   = copy.deepcopy(sd['val_windows'])
    test_windows  = copy.deepcopy(sd['test_windows'])

    train_ds = D.DualStreamDataset(train_windows)
    val_ds   = D.DualStreamDataset(val_windows)
    test_ds  = D.DualStreamDataset(test_windows)

    sampler      = D.make_weighted_sampler(train_windows)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              sampler=sampler, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE * 2,
                              shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE * 2,
                              shuffle=False, num_workers=4, pin_memory=True)

    ex_arr    = np.array([w['exercise'] for w in train_windows])
    ex_counts = np.bincount(ex_arr, minlength=N_EXERCISES).astype(np.float32)
    ex_counts = np.where(ex_counts == 0, 1, ex_counts)
    cls_w_ex  = torch.tensor(1.0 / ex_counts[:N_EXERCISES])
    cls_w_ex  = cls_w_ex / cls_w_ex.sum() * N_EXERCISES

    # Ablation: no_rom_guided_init -> pass rom_guided_inits=None (topology-only).
    rom_inits = sd['rom_guided_inits'] if use_rom_guided else None
    model = AblationNet(
        hidden_dim=HIDDEN_DIM, M=M, n_joints=D.J, n_exercises=N_EXERCISES,
        A_topology=sd['A_topology'], rom_guided_inits=rom_inits,
        use_attention=use_attention, bidirectional=bidirectional,
    ).to(device)

    def count_params(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"Model parameters (total): {count_params(model):,}")

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
        progress = (epoch - WARMUP_EPOCHS) / max(EPOCHS - WARMUP_EPOCHS, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_mad = float('inf')
    best_epoch   = 0
    patience_cnt = 0
    vc_flipped   = False

    print(f"\nTraining for {EPOCHS} epochs (early stop patience={EARLY_STOP})...\n")
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_m = D.train_one_epoch(
            model, train_loader, optimizer, device, lambda_ec, lambda_vc,
            cls_w_ex, desc=f"Ep[{epoch:3d}/{EPOCHS}]", first_epoch=(epoch == 1))
        val_m = D.evaluate(model, val_loader, device, lambda_ec, lambda_vc)
        elapsed = time.time() - t0

        # VC label auto-flip at epoch 1 (matches main()).
        if epoch == 1 and not vc_flipped and tr_m['vc_acc'] < 0.45:
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
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                                      sampler=sampler, num_workers=4, pin_memory=True)
            val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE * 2,
                                      shuffle=False, num_workers=4, pin_memory=True)
            test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE * 2,
                                      shuffle=False, num_workers=4, pin_memory=True)
            vc_flipped = True

        scheduler.step()

        if val_m['mad'] < best_val_mad:
            best_val_mad = val_m['mad']
            best_epoch   = epoch
            patience_cnt = 0
            torch.save({
                'epoch':      epoch,
                'state_dict': model.state_dict(),
                'optimizer':  optimizer.state_dict(),
                'val_mad':    best_val_mad,
                'config':     label,
                'switches':   dict(use_rom_guided=use_rom_guided,
                                   use_attention=use_attention,
                                   bidirectional=bidirectional,
                                   single_task=single_task),
                'rom_mean':   sd['rom_mean'],
                'rom_std':    sd['rom_std'],
            }, save_model)
        else:
            patience_cnt += 1
            if patience_cnt >= EARLY_STOP:
                print(f"Early stopping at epoch {epoch} "
                      f"(best val MAD={best_val_mad:.4f} at ep {best_epoch})")
                break

        if epoch % 10 == 0 or epoch <= 3:
            cur_lr = optimizer.param_groups[0]['lr']
            print(f"Ep[{epoch:3d}] LR={cur_lr:.2e} | L={tr_m['loss']:.4f} | "
                  f"QA={tr_m['qual_loss']:.4f} EC={tr_m['ec_acc']:.0%} "
                  f"VC={tr_m['vc_acc']:.0%} | val_MAD={val_m['mad']:.4f} "
                  f"val_EC={val_m['ec_acc']:.0%} | {elapsed:.1f}s")

    print(f"\nLoading best checkpoint (epoch {best_epoch}, val MAD={best_val_mad:.4f})...")
    ckpt = torch.load(save_model, map_location=device)
    model.load_state_dict(ckpt['state_dict'])

    test_m = D.evaluate(model, test_loader, device, lambda_ec, lambda_vc)
    # Writes save_csv (summary) + save_per_ex_csv (per-exercise breakdown).
    D.print_results_table(test_m, save_csv, save_per_ex_csv)
    print(f"  Checkpoint: {save_model}")
    return {'best_val_mad': best_val_mad, 'best_epoch': best_epoch, 'single_task': single_task}


# ─── Final ablation table ─────────────────────────────────────────────────────


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


def build_and_print_table(run_info):
    ckpt_val_mad = None
    if os.path.isfile(BEST_CKPT_PATH):
        try:
            ck = torch.load(BEST_CKPT_PATH, map_location='cpu', weights_only=False)
            ckpt_val_mad = ck.get('val_mad')
        except Exception as exc:
            print(f"[warn] could not read reference checkpoint: {exc}")

    rows = []
    for label, stem, sw in CONFIGS:
        summ = _read_summary(os.path.join(OUT_DIR, f'{stem}.csv'))
        single = sw['single_task']
        rows.append({
            'config': label,
            'rom_guided_init': sw['use_rom_guided'],
            'attention': sw['use_attention'],
            'bidirectional': sw['bidirectional'],
            'multi_task': not single,
            'single_task': single,
            'test_mad':  None if summ is None else summ.get('avg_mad'),
            'test_rmse': None if summ is None else summ.get('avg_rmse'),
            'test_mape': None if summ is None else summ.get('avg_mape'),
            'ec_acc':    None if summ is None else summ.get('ec_acc'),
            'vc_acc':    None if summ is None else summ.get('vc_acc'),
            'missing':   summ is None,
        })

    full_mad = rows[0]['test_mad']
    for r in rows:
        r['delta'] = (r['test_mad'] - full_mad) if (r['test_mad'] is not None and full_mad is not None) else None

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(TABLE_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['config', 'rom_guided_init', 'attention', 'bidirectional',
                    'multi_task', 'test_mad', 'test_rmse', 'test_mape',
                    'ec_acc', 'vc_acc', 'delta_mad_vs_full'])
        for r in rows:
            w.writerow([
                r['config'], r['rom_guided_init'], r['attention'],
                r['bidirectional'], r['multi_task'],
                '' if r['test_mad']  is None else f"{r['test_mad']:.4f}",
                '' if r['test_rmse'] is None else f"{r['test_rmse']:.4f}",
                '' if r['test_mape'] is None else f"{r['test_mape']:.4f}",
                '' if r['ec_acc']    is None else f"{r['ec_acc']:.2f}",
                '' if r['vc_acc']    is None else f"{r['vc_acc']:.2f}",
                '' if r['delta']     is None else f"{r['delta']:+.4f}",
            ])

    def cell(x, spec):
        return spec.format(x) if x is not None else "   --  "

    SEP = "-" * 104
    print("\n\n=== Component-Wise Ablation Study (UI-PRMD Test Set) ===\n")
    if ckpt_val_mad is not None:
        print(f"Reference: original full-model checkpoint val MAD = {ckpt_val_mad:.4f}\n")
    print(SEP)
    print(f"{'Config':<22} {'ROM':>4} {'Attn':>5} {'BiDir':>6} {'Multi':>6} "
          f"{'MAD':>8} {'RMSE':>8} {'MAPE':>8} {'EC%':>7} {'VC%':>7} {'dMAD':>9}")
    print(SEP)
    fl = lambda b: "Y" if b else "-"
    for r in rows:
        # EC/VC are untrained under single_task — flag rather than imply meaning.
        ec_disp = "untr." if r['single_task'] else cell(r['ec_acc'], '{:7.2f}')
        vc_disp = "untr." if r['single_task'] else cell(r['vc_acc'], '{:7.2f}')
        tail = "   [MISSING run]" if r['missing'] else ""
        print(f"{r['config']:<22} {fl(r['rom_guided_init']):>4} {fl(r['attention']):>5} "
              f"{fl(r['bidirectional']):>6} {fl(r['multi_task']):>6} "
              f"{cell(r['test_mad'], '{:8.4f}')} {cell(r['test_rmse'], '{:8.4f}')} "
              f"{cell(r['test_mape'], '{:8.2f}')} {ec_disp:>7} {vc_disp:>7} "
              f"{cell(r['delta'], '{:+9.4f}')}{tail}")
    print(SEP)
    print("\nNote: under Single-task the EC/VC heads exist but receive no gradient "
          "(lambda_ec=lambda_vc=0), so their accuracy is untrained and not meaningful.")
    print(f"\nSaved table: {TABLE_CSV}\n")
    return rows


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    os.makedirs(OUT_DIR, exist_ok=True)

    sd = build_shared_data(device)

    run_info = {}
    for label, stem, switches in CONFIGS:
        run_info[label] = run_config(label, stem, sd, device, **switches)

    build_and_print_table(run_info)
    print("All ablation runs complete.")


if __name__ == '__main__':
    main()
