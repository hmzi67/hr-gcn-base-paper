"""
6-Way Quality Assessment Comparison on UI-PRMD Dataset.

Trains the same LSTM architecture on six feature representations and compares
exercise quality (correct=1 / incorrect=0) classification:

  Path A — 3D Keypoints  : predicted body_3d (23 joints, 69-dim) from baseline
  Path B — Geometric ROM : ROM angles from predicted 3D joints  (12-dim)
  Path C — Learned ROM   : baseline angle head (λ=0, 12-dim, MAE≈4.59°)
  Path D — GCADA ROM     : novel angle head   (λ=0.1/0.05, 12-dim, MAE≈4.22°)
  Path E — +Deviation    : GCADA ROM (12) + per-exercise template deviations (12) = 24-dim
  Path F — Combined      : 3D keypoints (69) + GCADA ROM (12) + deviations (12) = 93-dim

Template deviations: for each exercise, correct-movement mean/std is computed
from training data and saved to rom_templates.json.  Deviation per frame is
|angle - template_mean| / template_std — a learned clinical normality signal.

Sequences are built with a sliding window (window=64, stride=16).

LSTM architecture (identical for all paths):
  Input → LSTM(64) → LSTM(32) → Dense(16, ReLU) → Dense(1, Sigmoid)
  Loss: BCE   Optimizer: Adam lr=1e-3   Epochs: 50

Training command:
    python quality_assessment_comparison.py \\
        --train \\
        --baseline_checkpoint checkpoint_rehab_baseline_v8/ckpt_best_rehab.pth.tar \\
        --novel_checkpoint    checkpoint_rehab_novel_v1/ckpt_best_rehab.pth.tar \\
        --cfg w32_adam_lr1e-3.yaml \\
        --data_train data/uiprmd_train.npz \\
        --data_test  data/uiprmd_test.npz \\
        --window_size 64 --stride 16 --min_window 32 \\
        --epochs 50 \\
        --save_dir results/quality_assessment/
"""
from __future__ import print_function, absolute_import, division

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

from lib.config import cfg
from common.graph_utils import adj_mx_from_skeleton
from utils.prepare_data_h3wb import Human3WBDataset
import models.graph_hrnet_multi_branch as ghrmb
import models.graph_resnet as GraphRes
import models.graph_hrnet as ghr
from models.graph_sh import GraphSH

# ── Constants ─────────────────────────────────────────────────────────────────

LSTM_H1  = 64
LSTM_H2  = 32
DENSE_H  = 16
EPOCHS   = 50
LR       = 1e-3
SEED     = 42

EXERCISE_NAMES = [
    'Deep Squat',     'Hurdle Step',    'Inline Lunge',  'Side Lunge',
    'Sit to Stand',   'Str Leg Raise',  'Sho Abduction', 'Sho Extension',
    'Sho Int-Ext Rot','Sho Scaption',
]

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='4-Way Quality Assessment Comparison (UI-PRMD)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('--train', action='store_true', default=True,
                   help='Run training + evaluation (default behaviour)')
    p.add_argument('--baseline_checkpoint',
                   default='checkpoint_rehab_baseline_v8/ckpt_best_rehab.pth.tar',
                   help='Baseline rehab checkpoint (λ_angle=0, 12-joint head)')
    p.add_argument('--novel_checkpoint',
                   default='checkpoint_rehab_novel_v1/ckpt_best_rehab.pth.tar',
                   help='GCADA (novel) rehab checkpoint (λ_angle=0.1, 12-joint head)')
    p.add_argument('--cfg',         default='w32_adam_lr1e-3.yaml')
    p.add_argument('--gcn',         default='dc_preagg')
    p.add_argument('--model',       default=1,   type=int)
    p.add_argument('--hid_dim',     default=64,  type=int)
    p.add_argument('--num_layers',  default=4,   type=int)
    p.add_argument('--data_train',  default='data/uiprmd_train.npz')
    p.add_argument('--data_test',   default='data/uiprmd_test.npz')
    # Sliding window
    p.add_argument('--window_size', default=64,  type=int,
                   help='Frames per window')
    p.add_argument('--stride',      default=16,  type=int,
                   help='Stride between consecutive windows')
    p.add_argument('--min_window',  default=32,  type=int,
                   help='Minimum frames at the start of a window to include it')
    # Training
    p.add_argument('--epochs',      default=EPOCHS, type=int)
    p.add_argument('--lr',          default=LR,     type=float)
    p.add_argument('--lstm_batch',  default=128,    type=int,
                   help='Trials per LSTM mini-batch')
    # Inference
    p.add_argument('--batch_size',  default=512,    type=int,
                   help='Frames per backbone inference batch')
    p.add_argument('--num_workers', default=4,      type=int)
    p.add_argument('--device',      default='cuda', choices=['cuda', 'cpu'])
    # Output
    p.add_argument('--save_dir',       default='results/', type=str)
    p.add_argument('--save_csv',
                   default='results/quality_assessment_improved.csv')
    p.add_argument('--save_per_ex',
                   default='results/quality_assessment_per_exercise_fixed.csv')
    p.add_argument('--save_templates',
                   default='results/rom_templates.json',
                   help='Path to save per-exercise ROM angle templates (JSON)')
    return p.parse_args()


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Backbone helpers ──────────────────────────────────────────────────────────

class _AngleHead(nn.Module):
    def __init__(self, in_features, hidden, n_joints):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, 64),          nn.ReLU(),
            nn.Linear(64, n_joints),
        )
    def forward(self, x):
        return self.net(x)


def _detect_head_dims(angle_state):
    hidden   = angle_state['net.0.weight'].shape[0]
    n_joints = angle_state['net.5.weight'].shape[0]
    in_feat  = angle_state['net.0.weight'].shape[1]
    return in_feat, hidden, n_joints


def _build_backbone(args, cfg_obj, adj, device):
    skeleton = Human3WBDataset('data/h3wb_train.npz', 'data/h3wb_test.npz').skeleton()
    if args.model == 1:
        return ghrmb.get_pose_net(cfg_obj, True, adj, None, args.gcn,
                                  skeleton.joints_group()).to(device)
    elif args.model == 2:
        return GraphRes.get_pose_net(True, adj, None, args.gcn, 50, True).to(device)
    elif args.model == 3:
        return ghr.get_pose_net(cfg_obj, True, adj, None, args.gcn,
                                skeleton.joints_group()).to(device)
    elif args.model == 4:
        return GraphSH(adj, args.hid_dim, skeleton.joints_group(),
                       num_layers=args.num_layers, p_dropout=None,
                       gcn_type=args.gcn).to(device)
    raise ValueError(f'Unknown model: {args.model}')


def infer_backbone(poses_2d: np.ndarray, ckpt_path: str,
                   args, cfg_obj, device) -> np.ndarray:
    """
    Run backbone only.  Returns body_3d (N, 23, 3).
    Loads the checkpoint once; does not load angle head.
    """
    print(f'  Loading: {ckpt_path}')
    ckpt    = torch.load(ckpt_path, map_location=device, weights_only=False)
    skeleton = Human3WBDataset('data/h3wb_train.npz', 'data/h3wb_test.npz').skeleton()
    adj      = adj_mx_from_skeleton(skeleton).to(device)
    model    = _build_backbone(args, cfg_obj, adj, device)
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model.eval()

    loader = DataLoader(TensorDataset(torch.from_numpy(poses_2d).float()),
                        batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=(args.device == 'cuda'))
    all_body = []
    with torch.no_grad():
        for (inp,) in loader:
            body_3d, _, _, _ = model(inp.to(device))
            all_body.append(body_3d.cpu().numpy())
    return np.concatenate(all_body, axis=0)          # (N, 23, 3)


def apply_angle_head(body_3d: np.ndarray, angle_state: dict,
                     device, batch_size: int = 4096) -> np.ndarray:
    """
    Apply a pre-loaded angle-head state dict to body_3d offline.
    body_3d : (N, 23, 3)
    returns : (N, n_joints)
    """
    in_feat, hidden, n_joints = _detect_head_dims(angle_state)
    head = _AngleHead(in_feat, hidden, n_joints).to(device)
    head.load_state_dict(angle_state)
    head.eval()

    flat = body_3d.reshape(len(body_3d), -1)          # (N, 69)
    x    = torch.from_numpy(flat).float()
    outs = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            outs.append(head(x[i:i+batch_size].to(device)).cpu().numpy())
    return np.concatenate(outs, axis=0)                # (N, n_joints)


def infer_backbone_and_head(poses_2d: np.ndarray, ckpt_path: str,
                             args, cfg_obj, device):
    """
    Run backbone + angle head in a single forward pass.
    Returns (body_3d, angles): (N,23,3), (N,n_joints).
    """
    print(f'  Loading: {ckpt_path}')
    ckpt        = torch.load(ckpt_path, map_location=device, weights_only=False)
    angle_state = ckpt.get('angle_head_state_dict')
    if angle_state is None:
        raise KeyError(f'angle_head_state_dict missing in {ckpt_path}')
    in_feat, hidden, n_joints = _detect_head_dims(angle_state)

    skeleton = Human3WBDataset('data/h3wb_train.npz', 'data/h3wb_test.npz').skeleton()
    adj      = adj_mx_from_skeleton(skeleton).to(device)
    model    = _build_backbone(args, cfg_obj, adj, device)
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model.eval()

    head = _AngleHead(in_feat, hidden, n_joints).to(device)
    head.load_state_dict(angle_state)
    head.eval()

    loader = DataLoader(TensorDataset(torch.from_numpy(poses_2d).float()),
                        batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=(args.device == 'cuda'))
    all_body, all_ang = [], []
    with torch.no_grad():
        for (inp,) in loader:
            body_3d = model(inp.to(device))[0]
            ang     = head(body_3d.reshape(body_3d.shape[0], -1))
            all_body.append(body_3d.cpu().numpy())
            all_ang.append(ang.cpu().numpy())
    return (np.concatenate(all_body, axis=0),
            np.concatenate(all_ang,  axis=0))


# ── Geometric ROM (same formulas as evaluate_rehab.py) ───────────────────────

def compute_rom_angles_from_coco(body_3d: np.ndarray) -> np.ndarray:
    """body_3d: (N, 23, 3) → (N, 12) degrees."""
    def _va(v1, v2):
        n1 = np.linalg.norm(v1, axis=-1, keepdims=True).clip(1e-8, None)
        n2 = np.linalg.norm(v2, axis=-1, keepdims=True).clip(1e-8, None)
        cos = ((v1/n1)*(v2/n2)).sum(-1).clip(-1+1e-7, 1-1e-7)
        return np.degrees(np.arccos(cos))

    N = body_3d.shape[0]
    head_  = body_3d[:, 0];  lsho  = body_3d[:, 5];  rsho  = body_3d[:, 6]
    lelb   = body_3d[:, 7];  relb  = body_3d[:, 8]
    l_hip  = body_3d[:, 11]; r_hip = body_3d[:, 12]
    lkne   = body_3d[:, 13]; rkne  = body_3d[:, 14]
    lank   = body_3d[:, 15]; rank  = body_3d[:, 16]
    ltoe   = body_3d[:, 17]; lhee  = body_3d[:, 19]
    rtoe   = body_3d[:, 20]; rhee  = body_3d[:, 22]

    up = np.zeros((N, 3), np.float32); up[:, 2] = 1.0
    c7  = (lsho + rsho) / 2.0
    mhp = (l_hip + r_hip) / 2.0
    sp  = c7 - mhp

    cerv  = _va(head_ - c7, up)
    trunk = _va(sp, up)

    la = lelb - lsho;  ra = relb - rsho
    lsf = 180.0 - _va(sp, la);  rsf = 180.0 - _va(sp, ra)

    lr_ax = rsho - lsho
    fn    = np.cross(lr_ax, up)
    fn   /= np.linalg.norm(fn, axis=-1, keepdims=True).clip(1e-8, None)
    lp    = la - (la * fn).sum(-1, keepdims=True) * fn
    rp    = ra - (ra * fn).sum(-1, keepdims=True) * fn
    lsabd = 180.0 - _va(up, lp);  rsabd = 180.0 - _va(up, rp)

    lf = lkne - l_hip;  rf = rkne - r_hip
    lha = 180.0 - _va(up, lf);  rha = 180.0 - _va(up, rf)

    lt = lank - lkne;  rt = rank - rkne
    lk = 180.0 - _va(-lf, lt);  rk = 180.0 - _va(-rf, rt)

    lft = ltoe - lhee;  rft = rtoe - rhee
    lak = _va(lt, lft);  rak = _va(rt, rft)

    return np.stack([cerv, trunk, lsf, rsf, lsabd, rsabd,
                     lha, rha, lk, rk, lak, rak], axis=1).astype(np.float32)


# ── Trial identification ──────────────────────────────────────────────────────

def identify_trials(subject_ids, exercise_ids, quality_scores):
    """
    Return list of trial dicts: start, end, label (int 0/1), exercise (int).
    A trial is a maximal consecutive run with the same (subject, exercise, quality).
    """
    trials = []
    N = len(subject_ids)
    prev_s, prev_e, prev_q = -1, -1, -999
    start = 0
    for i in range(N + 1):
        s = subject_ids[i]  if i < N else -2
        e = exercise_ids[i] if i < N else -2
        q = quality_scores[i] if i < N else -999
        if s != prev_s or e != prev_e or q != prev_q:
            if i > 0:
                trials.append(dict(start=start, end=i,
                                   label=int(prev_q > 0.5),
                                   exercise=int(prev_e)))
            prev_s, prev_e, prev_q = s, e, q
            start = i
    return trials


# ── Sliding-window feature builder ───────────────────────────────────────────

def build_windows(trials, feat_arrays: dict,
                  window_size: int, stride: int, min_window: int):
    """
    Apply sliding window within each trial.

    feat_arrays : {name: np.ndarray (N_total, F)}  — full-sequence feature arrays
    Returns:
        windows : {name: np.ndarray (W, window_size, F)}
        labels  : (W,) int
        exs     : (W,) int
    """
    accum   = {k: [] for k in feat_arrays}
    labels_out, exs_out = [], []

    for trial in trials:
        ts, te = trial['start'], trial['end']
        T      = te - ts
        if T < min_window:
            continue
        pos = 0
        while pos + min_window <= T:
            w_s = ts + pos
            w_e = w_s + window_size
            for k, arr in feat_arrays.items():
                seg = arr[w_s : min(w_e, te)]
                if len(seg) < window_size:
                    pad = np.repeat(seg[[-1]], window_size - len(seg), axis=0)
                    seg = np.concatenate([seg, pad], axis=0)
                accum[k].append(seg)
            labels_out.append(trial['label'])
            exs_out.append(trial['exercise'])
            pos += stride

    windows = {k: np.stack(v, 0).astype(np.float32) for k, v in accum.items()}
    return windows, np.array(labels_out, np.int64), np.array(exs_out, np.int64)


def normalise(tr_win, te_win):
    """Z-score per feature using train window statistics."""
    mu  = tr_win.mean(axis=(0, 1), keepdims=True)
    std = tr_win.std(axis=(0, 1),  keepdims=True).clip(1e-8, None)
    return (tr_win - mu) / std, (te_win - mu) / std


# ── Template deviation features ───────────────────────────────────────────────

_ROM_JOINT_NAMES = [
    'cervical_pitch', 'trunk_flex',
    'l_sho_flex', 'r_sho_flex', 'l_sho_abd', 'r_sho_abd',
    'l_hip', 'r_hip', 'l_knee', 'r_knee', 'l_ankle', 'r_ankle',
]


def compute_rom_templates(tr_wins_rom: np.ndarray,
                          tr_labs: np.ndarray,
                          tr_exs: np.ndarray) -> dict:
    """
    Compute per-exercise ROM angle templates from correct training windows only.

    tr_wins_rom : (N, T, 12)  raw (unnormalized) GCADA ROM angles
    tr_labs     : (N,)        binary labels (1 = correct)
    tr_exs      : (N,)        exercise index 0–9
    Returns     : {ex_id (int): {'mean': (12,), 'std': (12,)}}
    """
    templates = {}
    for ex in range(10):
        mask = (tr_exs == ex) & (tr_labs == 1)
        if mask.sum() == 0:
            mask = (tr_exs == ex)          # fall back to all if no correct
        if mask.sum() == 0:
            continue
        frames = tr_wins_rom[mask].reshape(-1, 12)   # (n_correct_wins × T, 12)
        templates[ex] = {
            'mean': frames.mean(axis=0).astype(np.float32),
            'std':  frames.std(axis=0).clip(1e-8, None).astype(np.float32),
        }
    return templates


def save_rom_templates(templates: dict, path: str) -> None:
    """Save templates as a human-readable JSON file."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    out = {}
    for ex, tmpl in templates.items():
        out[str(ex)] = {
            'exercise':  EXERCISE_NAMES[ex],
            'joints':    _ROM_JOINT_NAMES,
            'mean_deg':  [round(float(v), 4) for v in tmpl['mean']],
            'std_deg':   [round(float(v), 4) for v in tmpl['std']],
        }
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'  Saved ROM templates : {path}')


def compute_deviations(wins_rom: np.ndarray,
                       templates: dict,
                       exs: np.ndarray) -> np.ndarray:
    """
    Compute per-frame normalized deviation from the correct-movement template.

    wins_rom : (N, T, 12)  raw ROM angles (degrees)
    templates: {ex: {'mean': (12,), 'std': (12,)}}
    exs      : (N,)
    Returns  : (N, T, 12)  |rom - template_mean| / template_std  (≥ 0)
    """
    devs = np.zeros_like(wins_rom, dtype=np.float32)
    for i in range(len(wins_rom)):
        ex = int(exs[i])
        if ex in templates:
            devs[i] = np.abs(wins_rom[i] - templates[ex]['mean']) / templates[ex]['std']
    return devs


# ── LSTM classifier ───────────────────────────────────────────────────────────

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim: int,
                 h1: int = LSTM_H1, h2: int = LSTM_H2, dh: int = DENSE_H):
        super().__init__()
        self.lstm1 = nn.LSTM(input_dim, h1, batch_first=True)
        self.lstm2 = nn.LSTM(h1, h2,        batch_first=True)
        self.head  = nn.Sequential(
            nn.Linear(h2, dh), nn.ReLU(),
            nn.Linear(dh, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        return self.head(out[:, -1]).squeeze(-1)     # (B,)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ── Train & evaluate ──────────────────────────────────────────────────────────

def train_lstm(name: str, tr_win, tr_lab, te_win, te_lab, args, device):
    """
    Train LSTMClassifier and return metrics dict.
    tr_win : (n_train, window_size, F)
    """
    input_dim = tr_win.shape[-1]
    print(f'\n  [{name}]  input_dim={input_dim}  '
          f'train={len(tr_win):,}  test={len(te_win):,} windows')

    X_tr = torch.from_numpy(tr_win).float()
    y_tr = torch.from_numpy(tr_lab).float()
    X_te = torch.from_numpy(te_win).float()

    loader = DataLoader(TensorDataset(X_tr, y_tr),
                        batch_size=args.lstm_batch, shuffle=True, num_workers=0)
    set_seed(SEED)
    model   = LSTMClassifier(input_dim).to(device)
    opt     = torch.optim.Adam(model.parameters(), lr=args.lr)
    bce     = nn.BCELoss()

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = bce(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(xb)
        if epoch % 10 == 0:
            print(f'    epoch {epoch:3d}/{args.epochs}  '
                  f'loss={total_loss/len(X_tr):.4f}  '
                  f'elapsed={time.time()-t0:.1f}s')

    model.eval()
    with torch.no_grad():
        probs = model(X_te.to(device)).cpu().numpy()

    preds = (probs > 0.5).astype(int)
    gt    = te_lab.astype(int)
    acc   = accuracy_score(gt, preds) * 100.0
    f1    = f1_score(gt, preds, zero_division=0)
    try:
        auc = roc_auc_score(gt, probs)
    except ValueError:
        auc = float('nan')

    print(f'    Acc={acc:.2f}%  F1={f1:.4f}  AUC={auc:.4f}')
    return dict(acc=acc, f1=f1, auc=auc, n_params=model.n_params(),
                input_dim=input_dim, probs=probs)


def per_exercise_metrics(probs, labels, exs):
    """Return {ex: accuracy%} and {ex: n_windows} for each exercise."""
    preds = (probs > 0.5).astype(int)
    gt    = labels.astype(int)
    acc_d, cnt_d = {}, {}
    for ex in range(10):
        mask = exs == ex
        cnt_d[ex] = int(mask.sum())
        if mask.sum() == 0:
            acc_d[ex] = float('nan')
        else:
            acc_d[ex] = accuracy_score(gt[mask], preds[mask]) * 100.0
    return acc_d, cnt_d


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(SEED)

    if args.device == 'cuda' and not torch.cuda.is_available():
        print('CUDA unavailable, falling back to CPU')
        args.device = 'cpu'
    device = torch.device('cuda:0' if args.device == 'cuda' else 'cpu')
    print(f'Device: {device}')

    # Validate checkpoints
    for label, path in [('baseline', args.baseline_checkpoint),
                         ('novel',    args.novel_checkpoint)]:
        if not os.path.isfile(path):
            print(f'ERROR: {label} checkpoint not found: {path}')
            sys.exit(1)

    # Report checkpoint metadata
    for label, path in [('Baseline', args.baseline_checkpoint),
                         ('Novel',    args.novel_checkpoint)]:
        ck  = torch.load(path, map_location='cpu', weights_only=False)
        ast = ck.get('angle_head_state_dict', {})
        if ast:
            _, h, nj = _detect_head_dims(ast)
            a  = ck.get('args', {})
            la = a.get('lambda_angle', '?')    if isinstance(a, dict) else '?'
            lc = a.get('lambda_constraint', '?') if isinstance(a, dict) else '?'
            mae = ck.get('best_mae', float('nan'))
            print(f'  {label}: n_joints={nj}  hidden={h}  '
                  f'λ_angle={la}  λ_constraint={lc}  best_mae={mae:.4f}°')

    cfg.merge_from_file(args.cfg)

    # ── Load NPZ data ─────────────────────────────────────────────────────────
    print('\n==> Loading NPZ data...')
    tr = np.load(args.data_train, allow_pickle=True)
    te = np.load(args.data_test,  allow_pickle=True)

    tr_2d  = tr['poses_2d'];  tr_sid = tr['subject_ids']
    tr_eid = tr['exercise_ids']; tr_qs = tr['quality_scores']

    te_2d  = te['poses_2d'];  te_sid = te['subject_ids']
    te_eid = te['exercise_ids']; te_qs = te['quality_scores']

    print(f'  Train: {len(tr_2d):,} frames  '
          f'({(tr_qs>0.5).sum():,} correct / {(tr_qs<=0.5).sum():,} incorrect)')
    print(f'  Test : {len(te_2d):,} frames  '
          f'({(te_qs>0.5).sum():,} correct / {(te_qs<=0.5).sum():,} incorrect)')

    # ── Paths A + B + C: baseline backbone ───────────────────────────────────
    print('\n==> [Paths A, B, C] Baseline backbone inference...')
    print('  Train frames...')
    tr_body = infer_backbone(tr_2d, args.baseline_checkpoint, args, cfg, device)
    print('  Test frames...')
    te_body = infer_backbone(te_2d, args.baseline_checkpoint, args, cfg, device)

    # Path A: flatten 3D keypoints
    tr_kp = tr_body.reshape(len(tr_body), -1).astype(np.float32)  # (N, 69)
    te_kp = te_body.reshape(len(te_body), -1).astype(np.float32)

    # Path B: geometric ROM
    print('  Computing geometric ROM angles...')
    tr_geo = compute_rom_angles_from_coco(tr_body)   # (N, 12)
    te_geo = compute_rom_angles_from_coco(te_body)

    # Path C: baseline angle head (applied offline — fast MLP pass)
    print('  Applying baseline angle head (Path C)...')
    bl_ckpt       = torch.load(args.baseline_checkpoint, map_location=device,
                                weights_only=False)
    bl_head_state = bl_ckpt['angle_head_state_dict']
    tr_learned    = apply_angle_head(tr_body, bl_head_state, device)[:, :12]
    te_learned    = apply_angle_head(te_body, bl_head_state, device)[:, :12]
    del bl_ckpt

    # ── Path D: GCADA backbone + novel angle head ─────────────────────────────
    print('\n==> [Path D] Novel (GCADA) backbone + angle head inference...')
    print('  Train frames...')
    _, tr_gcada = infer_backbone_and_head(
        tr_2d, args.novel_checkpoint, args, cfg, device)
    print('  Test frames...')
    _, te_gcada = infer_backbone_and_head(
        te_2d, args.novel_checkpoint, args, cfg, device)
    tr_gcada = tr_gcada[:, :12].astype(np.float32)
    te_gcada = te_gcada[:, :12].astype(np.float32)

    # ── Identify trials ───────────────────────────────────────────────────────
    print('\n==> Identifying trials...')
    tr_trials = identify_trials(tr_sid, tr_eid, tr_qs)
    te_trials = identify_trials(te_sid, te_eid, te_qs)
    print(f'  Train: {len(tr_trials)} trials  '
          f'({sum(t["label"] for t in tr_trials)} correct)')
    print(f'  Test : {len(te_trials)} trials  '
          f'({sum(t["label"] for t in te_trials)} correct)')

    # ── Sliding-window feature extraction ────────────────────────────────────
    print(f'\n==> Building sliding windows '
          f'(size={args.window_size}, stride={args.stride}, '
          f'min={args.min_window})...')

    tr_feats = dict(A=tr_kp, B=tr_geo, C=tr_learned, D=tr_gcada)
    te_feats = dict(A=te_kp, B=te_geo, C=te_learned, D=te_gcada)

    tr_wins, tr_labs, tr_exs = build_windows(
        tr_trials, tr_feats, args.window_size, args.stride, args.min_window)
    te_wins, te_labs, te_exs = build_windows(
        te_trials, te_feats, args.window_size, args.stride, args.min_window)

    print(f'  Train windows: {len(tr_labs):,}  '
          f'({tr_labs.sum():,} correct / {(tr_labs==0).sum():,} incorrect)')
    print(f'  Test  windows: {len(te_labs):,}  '
          f'({te_labs.sum():,} correct / {(te_labs==0).sum():,} incorrect)')

    # ── Template deviation features (Paths E and F) ───────────────────────────
    # Templates built from raw (unnormalized) GCADA ROM — training correct only.
    print('\n==> Computing ROM templates from correct training windows (Path E/F)...')
    rom_templates = compute_rom_templates(tr_wins['D'], tr_labs, tr_exs)
    save_rom_templates(rom_templates, args.save_templates)

    # Print template summary (mean ROM per exercise)
    print(f'  {"Exercise":<18} {"CervPitch":>9} {"Trunk":>7} '
          f'{"LHip":>6} {"RHip":>6} {"LKnee":>7} {"RKnee":>7}')
    for ex in range(10):
        if ex not in rom_templates:
            continue
        m = rom_templates[ex]['mean']
        print(f'  {EXERCISE_NAMES[ex]:<18} {m[0]:>9.1f} {m[1]:>7.1f} '
              f'{m[6]:>6.1f} {m[7]:>6.1f} {m[8]:>7.1f} {m[9]:>7.1f}')

    print('\n  Computing deviations from templates...')
    tr_devs = compute_deviations(tr_wins['D'], rom_templates, tr_exs)
    te_devs = compute_deviations(te_wins['D'], rom_templates, te_exs)

    # Path E: GCADA ROM (12) + deviations (12) = 24-dim
    tr_wins['E'] = np.concatenate([tr_wins['D'], tr_devs], axis=-1)
    te_wins['E'] = np.concatenate([te_wins['D'], te_devs], axis=-1)

    # Path F: 3D keypoints (69) + GCADA ROM (12) + deviations (12) = 93-dim
    tr_wins['F'] = np.concatenate([tr_wins['A'], tr_wins['D'], tr_devs], axis=-1)
    te_wins['F'] = np.concatenate([te_wins['A'], te_wins['D'], te_devs], axis=-1)

    print(f'  Path E shape: {tr_wins["E"].shape}  (ROM 12 + dev 12 = 24-dim)')
    print(f'  Path F shape: {tr_wins["F"].shape}  (KP 69 + ROM 12 + dev 12 = 93-dim)')

    # Z-score normalise per path (must be done after E/F are built from raw D)
    for k in ('A', 'B', 'C', 'D', 'E', 'F'):
        tr_wins[k], te_wins[k] = normalise(tr_wins[k], te_wins[k])

    # ── Train LSTM classifiers ────────────────────────────────────────────────
    SEP = '='*66
    print('\n' + SEP)
    print('Training LSTM classifiers (%d epochs each)' % args.epochs)
    print(SEP)

    path_names = {
        'A': 'Path A — 3D Keypoints',
        'B': 'Path B — Geometric ROM',
        'C': 'Path C — Learned ROM (baseline, λ=0)',
        'D': 'Path D — GCADA ROM (novel, λ=0.1)',
        'E': 'Path E — +Deviation   (GCADA ROM 12 + dev 12 = 24-dim)',
        'F': 'Path F — Combined     (KP 69 + GCADA ROM 12 + dev 12 = 93-dim)',
    }
    results = {}
    for k in ('A', 'B', 'C', 'D', 'E', 'F'):
        results[k] = train_lstm(
            path_names[k],
            tr_wins[k], tr_labs,
            te_wins[k], te_labs,
            args, device)

    # ── Per-exercise accuracy ─────────────────────────────────────────────────
    per_ex_acc  = {}
    per_ex_cnt  = {}
    for k in ('A', 'B', 'C', 'D', 'E', 'F'):
        per_ex_acc[k], per_ex_cnt[k] = per_exercise_metrics(
            results[k]['probs'], te_labs, te_exs)
    n_test_ex = per_ex_cnt['A']    # same for all paths

    # ── Print summary table ───────────────────────────────────────────────────
    def _f4(v):  return f'{v:.4f}' if not np.isnan(v) else '  N/A'
    def _f2(v):  return f'{v:.2f}' if not np.isnan(v) else ' N/A'

    SEP2 = '-'*68
    print('\n\n' + SEP2)
    print(f'{"METRIC":<14} {"3D KP":>7} {"GEO":>7}'
          f' {"LEARNED":>8} {"GCADA":>7} {"+DEV":>7} {"COMBINED":>9}')
    print(SEP2)
    for metric, key in [('Accuracy(%)', 'acc'), ('F1 Score', 'f1'), ('AUC', 'auc')]:
        vals = [results[k][key] for k in ('A', 'B', 'C', 'D', 'E', 'F')]
        fn   = _f2 if key == 'acc' else _f4
        print(f'{metric:<14} {fn(vals[0]):>7} {fn(vals[1]):>7}'
              f' {fn(vals[2]):>8} {fn(vals[3]):>7} {fn(vals[4]):>7} {fn(vals[5]):>9}')
    print(f'{"Input dims":<14}'
          + ''.join(f' {results[k]["input_dim"]:>{"7" if k in "AB" else "8" if k == "C" else "7" if k in "DE" else "9"}}' for k in ('A','B','C','D','E','F')))
    print(SEP2)

    print('\nPER EXERCISE (Accuracy %):')
    SEP3 = '-'*82
    print(SEP3)
    print(f'{"Exercise":<18} {"3D KP":>7} {"GEO":>7}'
          f' {"LEARNED":>8} {"GCADA":>7} {"+DEV":>7} {"COMBINED":>9} {"N_test":>7}')
    print(SEP3)

    ex_accs = {k: [] for k in ('A', 'B', 'C', 'D', 'E', 'F')}
    for ex in range(10):
        va = per_ex_acc['A'][ex]; vb = per_ex_acc['B'][ex]
        vc = per_ex_acc['C'][ex]; vd = per_ex_acc['D'][ex]
        ve = per_ex_acc['E'][ex]; vf = per_ex_acc['F'][ex]
        n  = n_test_ex[ex]
        print(f'{EXERCISE_NAMES[ex]:<18}'
              f' {_f2(va):>7} {_f2(vb):>7}'
              f' {_f2(vc):>8} {_f2(vd):>7} {_f2(ve):>7} {_f2(vf):>9} {n:>7}')
        for k, v in zip(('A','B','C','D','E','F'), (va, vb, vc, vd, ve, vf)):
            if not np.isnan(v):
                ex_accs[k].append(v)

    avgs = {k: np.mean(ex_accs[k]) if ex_accs[k] else float('nan')
            for k in ('A','B','C','D','E','F')}
    tot  = sum(n_test_ex[ex] for ex in range(10))
    print(SEP3)
    print(f'{"AVERAGE":<18}'
          f' {_f2(avgs["A"]):>7} {_f2(avgs["B"]):>7}'
          f' {_f2(avgs["C"]):>8} {_f2(avgs["D"]):>7}'
          f' {_f2(avgs["E"]):>7} {_f2(avgs["F"]):>9} {tot:>7}')
    print(SEP3)

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.save_csv)    or '.', exist_ok=True)
    os.makedirs(os.path.dirname(args.save_per_ex) or '.', exist_ok=True)

    ALL_KEYS = ('A', 'B', 'C', 'D', 'E', 'F')

    # Summary CSV  →  results/quality_assessment_improved.csv
    with open(args.save_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Metric', '3D_Keypoints', 'Geometric_ROM',
                    'Learned_ROM_baseline', 'GCADA_ROM_novel',
                    'ROM_plus_Deviation', 'Combined_KP_ROM_Dev'])
        for metric, key in [('Accuracy_%', 'acc'), ('F1_Score', 'f1'), ('AUC', 'auc')]:
            w.writerow([metric] + [f'{results[k][key]:.4f}' for k in ALL_KEYS])
        w.writerow(['Input_dims'] + [results[k]['input_dim'] for k in ALL_KEYS])
        w.writerow(['Params']     + [results[k]['n_params']  for k in ALL_KEYS])
    print(f'\nSaved summary  : {args.save_csv}')

    # Per-exercise CSV
    def _c(v): return f'{v:.4f}' if not np.isnan(v) else 'N/A'
    with open(args.save_per_ex, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Exercise', '3D_KP', 'Geo_ROM',
                    'Learned_ROM', 'GCADA_ROM', 'ROM_Dev', 'Combined', 'N_test'])
        for ex in range(10):
            w.writerow([EXERCISE_NAMES[ex]] +
                       [_c(per_ex_acc[k][ex]) for k in ALL_KEYS] +
                       [n_test_ex[ex]])
        w.writerow(['AVERAGE'] +
                   [f'{avgs[k]:.4f}' for k in ALL_KEYS] + [tot])
    print(f'Saved per-ex   : {args.save_per_ex}')

    print('\n' + '='*68)
    print('Training command (retrain from scratch):')
    print('='*68)
    print(f"""
# Train Quality Assessment Module (6-way: A–F including template deviation)
python quality_assessment_comparison.py \\
  --train \\
  --baseline_checkpoint {args.baseline_checkpoint} \\
  --novel_checkpoint    {args.novel_checkpoint} \\
  --cfg {args.cfg} \\
  --data_train {args.data_train} \\
  --data_test  {args.data_test} \\
  --window_size {args.window_size} \\
  --stride {args.stride} \\
  --min_window {args.min_window} \\
  --epochs {args.epochs} \\
  --save_dir {args.save_dir} \\
  --save_csv results/quality_assessment_improved.csv \\
  --save_per_ex results/quality_assessment_per_exercise_fixed.csv \\
  --save_templates results/rom_templates.json
""")


if __name__ == '__main__':
    main()
