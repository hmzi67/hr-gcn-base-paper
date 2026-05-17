#!/usr/bin/env python3
"""
Rep-level quality assessment for UI-PRMD exercises.

Input : one full exercise repetition — sequence of 12 ROM angles (T, 12)
Output: binary quality score per sequence (0=incorrect, 1=correct)
Model : stacked LSTM → Dense → Sigmoid

If the quality NPZ files don't exist, they are built automatically from
  data/UI-PRMD/raw/Segmented Movements/Vicon/Positions/
  data/UI-PRMD/raw/Incorrect Segmented Movements/Vicon/Positions/

Usage:
    python quality_assessment_rep_level.py \\
        --data_train data/uiprmd_quality_train.npz \\
        --data_test  data/uiprmd_quality_test.npz  \\
        --epochs 50  \\
        --max_seq_len 300 \\
        --hidden_dim 128 \\
        --save_model results/qa_rep_level_best.pt \\
        --save_csv   results/qa_rep_level_results.csv
"""

import argparse
import glob
import os
import re
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset

# ─── scikit-learn is optional for AUC ────────────────────────────────────────
try:
    from sklearn.metrics import roc_auc_score
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

# ─────────────────────────────────────────────────────────────────────────────
# ROM angle computation (adapted from utils/prepare_data_uiprmd.py)
# Works on Vicon 39-joint positions (T, 39, 3) in mm, Z-up.
# ─────────────────────────────────────────────────────────────────────────────

def _va(v1, v2):
    """Batch angle (degrees) between (T,3) vectors."""
    n1 = np.linalg.norm(v1, axis=-1, keepdims=True).clip(1e-8, None)
    n2 = np.linalg.norm(v2, axis=-1, keepdims=True).clip(1e-8, None)
    cos = ((v1 / n1) * (v2 / n2)).sum(-1).clip(-1.0 + 1e-7, 1.0 - 1e-7)
    return np.degrees(np.arccos(cos))


def compute_rom_angles(pos):
    """
    pos: (T, 39, 3) Vicon positions in mm, Z-up
    returns: (T, 12) ROM angles in degrees
    """
    T = pos.shape[0]
    lasi = pos[:, 23]; rasi = pos[:, 24]
    lpsi = pos[:, 25]; rpsi = pos[:, 26]
    lkne = pos[:, 28]; rkne = pos[:, 34]
    lank = pos[:, 29]; rank = pos[:, 35]
    lsho = pos[:,  9]; rsho = pos[:,  8]
    lelb = pos[:, 11]; relb = pos[:, 17]
    head = pos[:,  0]; c7   = pos[:,  4]
    ltoe = pos[:, 30]; rtoe = pos[:, 36]
    lhee = pos[:, 32]; rhee = pos[:, 38]

    l_hip_ctr = (lasi + lpsi) / 2.0
    r_hip_ctr = (rasi + rpsi) / 2.0
    mid_hip   = (l_hip_ctr + r_hip_ctr) / 2.0
    mid_sho   = (lsho + rsho) / 2.0
    spine     = mid_sho - mid_hip
    up        = np.zeros((T, 3), dtype=np.float32)
    up[:, 2]  = 1.0

    cerv_pitch  = _va(head - c7, up)
    trunk_flex  = _va(spine, up)

    l_upper_arm = lelb - lsho
    r_upper_arm = relb - rsho
    l_sho_flex  = 180.0 - _va(spine, l_upper_arm)
    r_sho_flex  = 180.0 - _va(spine, r_upper_arm)

    lr_axis        = rsho - lsho
    frontal_normal = np.cross(lr_axis, up)
    fn_norm        = np.linalg.norm(frontal_normal, axis=-1, keepdims=True).clip(1e-8, None)
    frontal_normal = frontal_normal / fn_norm
    l_proj = l_upper_arm - (l_upper_arm * frontal_normal).sum(-1, keepdims=True) * frontal_normal
    r_proj = r_upper_arm - (r_upper_arm * frontal_normal).sum(-1, keepdims=True) * frontal_normal
    l_sho_abd = 180.0 - _va(up, l_proj)
    r_sho_abd = 180.0 - _va(up, r_proj)

    l_femur = lkne - l_hip_ctr
    r_femur = rkne - r_hip_ctr
    l_hip   = 180.0 - _va(up, l_femur)
    r_hip   = 180.0 - _va(up, r_femur)

    l_tibia = lank - lkne
    r_tibia = rank - rkne
    l_knee  = 180.0 - _va(-l_femur, l_tibia)
    r_knee  = 180.0 - _va(-r_femur, r_tibia)

    l_foot  = ltoe - lhee
    r_foot  = rtoe - rhee
    l_ankle = _va(l_tibia, l_foot)
    r_ankle = _va(r_tibia, r_foot)

    return np.stack([
        cerv_pitch, trunk_flex,
        l_sho_flex, r_sho_flex,
        l_sho_abd,  r_sho_abd,
        l_hip,      r_hip,
        l_knee,     r_knee,
        l_ankle,    r_ankle,
    ], axis=1).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Quality NPZ builder
# ─────────────────────────────────────────────────────────────────────────────

_TRAIN_SUBJECTS = set(range(1, 9))   # s01–s08
_TEST_SUBJECTS  = {9, 10}            # s09–s10


def _parse_seg_filename(basename):
    """Return (movement, subject, rep) as 1-indexed ints, or None."""
    m = re.match(r'm(\d+)_s(\d+)_e(\d+)_positions', basename)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def _load_seg_file(fpath):
    """Load one segmented Vicon positions file → (T, 39, 3) numpy array."""
    raw = np.loadtxt(fpath, delimiter=',')   # (T, 117)
    T = raw.shape[0]
    return raw.reshape(T, 39, 3).astype(np.float32)


def build_quality_npz(raw_dir, train_path, test_path, max_seq_len):
    """
    Build rep-level quality NPZ files from raw segmented Vicon position files.
    Saves two NPZ files with keys:
        sequences     : (N, max_seq_len, 12)  padded ROM angles
        lengths       : (N,)                  true sequence lengths
        quality_labels: (N,)                  0=incorrect, 1=correct
        exercise_ids  : (N,)                  0-indexed exercise
        subject_ids   : (N,)                  0-indexed subject
    """
    correct_dir   = os.path.join(raw_dir, 'Segmented Movements',
                                 'Vicon', 'Positions')
    incorrect_dir = os.path.join(raw_dir, 'Incorrect Segmented Movements',
                                 'Vicon', 'Positions')

    if not os.path.isdir(correct_dir):
        raise FileNotFoundError(f'Correct segmented dir not found: {correct_dir}')
    if not os.path.isdir(incorrect_dir):
        raise FileNotFoundError(f'Incorrect segmented dir not found: {incorrect_dir}')

    entries = []  # (fpath, label, movement_1idx, subject_1idx, rep_1idx)

    for fp in sorted(glob.glob(os.path.join(correct_dir, 'm??_s??_e??_positions.txt'))):
        parsed = _parse_seg_filename(os.path.basename(fp))
        if parsed:
            m, s, e = parsed
            entries.append((fp, 1, m, s, e))

    for fp in sorted(glob.glob(os.path.join(incorrect_dir, 'm??_s??_e??_positions_inc.txt'))):
        parsed = _parse_seg_filename(os.path.basename(fp))
        if parsed:
            m, s, e = parsed
            entries.append((fp, 0, m, s, e))

    print(f'Building quality NPZ: {len(entries)} total reps '
          f'({sum(1 for e in entries if e[1]==1)} correct, '
          f'{sum(1 for e in entries if e[1]==0)} incorrect)')

    splits = {'train': [], 'test': []}

    for fpath, label, movement, subject, rep in entries:
        split = 'train' if subject in _TRAIN_SUBJECTS else 'test'
        try:
            pos = _load_seg_file(fpath)
            angles = compute_rom_angles(pos)   # (T, 12)
            T = angles.shape[0]
            splits[split].append({
                'angles':     angles,
                'length':     min(T, max_seq_len),
                'label':      label,
                'exercise':   movement - 1,   # 0-indexed
                'subject':    subject - 1,    # 0-indexed
            })
        except Exception as exc:
            print(f'  WARNING: skipping {os.path.basename(fpath)}: {exc}')

    os.makedirs(os.path.dirname(train_path) if os.path.dirname(train_path) else '.', exist_ok=True)

    path_map = {'train': train_path, 'test': test_path}
    for split_name, reps in splits.items():
        if not reps:
            print(f'WARNING: no {split_name} reps found')
            continue
        N = len(reps)
        seqs    = np.zeros((N, max_seq_len, 12), dtype=np.float32)
        lengths = np.zeros(N, dtype=np.int32)
        labels  = np.zeros(N, dtype=np.int32)
        exids   = np.zeros(N, dtype=np.int32)
        subids  = np.zeros(N, dtype=np.int32)

        for i, r in enumerate(reps):
            T_clip = r['length']
            seqs[i, :T_clip, :] = r['angles'][:T_clip]
            lengths[i] = T_clip
            labels[i]  = r['label']
            exids[i]   = r['exercise']
            subids[i]  = r['subject']

        out_path = path_map[split_name]
        np.savez_compressed(
            out_path,
            sequences      = seqs,
            lengths        = lengths,
            quality_labels = labels,
            exercise_ids   = exids,
            subject_ids    = subids,
        )
        n1 = labels.sum(); n0 = N - n1
        print(f'Saved {split_name} → {out_path}')
        print(f'  sequences: {seqs.shape}  correct={n1}  incorrect={n0}')
        print(f'  length range: {lengths.min()}–{lengths.max()} frames')


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class QualityDataset(Dataset):
    def __init__(self, npz_path, max_seq_len=300):
        data = np.load(npz_path)
        self.sequences = torch.from_numpy(data['sequences'])          # (N, T, 12)
        self.lengths   = torch.from_numpy(data['lengths'].astype(np.int64))
        self.labels    = torch.from_numpy(data['quality_labels'].astype(np.float32))
        self.exercise_ids = data['exercise_ids']
        self.subject_ids  = data['subject_ids']
        self.max_seq_len  = max_seq_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.lengths[idx], self.labels[idx]


def collate_fn(batch):
    seqs, lengths, labels = zip(*batch)
    seqs    = torch.stack(seqs)
    lengths = torch.stack(lengths)
    labels  = torch.stack(labels)
    # Sort descending by length (required by pack_padded_sequence on CPU)
    order   = lengths.argsort(descending=True)
    return seqs[order], lengths[order], labels[order]


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class QualityLSTM(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True,
                             dropout=0.0)
        self.drop1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden_dim, 64, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x, lengths):
        # x: (B, T, 12), lengths: (B,) sorted descending
        lengths_cpu = lengths.cpu().clamp(min=1)
        packed = pack_padded_sequence(x, lengths_cpu, batch_first=True,
                                      enforce_sorted=True)
        out1, _ = self.lstm1(packed)
        # Unpack → dropout → re-pack for lstm2
        out1_pad, _ = torch.nn.utils.rnn.pad_packed_sequence(out1, batch_first=True)
        out1_pad = self.drop1(out1_pad)
        packed2 = pack_padded_sequence(out1_pad, lengths_cpu, batch_first=True,
                                       enforce_sorted=True)
        _, (h2, _) = self.lstm2(packed2)
        feat = h2.squeeze(0)      # (B, 64)
        return self.fc(feat).squeeze(-1)   # (B,)


# ─────────────────────────────────────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_auc(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for seqs, lengths, labels in loader:
            seqs, lengths = seqs.to(device), lengths.to(device)
            logits = model(seqs, lengths).cpu().numpy()
            all_logits.append(logits)
            all_labels.append(labels.numpy())
    all_logits = np.concatenate(all_logits)
    all_labels = np.concatenate(all_labels)
    if not _HAS_SKLEARN or len(np.unique(all_labels)) < 2:
        probs = 1 / (1 + np.exp(-all_logits))
        acc = ((probs >= 0.5).astype(float) == all_labels).mean()
        return acc, acc   # return acc as proxy for AUC
    auc = roc_auc_score(all_labels, all_logits)
    probs = 1 / (1 + np.exp(-all_logits))
    acc = ((probs >= 0.5).astype(float) == all_labels).mean()
    return auc, acc


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    n = 0
    for seqs, lengths, labels in loader:
        seqs, lengths, labels = seqs.to(device), lengths.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(seqs, lengths)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(labels)
        n += len(labels)
    return total_loss / max(n, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Rep-level quality assessment (LSTM on ROM angles)')
    parser.add_argument('--data_train',  default='data/uiprmd_quality_train.npz')
    parser.add_argument('--data_test',   default='data/uiprmd_quality_test.npz')
    parser.add_argument('--raw_dir',     default='data/UI-PRMD/raw',
                        help='Raw UI-PRMD dir (used only if quality NPZ files are absent)')
    parser.add_argument('--epochs',      type=int,   default=50)
    parser.add_argument('--max_seq_len', type=int,   default=300)
    parser.add_argument('--hidden_dim',  type=int,   default=128)
    parser.add_argument('--batch_size',  type=int,   default=32)
    parser.add_argument('--lr',          type=float, default=1e-3)
    parser.add_argument('--save_model',  default='results/qa_rep_level_best.pt')
    parser.add_argument('--save_csv',    default='results/qa_rep_level_results.csv')
    args = parser.parse_args()

    # ── Build quality NPZ if absent ──────────────────────────────────────────
    if not os.path.isfile(args.data_train) or not os.path.isfile(args.data_test):
        print('Quality NPZ files not found — building from raw segmented data...')
        build_quality_npz(args.raw_dir, args.data_train, args.data_test, args.max_seq_len)

    # ── Load datasets ────────────────────────────────────────────────────────
    train_ds = QualityDataset(args.data_train, args.max_seq_len)
    test_ds  = QualityDataset(args.data_test,  args.max_seq_len)

    # Hold out subject 7 (0-indexed=6) from train as val
    val_mask   = train_ds.subject_ids == 6
    train_mask = ~val_mask
    train_idx  = np.where(train_mask)[0].tolist()
    val_idx    = np.where(val_mask)[0].tolist()

    train_sub = torch.utils.data.Subset(train_ds, train_idx)
    val_sub   = torch.utils.data.Subset(train_ds, val_idx)

    train_labels = train_ds.labels[train_idx].numpy()
    n_pos = train_labels.sum()
    n_neg = len(train_labels) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)

    print(f'\n=== Dataset ===')
    print(f'  Train: {len(train_sub)} sequences  (pos={int(n_pos)}, neg={int(n_neg)})')
    print(f'  Val  : {len(val_sub)} sequences  (subject 7 held out)')
    print(f'  Test : {len(test_ds)} sequences')
    print(f'  pos_weight: {pos_weight.item():.3f}')

    train_loader = DataLoader(train_sub, batch_size=args.batch_size,
                              shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_sub,   batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,   batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_fn)

    # ── Model / optimizer / scheduler ───────────────────────────────────────
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'\nDevice: {device}')

    model     = QualityLSTM(input_dim=12, hidden_dim=args.hidden_dim).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5, factor=0.5, verbose=False)

    best_val_auc = 0.0
    patience_ctr = 0
    early_stop_patience = 10
    history = []

    print(f'\n{"Epoch":>5} {"Loss":>10} {"ValAUC":>10} {"ValAcc":>10}')
    print('-' * 40)

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_auc, val_acc = compute_auc(model, val_loader, device)
        scheduler.step(val_auc)

        history.append({'epoch': epoch, 'loss': loss,
                        'val_auc': val_auc, 'val_acc': val_acc})
        print(f'{epoch:>5} {loss:>10.4f} {val_auc:>10.4f} {val_acc:>10.4f}')

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_ctr = 0
            os.makedirs(os.path.dirname(args.save_model)
                        if os.path.dirname(args.save_model) else '.', exist_ok=True)
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                        'val_auc': val_auc, 'args': vars(args)}, args.save_model)
        else:
            patience_ctr += 1
            if patience_ctr >= early_stop_patience:
                print(f'\nEarly stop at epoch {epoch} (patience={early_stop_patience})')
                break

    # ── Final evaluation on test ─────────────────────────────────────────────
    ckpt = torch.load(args.save_model, map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    test_auc, test_acc = compute_auc(model, test_loader, device)
    print(f'\n=== Test Results ===')
    print(f'  AUC : {test_auc:.4f}')
    print(f'  Acc : {test_acc:.4f}')
    print(f'  Best val AUC: {best_val_auc:.4f} (epoch {ckpt["epoch"]})')

    # ── Save CSV ─────────────────────────────────────────────────────────────
    if args.save_csv:
        os.makedirs(os.path.dirname(args.save_csv)
                    if os.path.dirname(args.save_csv) else '.', exist_ok=True)
        import csv
        with open(args.save_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['epoch', 'loss', 'val_auc', 'val_acc'])
            writer.writeheader()
            writer.writerows(history)
            writer.writerow({'epoch': 'TEST', 'loss': '',
                             'val_auc': f'{test_auc:.4f}',
                             'val_acc': f'{test_acc:.4f}'})
        print(f'\nResults saved to {args.save_csv}')


if __name__ == '__main__':
    main()
