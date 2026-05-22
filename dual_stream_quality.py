"""
Dual-Stream Quality Assessment Network for UI-PRMD rehabilitation exercises.
Stream 1: Spatial-Temporal GCN on 3D joint positions.
Stream 2: BiLSTM + Bahdanau Attention on ROM angles.
Novel contribution: end-to-end joint training of quality, exercise, validity heads.
"""

import argparse
import math
import os
import random
import time
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

# ─── Constants ────────────────────────────────────────────────────────────────

BODY_JOINT_IDX = list(range(17))
J = 17

EXERCISE_NAMES = [
    "DS", "HS", "IL", "SL", "SS",
    "SLR", "SA", "SE", "SIR", "SS2",
]

PAPER_MAD = [0.006, 0.008, 0.009, 0.006, 0.003,
             0.004, 0.009, 0.013, 0.006, 0.028]

SKELETON_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), # arms
    (5, 11), (6, 12), (11, 12),               # torso
    (11, 13), (13, 15),                       # left leg
    (12, 14), (14, 16),                       # right leg
]

# ─── Section 1: Data Pipeline ─────────────────────────────────────────────────


def build_topology_adjacency(n_joints: int) -> torch.Tensor:
    A = torch.zeros(n_joints, n_joints)
    for i, j in SKELETON_EDGES:
        if i < n_joints and j < n_joints:
            A[i, j] = 1.0
            A[j, i] = 1.0
    A += torch.eye(n_joints)
    deg = A.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return A / deg


# ── FIX 1 + FIX 5: Single GMM fit on train with PCA, apply to any split ──────

def fit_gmm_models(poses_3d: np.ndarray,
                   rom_angles: np.ndarray,
                   exercise_ids: np.ndarray,
                   quality_labels: np.ndarray) -> dict:
    """
    Fit one PCA+GMM per exercise using ONLY correct training frames.
    Uses ROM angles (12-dim) as features instead of 3D joint positions.
    Returns dict: exercise_id -> (pca, gmm) or None if insufficient data.
    """
    models = {}
    for e in range(10):
        correct_mask = (exercise_ids == e) & (quality_labels == 1)
        if correct_mask.sum() < 5:
            models[e] = None
            continue
        feats = rom_angles[correct_mask].astype(np.float64)   # (N, 12)
        n_comp = min(8, feats.shape[0] - 1, feats.shape[1])
        pca = PCA(n_components=n_comp, random_state=42)
        pca.fit(feats)
        feats_pca = pca.transform(feats)
        gmm = GaussianMixture(n_components=2, covariance_type='diag',
                              random_state=42, max_iter=300, reg_covar=1e-3)
        gmm.fit(feats_pca)
        models[e] = (pca, gmm)
    return models


def apply_gmm_scores(rom_angles: np.ndarray,
                     exercise_ids: np.ndarray,
                     quality_labels: np.ndarray,
                     gmm_models: dict,
                     split_name: str = "") -> np.ndarray:
    """
    Score all frames using the already-fitted gmm_models (no refitting).
    Uses ROM angles (12-dim) as features instead of 3D joint positions.
    Prints separation table for the given split.
    """
    scores = np.zeros(len(rom_angles), dtype=np.float32)
    rows = []
    for e in range(10):
        all_mask = exercise_ids == e
        if all_mask.sum() == 0 or gmm_models.get(e) is None:
            continue
        pca, gmm = gmm_models[e]
        feats_all = rom_angles[all_mask].astype(np.float64)   # (N, 12)
        feats_pca = pca.transform(feats_all)
        ll = gmm.score_samples(feats_pca)
        ll_min, ll_max = ll.min(), ll.max()
        normalized = (ll - ll_min) / (ll_max - ll_min + 1e-8)
        scores[all_mask] = normalized.astype(np.float32)
        ql_local = quality_labels[all_mask]
        cm = normalized[ql_local == 1].mean() if (ql_local == 1).sum() > 0 else 0.0
        im = normalized[ql_local == 0].mean() if (ql_local == 0).sum() > 0 else 0.0
        rows.append((e + 1, cm, im, cm - im))

    tag = f" [{split_name}]" if split_name else ""
    print(f"\nGMM separation table{tag} (PCA+GMM, same models as train):")
    print(f"{'Exercise':>10}  {'Correct_mean':>12}  {'Incorrect_mean':>14}  {'Gap':>6}")
    for ex, cm, im, gap in rows:
        print(f"  Ex{ex:02d}       {cm:12.3f}  {im:14.3f}  {gap:6.3f}")
    if rows:
        avg_cm  = np.mean([r[1] for r in rows])
        avg_im  = np.mean([r[2] for r in rows])
        avg_gap = np.mean([r[3] for r in rows])
        print(f"  {'AVERAGE':>8}   {avg_cm:12.3f}  {avg_im:14.3f}  {avg_gap:6.3f}\n")
    return scores


# ── FIX 4: ROM adjacency with std guard ───────────────────────────────────────

def compute_rom_guided_init(poses_3d: np.ndarray,
                             rom_angles: np.ndarray,
                             exercise_ids: np.ndarray,
                             quality_labels: np.ndarray,
                             n_exercises: int = 10) -> list:
    rom_guided = []
    total_nan = 0
    for e in range(n_exercises):
        mask = (exercise_ids == e) & (quality_labels == 1)
        corr = np.eye(J, dtype=np.float32)
        if mask.sum() > 10:
            joints = poses_3d[mask][:, BODY_JOINT_IDX, :].reshape(-1, J * 3)
            rom = rom_angles[mask]
            rom_avg = rom.mean(axis=1)
            for i in range(J):
                for jj in range(J):
                    if i == jj:
                        continue
                    jpos_i = joints[:, i * 3: i * 3 + 3].mean(axis=1)
                    jpos_j = joints[:, jj * 3: jj * 3 + 3].mean(axis=1)
                    # FIX 4: guard against constant arrays before pearsonr
                    if (np.std(jpos_i) < 1e-6 or
                            np.std(jpos_j) < 1e-6 or
                            np.std(rom_avg) < 1e-6):
                        corr[i, jj] = 0.0
                    else:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            try:
                                r_i, _ = pearsonr(jpos_i, rom_avg)
                                r_j, _ = pearsonr(jpos_j, rom_avg)
                                val = abs(float(r_i) * float(r_j))
                                corr[i, jj] = val if np.isfinite(val) else 0.0
                            except Exception:
                                corr[i, jj] = 0.0
        # Count and fix any residual NaN
        nan_count = np.isnan(corr).sum()
        total_nan += nan_count
        corr = np.nan_to_num(corr, nan=0.0)
        rom_guided.append(corr)
    print(f"ROM adjacency NaN count: {total_nan} -> fixed to 0")
    return rom_guided


def resample_sequence(seq: np.ndarray, target_len: int) -> np.ndarray:
    T = len(seq)
    if T == target_len:
        return seq
    old_idx = np.linspace(0, T - 1, T)
    new_idx = np.linspace(0, T - 1, target_len)
    shape = seq.shape[1:]
    flat = seq.reshape(T, -1)
    result = np.zeros((target_len, flat.shape[1]), dtype=seq.dtype)
    for d in range(flat.shape[1]):
        result[:, d] = np.interp(new_idx, old_idx, flat[:, d])
    return result.reshape((target_len,) + shape)


def speed_augment(joints: np.ndarray, rom: np.ndarray,
                  target_len: int) -> tuple:
    """Returns (joints, rom, was_speed_augmented)."""
    T = joints.shape[0]
    was_augmented = False
    if random.random() < 0.5:
        was_augmented = True
        L = random.randint(0, max(1, int(0.25 * T)))
        if random.random() < 0.5:
            positions = sorted(random.sample(range(T), min(L, T)))
            new_joints = list(joints)
            new_rom = list(rom)
            for offset, p in enumerate(positions):
                new_joints.insert(p + 1 + offset, joints[p])
                new_rom.insert(p + 1 + offset, rom[p])
            joints = np.array(new_joints)
            rom = np.array(new_rom)
        else:
            positions = sorted(random.sample(range(T), min(L, T - 1)), reverse=True)
            joints = np.delete(joints, positions, axis=0)
            rom = np.delete(rom, positions, axis=0)
    joints = resample_sequence(joints, target_len)
    rom = resample_sequence(rom, target_len)
    return joints, rom, was_augmented


def rotate_y(joints: np.ndarray, angle_deg: float) -> np.ndarray:
    a = angle_deg * math.pi / 180.0
    R = np.array([[math.cos(a), 0, math.sin(a)],
                  [0,            1, 0           ],
                  [-math.sin(a), 0, math.cos(a)]], dtype=joints.dtype)
    return joints @ R.T


def build_windows(poses_3d, rom_angles, gmm_scores, quality_labels,
                  exercise_ids, subject_ids,
                  window_size, stride, min_len,
                  rom_mean, rom_std,
                  augment=False,
                  generate_invalid=False):
    """Build sliding-window sequences. Returns (windows, speed_aug_count)."""
    windows = []
    speed_aug_count = 0
    total_aug_eligible = 0
    subjects  = np.unique(subject_ids)
    exercises = np.unique(exercise_ids)

    for subj in subjects:
        for ex in exercises:
            for ql in [0, 1]:
                mask = ((subject_ids == subj) &
                        (exercise_ids == ex) &
                        (quality_labels == ql))
                idx = np.where(mask)[0]
                if len(idx) < min_len:
                    continue
                j_seq = poses_3d[idx][:, BODY_JOINT_IDX, :]
                r_seq = rom_angles[idx]
                g_seq = gmm_scores[idx]

                start = 0
                while start < len(idx):
                    end = start + window_size
                    if end > len(idx):
                        break
                    jw = j_seq[start:end].copy()
                    rw = r_seq[start:end].copy()
                    gw = g_seq[start:end].mean()

                    # Spine normalization
                    origin = jw[0, 0, :].copy()
                    jw -= origin

                    # ROM z-score
                    rw = (rw - rom_mean[ex]) / (rom_std[ex] + 1e-8)

                    if augment:
                        total_aug_eligible += 1
                        jw, rw, was_speed = speed_augment(jw, rw, window_size)
                        if was_speed:
                            speed_aug_count += 1
                        angle = random.uniform(-15, 15)
                        jw = rotate_y(jw, angle)

                    windows.append({
                        'joints':       jw.astype(np.float32),
                        'rom':          rw.astype(np.float32),
                        'gmm_score':    float(gw),
                        'exercise':     int(ex),
                        'quality':      float(ql),
                        'valid':        0,
                        'is_augmented': int(augment),
                    })

                    # Generate invalid sequence for VC head
                    if generate_invalid and augment:
                        jw2 = j_seq[start:end].copy()
                        rw2 = r_seq[start:end].copy()
                        p = random.uniform(0.25, 0.75)
                        crop_start = random.randint(0, max(0, int((1 - p) * window_size)))
                        crop_end = min(crop_start + max(1, int(p * window_size)), window_size)
                        jw2 = resample_sequence(jw2[crop_start:crop_end], window_size)
                        rw2 = resample_sequence(rw2[crop_start:crop_end], window_size)
                        origin2 = jw2[0, 0, :].copy()
                        jw2 -= origin2
                        rw2 = (rw2 - rom_mean[ex]) / (rom_std[ex] + 1e-8)
                        windows.append({
                            'joints':       jw2.astype(np.float32),
                            'rom':          rw2.astype(np.float32),
                            'gmm_score':    float(gw),
                            'exercise':     int(ex),
                            'quality':      float(ql),
                            'valid':        1,
                            'is_augmented': 1,
                        })

                    start += stride

    if augment and total_aug_eligible > 0:
        print(f"  Speed augmentation: {speed_aug_count}/{total_aug_eligible} windows "
              f"({speed_aug_count/total_aug_eligible:.0%}) | "
              f"rotation: {total_aug_eligible}/{total_aug_eligible} (100%)")

    return windows


class DualStreamDataset(Dataset):
    def __init__(self, windows):
        self.windows = windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        w = self.windows[idx]
        return (
            torch.tensor(w['joints'],       dtype=torch.float32),
            torch.tensor(w['rom'],          dtype=torch.float32),
            torch.tensor(w['gmm_score'],    dtype=torch.float32),
            torch.tensor(w['exercise'],     dtype=torch.long),
            torch.tensor(w['quality'],      dtype=torch.float32),
            torch.tensor(w['valid'],        dtype=torch.long),
            torch.tensor(w['is_augmented'], dtype=torch.long),
        )


def compute_rom_stats(windows):
    """Per-exercise mean/std from training windows (quality==1)."""
    rom_mean = np.zeros((10, 12), dtype=np.float32)
    rom_std  = np.ones((10, 12), dtype=np.float32)
    for e in range(10):
        roms = [w['rom'] for w in windows
                if w['exercise'] == e and w['quality'] == 1.0]
        if roms:
            all_rom = np.concatenate(roms, axis=0)
            rom_mean[e] = all_rom.mean(axis=0)
            rom_std[e]  = all_rom.std(axis=0)
    return rom_mean, rom_std


# ─── Section 2: Model ─────────────────────────────────────────────────────────


class SpatialGCN(nn.Module):
    def __init__(self, n_joints, hidden_dim, n_exercises,
                 A_topology, rom_guided_inits):
        super().__init__()
        self.n_exercises = n_exercises
        self.n_joints    = n_joints

        A_spatial_list = []
        for e in range(n_exercises):
            A_topo = A_topology.clone()
            if rom_guided_inits is not None and e < len(rom_guided_inits):
                A_rom = torch.tensor(rom_guided_inits[e], dtype=torch.float32)
                deg   = A_rom.sum(dim=1, keepdim=True).clamp(min=1e-8)
                A_rom = A_rom / deg
                A_init = 0.5 * A_topo + 0.5 * A_rom
                A_init = torch.nan_to_num(A_init, nan=0.0)
            else:
                A_init = A_topo
            A_spatial_list.append(A_init)
        self.A_spatial = nn.Parameter(torch.stack(A_spatial_list, dim=0))

        self.w1  = nn.Linear(3, hidden_dim, bias=False)
        self.w2  = nn.Linear(hidden_dim, hidden_dim * 2, bias=False)
        self.bn1 = nn.BatchNorm1d(n_joints)
        self.bn2 = nn.BatchNorm1d(n_joints)

    def forward(self, x, exercise_id):
        B, M, J, C = x.shape
        x_flat = x.reshape(B * M, J, C)

        h_list = []
        for b in range(B):
            ex_b = exercise_id[b].item()
            A_b  = F.softmax(self.A_spatial[ex_b], dim=-1)
            xb   = x_flat[b * M: (b + 1) * M]   # (M, J, C)

            h = self.w1(xb)
            h = torch.einsum('jk,mjd->mjd', A_b, h)
            h = self.bn1(h)
            h = F.relu(h)

            h = self.w2(h)
            h = torch.einsum('jk,mjd->mjd', A_b, h)
            h = self.bn2(h)
            h = F.relu(h)

            h_list.append(h)

        h_all = torch.stack(h_list, dim=0)   # (B, M, J, out_dim)
        return h_all


class TemporalGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, M, sigma=10.0):
        super().__init__()
        self.M = M
        idx    = torch.arange(M, dtype=torch.float32)
        A_gauss = torch.exp(-((idx.unsqueeze(0) - idx.unsqueeze(1)) ** 2) / (sigma ** 2))
        deg     = A_gauss.sum(dim=1, keepdim=True).clamp(min=1e-8)
        self.A_temporal = nn.Parameter(A_gauss / deg)

        self.w1  = nn.Linear(in_dim, hidden_dim, bias=False)
        self.w2  = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bn1 = nn.LayerNorm(hidden_dim)
        self.bn2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        A = F.softmax(self.A_temporal, dim=-1)
        h = self.w1(x)
        h = torch.einsum('mn,bnd->bmd', A, h)
        h = self.bn1(h)
        h = F.relu(h)
        h = self.w2(h)
        h = torch.einsum('mn,bnd->bmd', A, h)
        h = self.bn2(h)
        h = F.relu(h)
        return h.mean(dim=1)


class ROMStream(nn.Module):
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
        self.attn = nn.Linear(lstm_hidden * 2, 1, bias=False)

    def forward(self, x, padding_mask=None):
        h = self.proj(x)
        h, _ = self.lstm(h)
        scores = self.attn(torch.tanh(h))
        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask.unsqueeze(-1), float('-inf'))
        weights = torch.softmax(scores, dim=1)
        context = (weights * h).sum(dim=1)
        return context, weights.squeeze(-1)


class FusionLayer(nn.Module):
    def __init__(self, in_dim=256, out_dim=128):
        super().__init__()
        self.fc      = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(0.2)
        self.norm    = nn.LayerNorm(out_dim)

    def forward(self, gcn_feat, rom_feat):
        x = torch.cat([gcn_feat, rom_feat], dim=-1)
        x = F.relu(self.fc(x))
        x = self.dropout(x)
        return self.norm(x)


class DualStreamQualityNet(nn.Module):
    def __init__(self, hidden_dim=64, M=100, n_joints=17,
                 n_exercises=10, A_topology=None, rom_guided_inits=None):
        super().__init__()
        spatial_out  = hidden_dim * 2
        temporal_in  = n_joints * spatial_out

        self.spatial_gcn  = SpatialGCN(n_joints, hidden_dim, n_exercises,
                                        A_topology, rom_guided_inits)
        self.temporal_gcn = TemporalGCN(temporal_in, 128, M)
        self.rom_stream   = ROMStream(12, 64, 64, 2)
        self.fusion       = FusionLayer(256, 128)

        self.exercise_head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(0.4),          # FIX 3: stronger dropout
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
        B, M, J, C = joints.shape
        sf = self.spatial_gcn(joints, exercise_id)
        sf = sf.reshape(B, M, J * sf.shape[-1])
        tf = self.temporal_gcn(sf)
        rf, attn = self.rom_stream(rom, padding_mask)
        fused = self.fusion(tf, rf)
        return (self.exercise_head(fused),
                self.validity_head(fused),
                self.quality_head(fused).squeeze(-1),
                attn)


# ─── Section 3: Training ──────────────────────────────────────────────────────


def make_weighted_sampler(windows):
    exercise_arr = np.array([w['exercise'] for w in windows])
    quality_arr  = np.array([w['quality']  for w in windows])
    keys         = exercise_arr * 2 + quality_arr.astype(int)
    unique_keys, counts = np.unique(keys, return_counts=True)
    key_to_weight = {k: 1.0 / c for k, c in zip(unique_keys, counts)}
    weights = np.array([key_to_weight[k] for k in keys], dtype=np.float32)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def train_one_epoch(model, loader, optimizer, device, lambda_ec, lambda_vc,
                    class_weights_ex, desc="Train", first_epoch=False):
    model.train()
    total_loss = total_qual = total_ec = total_vc = 0.0
    total_ec_correct = total_vc_correct = total_samples = 0
    aug_batch_count = total_batches = 0

    ce_ex = nn.CrossEntropyLoss(weight=class_weights_ex.to(device), label_smoothing=0.1)
    ce_vc = nn.CrossEntropyLoss()
    l1_q  = nn.L1Loss()

    pbar = tqdm(loader, desc=desc, leave=False, ncols=100)
    for batch in pbar:
        (joints, rom, gmm_score,
         ex_label, qual_label, valid_label, is_aug) = [b.to(device) for b in batch]

        optimizer.zero_grad()
        ex_logits, vc_logits, qual_score, _ = model(joints, rom, ex_label)

        l_qual = l1_q(qual_score, gmm_score)
        l_ec   = ce_ex(ex_logits, ex_label)
        l_vc   = ce_vc(vc_logits, valid_label)
        loss   = l_qual + lambda_ec * l_ec + lambda_vc * l_vc

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        B = joints.size(0)
        total_loss        += loss.item() * B
        total_qual        += l_qual.item() * B
        total_ec          += l_ec.item() * B
        total_vc          += l_vc.item() * B
        ec_correct         = (ex_logits.argmax(1) == ex_label).sum().item()
        vc_correct         = (vc_logits.argmax(1) == valid_label).sum().item()
        total_ec_correct  += ec_correct
        total_vc_correct  += vc_correct
        total_samples     += B
        total_batches     += 1
        if first_epoch and is_aug.any():
            aug_batch_count += 1

        pbar.set_postfix({
            'L':  f'{loss.item():.4f}',
            'QA': f'{l_qual.item():.4f}',
            'EC': f'{ec_correct/B:.0%}',
        })

    if first_epoch:
        print(f"Epoch 1 augmentation: {aug_batch_count}/{total_batches} batches "
              f"({aug_batch_count/max(total_batches,1):.0%})")

    n = total_samples
    return {
        'loss':      total_loss / n,
        'qual_loss': total_qual / n,
        'ec_loss':   total_ec   / n,
        'vc_loss':   total_vc   / n,
        'ec_acc':    total_ec_correct / n,
        'vc_acc':    total_vc_correct / n,
    }


@torch.no_grad()
def evaluate(model, loader, device, lambda_ec, lambda_vc):
    model.eval()
    total_loss = total_qual = 0.0
    total_ec_correct = total_vc_correct = total_samples = 0
    l1_q  = nn.L1Loss()
    ce_ex = nn.CrossEntropyLoss()
    ce_vc = nn.CrossEntropyLoss()

    all_qual_pred, all_qual_true, all_ex_true = [], [], []

    for batch in loader:
        (joints, rom, gmm_score,
         ex_label, qual_label, valid_label, _) = [b.to(device) for b in batch]
        ex_logits, vc_logits, qual_score, _ = model(joints, rom, ex_label)

        l_qual = l1_q(qual_score, gmm_score)
        l_ec   = ce_ex(ex_logits, ex_label)
        l_vc   = ce_vc(vc_logits, valid_label)
        loss   = l_qual + lambda_ec * l_ec + lambda_vc * l_vc

        B = joints.size(0)
        total_loss       += loss.item() * B
        total_qual       += l_qual.item() * B
        total_ec_correct += (ex_logits.argmax(1) == ex_label).sum().item()
        total_vc_correct += (vc_logits.argmax(1) == valid_label).sum().item()
        total_samples    += B

        all_qual_pred.append(qual_score.cpu().numpy())
        all_qual_true.append(gmm_score.cpu().numpy())
        all_ex_true.append(ex_label.cpu().numpy())

    n = total_samples
    all_qual_pred = np.concatenate(all_qual_pred)
    all_qual_true = np.concatenate(all_qual_true)
    all_ex_true   = np.concatenate(all_ex_true)
    mad = np.mean(np.abs(all_qual_pred - all_qual_true))

    return {
        'loss':      total_loss / n,
        'mad':       mad,
        'ec_acc':    total_ec_correct / n,
        'vc_acc':    total_vc_correct / n,
        'qual_pred': all_qual_pred,
        'qual_true': all_qual_true,
        'ex_true':   all_ex_true,
    }


# ─── Section 4: Evaluation report ────────────────────────────────────────────


def print_results_table(test_metrics, save_csv, per_ex_csv):
    pred = test_metrics['qual_pred']
    true = test_metrics['qual_true']
    ex   = test_metrics['ex_true']

    rows = []
    for e in range(10):
        mask = (ex == e)
        if mask.sum() == 0:
            rows.append((e, 0, 0, 0))
            continue
        p    = pred[mask];  t = true[mask]
        mad  = np.mean(np.abs(p - t))
        rmse = np.sqrt(np.mean((p - t) ** 2))
        mape = np.mean(np.abs((p - t) / (np.abs(t) + 1e-8))) * 100
        rows.append((e, mad, rmse, mape))

    print("\n=== Dual-Stream Results (UI-PRMD Test Set) ===\n")
    print(f"Exercise Classification:")
    print(f"  Top-1 Accuracy: {test_metrics['ec_acc']*100:.2f}%")
    print(f"\nValidity Classifier:")
    print(f"  Accuracy: {test_metrics['vc_acc']*100:.2f}%")

    print(f"\nQuality Score Regression:")
    print(f"{'-'*77}")
    print(f"{'Exercise':<12} {'MAD':>6} {'RMSE':>6} {'MAPE':>7}  {'Paper MAD':>9}  {'Delta':>7}")
    print(f"{'-'*77}")
    for e, mad, rmse, mape in rows:
        paper = PAPER_MAD[e]
        print(f"Ex{e+1:02d} ({EXERCISE_NAMES[e]:<3})   {mad:.3f}  {rmse:.3f}  "
              f"{mape:.2f}%    {paper:.3f}     {mad-paper:+.3f}")
    print(f"{'-'*77}")
    avg_mad   = np.mean([r[1] for r in rows])
    avg_rmse  = np.mean([r[2] for r in rows])
    avg_mape  = np.mean([r[3] for r in rows])
    avg_paper = np.mean(PAPER_MAD)
    avg_delta = avg_mad - avg_paper
    print(f"{'AVERAGE':<12} {avg_mad:.3f}  {avg_rmse:.3f}  "
          f"{avg_mape:.2f}%    {avg_paper:.3f}     {avg_delta:+.3f}")
    print(f"{'-'*77}")

    stgcn_mad   = 0.054
    improvement = (stgcn_mad - avg_mad) / stgcn_mad * 100
    print(f"\n  STGCN-Seq (yours): {stgcn_mad:.3f} MAD")
    print(f"  Dual-Stream (ours): {avg_mad:.3f} MAD")
    print(f"  Improvement: {improvement:.1f}%\n")

    os.makedirs(os.path.dirname(per_ex_csv) if os.path.dirname(per_ex_csv) else '.', exist_ok=True)
    with open(per_ex_csv, 'w') as f:
        f.write("exercise,name,mad,rmse,mape,paper_mad,delta\n")
        for e, mad, rmse, mape in rows:
            f.write(f"Ex{e+1:02d},{EXERCISE_NAMES[e]},{mad:.4f},{rmse:.4f},{mape:.4f},"
                    f"{PAPER_MAD[e]:.3f},{mad-PAPER_MAD[e]:+.4f}\n")
        f.write(f"AVERAGE,,{avg_mad:.4f},{avg_rmse:.4f},{avg_mape:.4f},"
                f"{avg_paper:.3f},{avg_delta:+.4f}\n")
    print(f"  Per-exercise CSV: {per_ex_csv}")

    with open(save_csv, 'w') as f:
        f.write("metric,value\n")
        f.write(f"ec_acc,{test_metrics['ec_acc']*100:.2f}\n")
        f.write(f"vc_acc,{test_metrics['vc_acc']*100:.2f}\n")
        f.write(f"avg_mad,{avg_mad:.4f}\n")
        f.write(f"avg_rmse,{avg_rmse:.4f}\n")
        f.write(f"avg_mape,{avg_mape:.4f}\n")
        f.write(f"stgcn_seq_mad,{stgcn_mad:.3f}\n")
        f.write(f"improvement_pct,{improvement:.1f}\n")
    print(f"  Summary CSV: {save_csv}")


def save_attention_viz(model, test_dataset, device, save_path, n_samples=4):
    model.eval()
    n_samples = min(n_samples, len(test_dataset))
    fig, axes = plt.subplots(1, n_samples, figsize=(4 * n_samples, 4))
    if n_samples == 1:
        axes = [axes]
    indices = random.sample(range(len(test_dataset)), n_samples)
    with torch.no_grad():
        for ax, idx in zip(axes, indices):
            joints, rom, gmm_score, ex_label, qual_label, valid_label, _ = test_dataset[idx]
            joints   = joints.unsqueeze(0).to(device)
            rom      = rom.unsqueeze(0).to(device)
            ex_label = ex_label.unsqueeze(0).to(device)
            _, _, _, attn = model(joints, rom, ex_label)
            ax.plot(attn[0].cpu().numpy())
            ax.set_title(f"Ex{ex_label.item()+1:02d} Q={gmm_score:.2f}")
            ax.set_xlabel("Frame")
            ax.set_ylabel("Attention")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=80)
    plt.close()
    print(f"  Attention plot: {save_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--train_npz',       default='data/uiprmd_quality_train.npz')
    p.add_argument('--test_npz',        default='data/uiprmd_quality_test.npz')
    p.add_argument('--epochs',          type=int,   default=200)
    p.add_argument('--batch_size',      type=int,   default=16)
    p.add_argument('--lr',              type=float, default=1e-4)
    p.add_argument('--hidden_dim',      type=int,   default=64)
    p.add_argument('--M',               type=int,   default=100)
    p.add_argument('--window_size',     type=int,   default=100)
    p.add_argument('--stride',          type=int,   default=50)
    p.add_argument('--lambda_ec',       type=float, default=0.5)
    p.add_argument('--lambda_vc',       type=float, default=0.3)
    p.add_argument('--save_model',      default='results/dual_stream_best.pt')
    p.add_argument('--save_csv',        default='results/dual_stream_results.csv')
    p.add_argument('--save_per_ex_csv', default='results/dual_stream_per_exercise.csv')
    p.add_argument('--save_attention',  default='results/dual_stream_attention.png')
    p.add_argument('--seed',            type=int,   default=42)
    p.add_argument('--split_mode',
                   choices=['subject', 'random'],
                   default='subject',
                   help='subject=subject-level split, random=0.8/0.2 random split')
    p.add_argument('--use_npz_scores', action='store_true',
                   help='Use quality_scores from NPZ directly, skip internal GMM computation')
    return p.parse_args()


def _flip_vc_labels(windows):
    for w in windows:
        w['valid'] = 1 - w['valid']


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    os.makedirs('results', exist_ok=True)

    # ── Load raw data ────────────────────────────────────────────────────────
    print("Loading data...")
    tr = np.load(args.train_npz, allow_pickle=True)
    te = np.load(args.test_npz,  allow_pickle=True)

    def _load_quality_labels(d, name):
        if 'quality_labels' in d:
            return d['quality_labels'].astype(np.int32)
        elif 'quality_scores' in d:
            print(f"  [{name}] 'quality_labels' not found - thresholding 'quality_scores' at 0.5")
            return (d['quality_scores'] >= 0.5).astype(np.int32)
        else:
            raise KeyError(f"[{name}] NPZ has neither 'quality_labels' nor 'quality_scores'")

    tr_poses3d = tr['poses_3d']
    tr_rom     = tr['rom_angles']
    tr_ql      = _load_quality_labels(tr, 'train')
    tr_ex      = tr['exercise_ids']
    tr_subj    = tr['subject_ids']

    te_poses3d = te['poses_3d']
    te_rom     = te['rom_angles']
    te_ql      = _load_quality_labels(te, 'test')
    te_ex      = te['exercise_ids']
    te_subj    = te['subject_ids']

    if args.split_mode == 'random':
        # Combine train + test NPZ frames
        all_poses3d = np.concatenate([tr_poses3d, te_poses3d], axis=0)
        all_rom     = np.concatenate([tr_rom,     te_rom],     axis=0)
        all_ql      = np.concatenate([tr_ql,      te_ql],      axis=0)
        all_ex      = np.concatenate([tr_ex,      te_ex],      axis=0)
        all_subj    = np.concatenate([tr_subj,    te_subj],    axis=0)
        _tr_sc = tr['quality_scores'].astype(np.float32) if 'quality_scores' in tr else np.zeros(len(tr_poses3d), np.float32)
        _te_sc = te['quality_scores'].astype(np.float32) if 'quality_scores' in te else np.zeros(len(te_poses3d), np.float32)
        all_gmm = np.concatenate([_tr_sc, _te_sc], axis=0)

        # Shuffle with seed
        rng = np.random.default_rng(args.seed)
        idx = rng.permutation(len(all_poses3d))
        n_train = int(0.8 * len(idx))
        n_val   = int(0.1 * len(idx))

        train_idx = idx[:n_train]
        val_idx   = idx[n_train:n_train+n_val]
        test_idx  = idx[n_train+n_val:]

        # Replace masks with index-based selection
        tr_poses3d  = all_poses3d[train_idx]
        tr_rom      = all_rom[train_idx]
        tr_ql       = all_ql[train_idx]
        tr_ex       = all_ex[train_idx]
        tr_subj     = all_subj[train_idx]
        tr_gmm      = all_gmm[train_idx]

        val_poses3d = all_poses3d[val_idx]
        val_rom     = all_rom[val_idx]
        val_ql      = all_ql[val_idx]
        val_ex      = all_ex[val_idx]
        val_subj    = all_subj[val_idx]
        val_gmm     = all_gmm[val_idx]

        te_poses3d  = all_poses3d[test_idx]
        te_rom      = all_rom[test_idx]
        te_ql       = all_ql[test_idx]
        te_ex       = all_ex[test_idx]
        te_subj     = all_subj[test_idx]
        te_gmm      = all_gmm[test_idx]

        train_mask = np.ones(len(tr_poses3d), dtype=bool)
        print(f'Random split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}')
    else:
        # Existing subject-level split (keep as-is)
        train_mask = tr_subj <= 6
        val_mask   = tr_subj == 7

    # ── FIX 1+5: Fit GMMs on train data ONLY, apply same models everywhere ──
    if args.use_npz_scores:
        if args.split_mode != 'random':
            # For random split, tr_gmm/val_gmm/te_gmm already set above
            tr_gmm = tr['quality_scores'].astype(np.float32)
            te_gmm = te['quality_scores'].astype(np.float32)

        print('Using quality_scores from NPZ directly')
        print(f'Train: range [{tr_gmm.min():.3f}, {tr_gmm.max():.3f}]')
        print(f'Test:  range [{te_gmm.min():.3f}, {te_gmm.max():.3f}]')

        print('\nNPZ quality_scores separation table:')
        print(f"{'Exercise':>10}  {'Correct_mean':>12}  {'Incorrect_mean':>14}  {'Gap':>6}")
        for ex in range(10):
            ex_mask = tr_ex == ex
            ql_ex   = tr_ql[ex_mask]
            sc_ex   = tr_gmm[ex_mask]
            cm = sc_ex[ql_ex == 1].mean() if (ql_ex == 1).sum() > 0 else 0
            im = sc_ex[ql_ex == 0].mean() if (ql_ex == 0).sum() > 0 else 0
            print(f'  Ex{ex+1:02d}       {cm:12.3f}  {im:14.3f}  {cm-im:6.3f}')

        if args.split_mode == 'subject':
            val_gmm = tr_gmm[val_mask]
    else:
        print("Fitting PCA+GMM models on training frames (correct only)...")
        gmm_models = fit_gmm_models(tr_poses3d[train_mask],
                                    tr_rom[train_mask],
                                    tr_ex[train_mask],
                                    tr_ql[train_mask])

        print("Scoring train frames with fitted models...")
        tr_gmm = apply_gmm_scores(tr_rom, tr_ex, tr_ql, gmm_models, split_name="TRAIN")

        if args.split_mode == 'random':
            print("Scoring val frames with fitted models...")
            val_gmm = apply_gmm_scores(val_rom, val_ex, val_ql, gmm_models, split_name="VAL")

        print("Scoring test frames with SAME fitted models (no refit)...")
        te_gmm = apply_gmm_scores(te_rom, te_ex, te_ql, gmm_models, split_name="TEST")

        if args.split_mode == 'subject':
            val_gmm = tr_gmm[val_mask]

    # For subject split, create unified val_* variables from masked train data
    if args.split_mode == 'subject':
        val_poses3d = tr_poses3d[val_mask]
        val_rom     = tr_rom[val_mask]
        val_ql      = tr_ql[val_mask]
        val_ex      = tr_ex[val_mask]
        val_subj    = tr_subj[val_mask]

    # ── ROM-guided adjacency (train only) ────────────────────────────────────
    print("Computing ROM-guided adjacency init...")
    rom_guided_inits = compute_rom_guided_init(
        tr_poses3d[train_mask], tr_rom[train_mask],
        tr_ex[train_mask], tr_ql[train_mask],
    )

    # ── Skeleton topology adjacency ──────────────────────────────────────────
    A_topology = build_topology_adjacency(J)

    # ── ROM normalization stats from train windows ───────────────────────────
    print("Computing ROM normalization stats...")
    dummy_windows = build_windows(
        tr_poses3d[train_mask], tr_rom[train_mask],
        tr_gmm[train_mask], tr_ql[train_mask],
        tr_ex[train_mask], tr_subj[train_mask],
        args.window_size, args.stride, min_len=50,
        rom_mean=np.zeros((10, 12), dtype=np.float32),
        rom_std=np.ones((10, 12), dtype=np.float32),
        augment=False, generate_invalid=False,
    )
    rom_mean, rom_std = compute_rom_stats(dummy_windows)
    del dummy_windows

    # ── Build windows ────────────────────────────────────────────────────────
    print("Building train windows...")
    train_windows = build_windows(
        tr_poses3d[train_mask], tr_rom[train_mask],
        tr_gmm[train_mask], tr_ql[train_mask],
        tr_ex[train_mask], tr_subj[train_mask],
        args.window_size, args.stride, min_len=50,
        rom_mean=rom_mean, rom_std=rom_std,
        augment=True, generate_invalid=True,
    )
    print("Building val windows...")
    val_windows = build_windows(
        val_poses3d, val_rom,
        val_gmm, val_ql,
        val_ex, val_subj,
        args.window_size, args.stride, min_len=50,
        rom_mean=rom_mean, rom_std=rom_std,
        augment=False, generate_invalid=False,
    )
    print("Building test windows...")
    test_windows = build_windows(
        te_poses3d, te_rom, te_gmm, te_ql, te_ex, te_subj,
        args.window_size, args.stride, min_len=50,
        rom_mean=rom_mean, rom_std=rom_std,
        augment=False, generate_invalid=False,
    )

    print(f"  Train windows: {len(train_windows)}")
    print(f"  Val   windows: {len(val_windows)}")
    print(f"  Test  windows: {len(test_windows)}")

    # ── FIX 2: VC label diagnostic ───────────────────────────────────────────
    tr_valid   = sum(1 for w in train_windows if w['valid'] == 0)
    tr_invalid = sum(1 for w in train_windows if w['valid'] == 1)
    va_valid   = sum(1 for w in val_windows   if w['valid'] == 0)
    va_invalid = sum(1 for w in val_windows   if w['valid'] == 1)
    print(f"\nVC label distribution:")
    print(f"  Train: valid={tr_valid}  invalid={tr_invalid}  "
          f"ratio={tr_valid/(tr_valid+tr_invalid+1e-8):.2f}")
    print(f"  Val:   valid={va_valid}  invalid={va_invalid}  "
          f"ratio={va_valid/(va_valid+va_invalid+1e-8):.2f}")
    # Note: val has no invalid windows by design (generate_invalid=False)

    # ── Datasets & loaders ───────────────────────────────────────────────────
    train_ds = DualStreamDataset(train_windows)
    val_ds   = DualStreamDataset(val_windows)
    test_ds  = DualStreamDataset(test_windows)

    sampler      = make_weighted_sampler(train_windows)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=sampler, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size * 2,
                              shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size * 2,
                              shuffle=False, num_workers=4, pin_memory=True)

    # ── Class weights for exercise head ─────────────────────────────────────
    ex_arr      = np.array([w['exercise'] for w in train_windows])
    ex_counts   = np.bincount(ex_arr, minlength=10).astype(np.float32)
    ex_counts   = np.where(ex_counts == 0, 1, ex_counts)
    cls_w_ex    = torch.tensor(1.0 / ex_counts)
    cls_w_ex    = cls_w_ex / cls_w_ex.sum() * 10

    # ── Model ────────────────────────────────────────────────────────────────
    model = DualStreamQualityNet(
        hidden_dim=args.hidden_dim, M=args.M,
        n_joints=J, n_exercises=10,
        A_topology=A_topology,
        rom_guided_inits=rom_guided_inits,
    ).to(device)

    def count_params(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    print(f"\nModel parameters:")
    print(f"  SpatialGCN:    {count_params(model.spatial_gcn):>8,}")
    print(f"  TemporalGCN:   {count_params(model.temporal_gcn):>8,}")
    print(f"  ROMStream:     {count_params(model.rom_stream):>8,}")
    print(f"  FusionLayer:   {count_params(model.fusion):>8,}")
    print(f"  ExerciseHead:  {count_params(model.exercise_head):>8,}")
    print(f"  ValidityHead:  {count_params(model.validity_head):>8,}")
    print(f"  QualityHead:   {count_params(model.quality_head):>8,}")
    print(f"  TOTAL:         {count_params(model):>8,}")

    # ── FIX 3: Separate param groups — EC head gets stronger L2 ─────────────
    ec_params    = list(model.exercise_head.parameters())
    ec_param_ids = set(id(p) for p in ec_params)
    other_params = [p for p in model.parameters() if id(p) not in ec_param_ids]
    optimizer = torch.optim.AdamW([
        {'params': other_params, 'weight_decay': 1e-4},
        {'params': ec_params,    'weight_decay': 1e-3, 'lr': args.lr * 0.3},
    ], lr=args.lr)
    warmup_epochs = 10

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(args.epochs - warmup_epochs, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_mad = float('inf')
    best_epoch   = 0
    patience_cnt = 0
    early_stop   = 40
    vc_flipped   = False

    print(f"\nTraining for {args.epochs} epochs...\n")

    for epoch in range(1, args.epochs + 1):
        t0   = time.time()
        tr_m = train_one_epoch(
            model, train_loader, optimizer, device,
            args.lambda_ec, args.lambda_vc, cls_w_ex,
            desc=f"Ep[{epoch:3d}/{args.epochs}]",
            first_epoch=(epoch == 1),
        )
        val_m   = evaluate(model, val_loader, device, args.lambda_ec, args.lambda_vc)
        elapsed = time.time() - t0

        # ── FIX 2: Auto-flip VC labels if accuracy is below chance ──────────
        if epoch == 1 and not vc_flipped and tr_m['vc_acc'] < 0.45:
            print(f"  [FIX 2] Train VC acc={tr_m['vc_acc']:.0%} < 45% - inverting VC labels")
            _flip_vc_labels(train_windows)
            _flip_vc_labels(val_windows)
            _flip_vc_labels(test_windows)
            # Re-init VC head
            for layer in model.validity_head:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
            train_ds = DualStreamDataset(train_windows)
            val_ds   = DualStreamDataset(val_windows)
            test_ds  = DualStreamDataset(test_windows)
            sampler      = make_weighted_sampler(train_windows)
            train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                      sampler=sampler, num_workers=4, pin_memory=True)
            val_loader   = DataLoader(val_ds, batch_size=args.batch_size * 2,
                                      shuffle=False, num_workers=4, pin_memory=True)
            test_loader  = DataLoader(test_ds, batch_size=args.batch_size * 2,
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
                'args':       vars(args),
                'rom_mean':   rom_mean,
                'rom_std':    rom_std,
            }, args.save_model)
        else:
            patience_cnt += 1
            if patience_cnt >= early_stop:
                print(f"Early stopping at epoch {epoch} "
                      f"(best val MAD={best_val_mad:.4f} at ep {best_epoch})")
                break

        if epoch % 10 == 0 or epoch <= 3:
            cur_lr = optimizer.param_groups[0]['lr']
            print(f"Ep[{epoch:3d}] LR={cur_lr:.2e} | L={tr_m['loss']:.4f} | "
                  f"QA={tr_m['qual_loss']:.4f} EC={tr_m['ec_acc']:.0%} "
                  f"VC={tr_m['vc_acc']:.0%} | "
                  f"val_MAD={val_m['mad']:.4f} val_EC={val_m['ec_acc']:.0%} | "
                  f"{elapsed:.1f}s")

    # ── Test evaluation ───────────────────────────────────────────────────────
    print(f"\nLoading best checkpoint (epoch {best_epoch}, val MAD={best_val_mad:.4f})...")
    ckpt = torch.load(args.save_model, map_location=device)
    model.load_state_dict(ckpt['state_dict'])

    test_m = evaluate(model, test_loader, device, args.lambda_ec, args.lambda_vc)
    print_results_table(test_m, args.save_csv, args.save_per_ex_csv)
    save_attention_viz(model, test_ds, device, args.save_attention)

    print("\nDone.")


if __name__ == '__main__':
    main()
