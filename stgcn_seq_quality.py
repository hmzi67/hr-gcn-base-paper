"""
STGCN-Seq for UI-PRMD quality assessment.
Implements Kourbane et al., 2025, Computers in Biology and Medicine.

Architecture:
  - SpatialGCN  : per-exercise learnable adjacency, k=2 layers
  - TemporalGCN : Gaussian-init learnable adjacency over M frames, k=2 layers
  - ExerciseClassifier (EC) : 10-class CrossEntropy
  - ValidityClassifier  (VC): 2-class CrossEntropy (0=valid, 1=invalid)
  - QualityHead (QR)        : L1 regression of quality score in [0,1]

Training stages:
  Stage 1 — EC alone
  Stage 2 — VC alone  (with invalid sequences = randomly cropped originals)
  Stage 3 — QR (backbone fine-tuned, EC/VC can be frozen)

Data format (uiprmd_quality_*.npz):
  sequences     : (N, T_max, 12)  ROM angles in degrees, zero-padded
  lengths       : (N,)            actual frame counts
  quality_labels: (N,)            binary 0/1
  exercise_ids  : (N,)            0-9
  subject_ids   : (N,)            0-indexed
"""

import argparse
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.mixture import GaussianMixture

# ── ROM angle definitions ──────────────────────────────────────────────────
# 12 ROM angles (from prepare_data_uiprmd.py):
#  0:cerv_pitch  1:trunk_flex
#  2:l_sho_flex  3:r_sho_flex  4:l_sho_abd  5:r_sho_abd
#  6:l_hip       7:r_hip
#  8:l_knee      9:r_knee
# 10:l_ankle    11:r_ankle
ROM_J = 12  # number of ROM angle nodes

ROM_EDGES = [
    (0, 1),            # cervical - trunk
    (1, 2), (1, 3),    # trunk - shoulder flex (L/R)
    (2, 4), (3, 5),    # shoulder flex - abduction (same side)
    (2, 3), (4, 5),    # bilateral shoulders
    (1, 6), (1, 7),    # trunk - hips
    (6, 7),            # bilateral hips
    (6, 8), (7, 9),    # hip - knee
    (8, 9),            # bilateral knees
    (8, 10), (9, 11),  # knee - ankle
    (10, 11),          # bilateral ankles
]

EXERCISE_NAMES = [
    "DS", "HS", "IL", "SL", "SS", "SLR", "SA", "SE", "SIR", "SS2"
]

PAPER_MAD = [0.006, 0.008, 0.009, 0.006, 0.003, 0.004, 0.009, 0.013, 0.006, 0.028]

ROM_NORM = 180.0  # divide angles by this to get [0, 1] range


# ── Graph helpers ──────────────────────────────────────────────────────────

def build_adjacency(J: int, edges) -> np.ndarray:
    A = np.zeros((J, J), dtype=np.float32)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    np.fill_diagonal(A, 1.0)
    D = A.sum(axis=1, keepdims=True).clip(min=1e-8)
    return A / D


def gaussian_adjacency(M: int, sigma: float = 10.0) -> torch.Tensor:
    idx = torch.arange(M, dtype=torch.float32)
    diff = (idx.unsqueeze(0) - idx.unsqueeze(1)) ** 2
    A = torch.exp(-diff / (2.0 * sigma ** 2))
    D = A.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return A / D


# ── Model components ───────────────────────────────────────────────────────

class SpatialGCN(nn.Module):
    """
    Per-exercise learnable adjacency GCN over joints/angle nodes.
    Input:  (B, J, in_dim)
    Output: (B, J, hidden_dim)
    """
    def __init__(self, in_dim: int, hidden_dim: int, J: int,
                 n_exercises: int = 10, n_layers: int = 2,
                 edges=None):
        super().__init__()
        if edges is None:
            edges = ROM_EDGES
        self.n_exercises = n_exercises
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        A_init = torch.FloatTensor(build_adjacency(J, edges))
        self.A_spatial = nn.ParameterList(
            [nn.Parameter(A_init.clone()) for _ in range(n_exercises)]
        )

        dims = [in_dim] + [hidden_dim] * n_layers
        self.W = nn.ModuleList(
            [nn.Linear(dims[k], dims[k + 1], bias=False) for k in range(n_layers)]
        )
        self.bn = nn.ModuleList(
            [nn.BatchNorm1d(J) for _ in range(n_layers)]
        )

    def forward(self, x: torch.Tensor, exercise_id: torch.Tensor) -> torch.Tensor:
        B, J, _ = x.shape
        out = x
        for k in range(self.n_layers):
            result = torch.zeros(B, J, self.hidden_dim, device=x.device, dtype=x.dtype)
            for eid in range(self.n_exercises):
                mask = (exercise_id == eid)
                if mask.sum() == 0:
                    continue
                A = torch.softmax(self.A_spatial[eid], dim=-1)
                h = out[mask]
                h = torch.einsum('ij,njc->nic', A, h)
                h = self.W[k](h)
                result[mask] = h
            out = F.relu(self.bn[k](result))
        return out


class TemporalGCN(nn.Module):
    """
    Learnable Gaussian-initialised adjacency GCN over time.
    Input:  (B, M, in_dim)
    Output: (B, hidden_dim)  — global-average-pooled
    """
    def __init__(self, in_dim: int, hidden_dim: int, M: int = 100,
                 n_layers: int = 2, sigma: float = 10.0):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.A_temporal = nn.Parameter(gaussian_adjacency(M, sigma))

        dims = [in_dim] + [hidden_dim] * n_layers
        self.W = nn.ModuleList(
            [nn.Linear(dims[k], dims[k + 1], bias=False) for k in range(n_layers)]
        )
        self.bn = nn.ModuleList(
            [nn.BatchNorm1d(M) for _ in range(n_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, M, _ = x.shape
        out = x
        A = torch.softmax(self.A_temporal, dim=-1)
        for k in range(self.n_layers):
            out = torch.einsum('ij,bjc->bic', A, out)
            out = self.W[k](out)
            out = F.relu(self.bn[k](out))
        out = out.mean(dim=1)
        return out


class STGCNBackbone(nn.Module):
    """
    Spatial-temporal GCN backbone.
    Input x: (B, M, J, C)  where J=ROM_J=12, C=1 for ROM angle data.
    """
    def __init__(self, J: int = ROM_J, in_dim: int = 1,
                 hidden_dim: int = 64, M: int = 100,
                 n_exercises: int = 10, edges=None):
        super().__init__()
        self.J = J
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.M = M

        self.spatial_gcn  = SpatialGCN(in_dim, hidden_dim, J, n_exercises, edges=edges)
        self.temporal_gcn = TemporalGCN(J * hidden_dim, hidden_dim, M)

    def forward(self, x: torch.Tensor, exercise_id: torch.Tensor) -> torch.Tensor:
        # x: (B, M, J, C)
        B, M, J, C = x.shape
        x_flat = x.reshape(B * M, J, C)
        ex_rep = exercise_id.unsqueeze(1).expand(B, M).reshape(B * M)
        sp_out = self.spatial_gcn(x_flat, ex_rep)          # (B*M, J, hidden)
        tmp_in = sp_out.reshape(B, M, J * self.hidden_dim)
        feat   = self.temporal_gcn(tmp_in)                 # (B, hidden)
        return feat


class ExerciseClassifier(nn.Module):
    def __init__(self, hidden_dim: int, n_exercises: int = 10):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.ReLU(),
            nn.Linear(32, n_exercises),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.fc(feat)


class ValidityClassifier(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.fc(feat)


class QualityHead(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid(),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.fc(feat)


class STGCNSeq(nn.Module):
    def __init__(self, J: int = ROM_J, in_dim: int = 1,
                 hidden_dim: int = 64, M: int = 100,
                 n_exercises: int = 10, edges=None):
        super().__init__()
        self.backbone      = STGCNBackbone(J, in_dim, hidden_dim, M, n_exercises, edges)
        self.exercise_clf  = ExerciseClassifier(hidden_dim, n_exercises)
        self.validity_clf  = ValidityClassifier(hidden_dim)
        self.quality_head  = QualityHead(hidden_dim)

    def forward(self, x: torch.Tensor, exercise_id: torch.Tensor):
        feat = self.backbone(x, exercise_id)
        return (
            self.exercise_clf(feat),   # (B, 10)
            self.validity_clf(feat),   # (B, 2)
            self.quality_head(feat),   # (B, 1)
        )


# ── Sequence helpers ───────────────────────────────────────────────────────

def _resize_sequence(seq: np.ndarray, target_len: int) -> np.ndarray:
    """Linearly interpolate sequence (T, F) to (target_len, F)."""
    T, F = seq.shape
    if T == target_len:
        return seq
    x_old = np.linspace(0.0, 1.0, T)
    x_new = np.linspace(0.0, 1.0, target_len)
    out = np.empty((target_len, F), dtype=seq.dtype)
    for f in range(F):
        out[:, f] = np.interp(x_new, x_old, seq[:, f])
    return out


def _augment_rom(seq: np.ndarray, M: int) -> np.ndarray:
    """Speed augmentation (random frame drop/repeat) + small Gaussian noise."""
    T, F = seq.shape
    L = np.random.randint(0, max(1, int(0.25 * T)) + 1)
    if L > 0:
        if np.random.random() < 0.5:
            extra = np.random.choice(T, L, replace=False)
            seq = np.concatenate([seq, seq[extra]], axis=0)
        else:
            keep = np.sort(np.random.choice(T, max(2, T - L), replace=False))
            seq = seq[keep]
    seq = _resize_sequence(seq, M)
    seq = seq + np.random.randn(*seq.shape).astype(np.float32) * 0.005
    return seq


def _build_invalid_seq(seq: np.ndarray) -> np.ndarray:
    """Return a random crop (25-75 %) to simulate an invalid (truncated) sequence."""
    T = len(seq)
    p = np.random.uniform(0.25, 0.75)
    crop_len = max(2, int(p * T))
    start = np.random.randint(0, max(1, T - crop_len))
    return seq[start: start + crop_len]


# ── GMM quality scoring ────────────────────────────────────────────────────

def _compute_gmm_from_arrays(rom_means, exercise_ids, quality_labels,
                              print_table=True):
    """
    Fit per-exercise GMM on correct-sequence mean ROM angles; score all sequences.
    rom_means   : (N, 12) — mean ROM angles per sequence
    exercise_ids: (N,)
    quality_labels: (N,)  binary 0/1
    Returns quality_scores (N,) float32 in [0, 1].
    """
    N = len(rom_means)
    scores = np.zeros(N, dtype=np.float32)

    if print_table:
        print("\n=== GMM Score Separation ===")
        print(f"{'Exercise':<10}{'Correct_mean':>14}{'Incorrect_mean':>16}{'Gap':>8}")

    gaps = []
    for e in range(10):
        mask_correct   = (exercise_ids == e) & (quality_labels == 1)
        mask_incorrect = (exercise_ids == e) & (quality_labels == 0)
        mask_all       = (exercise_ids == e)

        if mask_correct.sum() < 2:
            if mask_all.sum() > 0:
                scores[mask_all] = 0.5
            if print_table:
                print(f"Ex{e+1:02d}{'':6} {'N/A':>14} {'N/A':>16} {'N/A':>8}")
            continue

        gmm = GaussianMixture(
            n_components=2, covariance_type='diag',
            random_state=42, max_iter=200
        )
        gmm.fit(rom_means[mask_correct])

        ll = gmm.score_samples(rom_means[mask_all])
        ll_min, ll_max = ll.min(), ll.max()
        scores[mask_all] = ((ll - ll_min) / (ll_max - ll_min + 1e-8)).astype(np.float32)

        if print_table:
            c_mean = float(scores[mask_correct].mean()) if mask_correct.sum() > 0 else float('nan')
            i_mean = float(scores[mask_incorrect].mean()) if mask_incorrect.sum() > 0 else float('nan')
            if not (np.isnan(c_mean) or np.isnan(i_mean)):
                gap = c_mean - i_mean
                gaps.append(gap)
                gap_str = f"{gap:.3f}"
            else:
                gap_str = "N/A"
            print(f"Ex{e+1:02d}{'':6} {c_mean:>14.3f} {i_mean:>16.3f} {gap_str:>8}")

    if print_table:
        avg_gap = float(np.mean(gaps)) if gaps else 0.0
        print(f"{'AVERAGE':<10}{'':>14}{'':>16}{avg_gap:>8.3f}")
        if avg_gap > 0.1:
            print("Good separation: Gap > 0.1")
        elif avg_gap < 0.05:
            print("Poor separation: Gap < 0.05")

    return scores


def compute_gmm_scores(npz_path):
    """
    Compute GMM-based quality scores for a single quality NPZ file.
    Returns (N,) float32 scores in [0, 1].
    """
    d = np.load(npz_path, allow_pickle=True)
    sequences      = d['sequences']          # (N, T_max, 12)
    exercise_ids   = d['exercise_ids']
    quality_labels = d['quality_labels']
    rom_means      = sequences.mean(axis=1)  # (N, 12) — temporal mean per sequence
    return _compute_gmm_from_arrays(rom_means, exercise_ids, quality_labels,
                                    print_table=True)


# ── Dataset ────────────────────────────────────────────────────────────────

class UIPromdSeqDataset(Dataset):
    """
    Loads pre-built ROM angle sequences from a quality NPZ file.

    NPZ keys expected:
      sequences     (N, T_max, 12)  ROM angles in degrees
      lengths       (N,)            actual frame counts per sequence
      quality_labels(N,)            binary 0=incorrect, 1=correct
      exercise_ids  (N,)
      subject_ids   (N,)

    Each item: (seq_resized: (M, 12, 1), eid, quality_score, validity)
    validity: 0=real sequence, 1=synthetically cropped (for VC training)
    """

    def __init__(self, npz_path: str = None, M: int = 100,
                 augment: bool = False,
                 val_subject: int = None,
                 is_val: bool = False,
                 include_invalid: bool = False,
                 seed: int = 42,
                 _prebuilt_items=None,
                 ext_quality_scores=None):
        self.M = M
        self.augment = augment

        if _prebuilt_items is not None:
            self._items = list(_prebuilt_items)
            return

        d = np.load(npz_path, allow_pickle=True)
        sequences      = d['sequences']           # (N, T_max, 12)
        lengths        = d['lengths']             # (N,)
        exercise_ids   = d['exercise_ids']
        subject_ids    = d['subject_ids']
        quality_labels = d['quality_labels']

        if ext_quality_scores is not None:
            quality_scores = ext_quality_scores
        else:
            quality_scores = quality_labels.astype(np.float32)

        # Subject split
        if val_subject is not None:
            keep = (subject_ids == val_subject) if is_val else (subject_ids != val_subject)
            sequences      = sequences[keep]
            lengths        = lengths[keep]
            exercise_ids   = exercise_ids[keep]
            quality_scores = quality_scores[keep]

        self._items = _build_items_from_arrays(
            sequences, lengths, exercise_ids, quality_scores,
            M, include_invalid=include_invalid)

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        seq, eid, qs, validity = self._items[idx]
        if self.augment:
            seq = _augment_rom(seq.copy(), self.M)  # (M, 12)
        # reshape to (M, 12, 1) for SpatialGCN (J=12, C=1)
        poses = torch.from_numpy(seq.reshape(self.M, ROM_J, 1))
        return {
            'poses':         poses,
            'exercise_id':   torch.tensor(eid,      dtype=torch.long),
            'quality_score': torch.tensor([qs],     dtype=torch.float32),
            'validity':      torch.tensor(validity, dtype=torch.long),
        }


def _build_items_from_arrays(sequences, lengths, exercise_ids, quality_scores,
                              M, include_invalid=False):
    """
    Build (seq, eid, qs, validity) tuples from quality-NPZ arrays.
    sequences     : (N, T_max, 12)
    lengths       : (N,)
    exercise_ids  : (N,)
    quality_scores: (N,)
    Returns list of (ndarray(M,12), int, float, int).
    """
    items = []
    N = len(sequences)
    for i in range(N):
        T = int(lengths[i])
        raw = sequences[i, :T, :].copy().astype(np.float32)
        raw /= ROM_NORM  # normalise to [0, 1]
        proc = _resize_sequence(raw, M)
        qs   = float(quality_scores[i])
        eid  = int(exercise_ids[i])
        items.append((proc, eid, qs, 0))  # 0 = valid

        if include_invalid:
            inv_raw  = _build_invalid_seq(raw)
            inv_proc = _resize_sequence(inv_raw, M)
            items.append((inv_proc, eid, qs, 1))  # 1 = invalid

    return items


# ── Training helpers ───────────────────────────────────────────────────────

def _run_epoch_ec(model, loader, optimizer, device, train: bool):
    model.train(train)
    ce = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for batch in loader:
            poses = batch['poses'].to(device)
            ex_id = batch['exercise_id'].to(device)
            logits, _, _ = model(poses, ex_id)
            loss = ce(logits, ex_id)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(ex_id)
            correct += (logits.argmax(1) == ex_id).sum().item()
            total += len(ex_id)
    return total_loss / total, correct / total


def _run_epoch_vc(model, loader, optimizer, device, train: bool):
    model.train(train)
    ce = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for batch in loader:
            poses    = batch['poses'].to(device)
            ex_id    = batch['exercise_id'].to(device)
            validity = batch['validity'].to(device)
            _, v_logits, _ = model(poses, ex_id)
            loss = ce(v_logits, validity)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(validity)
            correct += (v_logits.argmax(1) == validity).sum().item()
            total += len(validity)
    return total_loss / total, correct / total


def _run_epoch_reg(model, loader, optimizer, device, train: bool):
    model.train(train)
    total_loss, total = 0.0, 0
    with torch.set_grad_enabled(train):
        for batch in loader:
            poses = batch['poses'].to(device)
            ex_id = batch['exercise_id'].to(device)
            qs    = batch['quality_score'].to(device)
            _, _, pred = model(poses, ex_id)
            loss = F.l1_loss(pred, qs)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(qs)
            total += len(qs)
    return total_loss / total


# ── Evaluation ─────────────────────────────────────────────────────────────

def evaluate_quality(model, loader, device, n_exercises: int = 10):
    model.eval()
    all_pred, all_gt, all_ex = [], [], []
    with torch.no_grad():
        for batch in loader:
            poses = batch['poses'].to(device)
            ex_id = batch['exercise_id'].to(device)
            _, _, pred = model(poses, ex_id)
            all_pred.append(pred.squeeze(1).cpu().numpy())
            all_gt.append(batch['quality_score'].squeeze(1).numpy())
            all_ex.append(batch['exercise_id'].numpy())

    pred  = np.concatenate(all_pred)
    gt    = np.concatenate(all_gt)
    ex_id = np.concatenate(all_ex)

    rows = []
    for e in range(n_exercises):
        mask = (ex_id == e)
        if mask.sum() == 0:
            rows.append((e, float('nan'), float('nan'), float('nan')))
            continue
        p, g = pred[mask], gt[mask]
        mad  = float(np.mean(np.abs(p - g)))
        rmse = float(np.sqrt(np.mean((p - g) ** 2)))
        safe_g = np.where(np.abs(g) > 1e-8, g, np.nan)
        mape = float(np.nanmean(np.abs((p - g) / safe_g) * 100))
        rows.append((e, mad, rmse, mape))

    overall_mad  = float(np.mean(np.abs(pred - gt)))
    overall_rmse = float(np.sqrt(np.mean((pred - gt) ** 2)))
    safe_gt = np.where(np.abs(gt) > 1e-8, gt, np.nan)
    overall_mape = float(np.nanmean(np.abs((pred - gt) / safe_gt) * 100))

    return rows, (overall_mad, overall_rmse, overall_mape)


# ── Main ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="STGCN-Seq quality assessment on UI-PRMD")
    p.add_argument('--train_npz',   default='data/uiprmd_quality_train.npz')
    p.add_argument('--test_npz',    default='data/uiprmd_quality_test.npz')
    p.add_argument('--n_joints',    type=int, default=17,
                   help='Ignored for quality NPZ (ROM_J=12 is always used)')
    p.add_argument('--M',           type=int, default=100, help='frames per sequence')
    p.add_argument('--hidden_dim',  type=int, default=64)
    p.add_argument('--batch_size',  type=int, default=16)
    p.add_argument('--lr',          type=float, default=1e-4)
    p.add_argument('--epochs_ec',   type=int, default=30)
    p.add_argument('--epochs_vc',   type=int, default=30)
    p.add_argument('--epochs_reg',  type=int, default=100)
    p.add_argument('--val_subject', type=int, default=7,
                   help='Subject index held out for validation (subject split only)')
    p.add_argument('--split_mode',  choices=['subject', 'random'], default='subject',
                   help='subject: hold out val_subject; random: 80/10/20 random split')
    p.add_argument('--use_gmm_scores', action='store_true', default=False,
                   help='Replace binary quality_labels with GMM log-likelihood scores')
    p.add_argument('--save_model',  default='results/stgcn_seq_best.pt')
    p.add_argument('--save_csv',    default='results/stgcn_seq_results.csv')
    p.add_argument('--seed',        type=int, default=42)
    p.add_argument('--freeze_backbone_reg', action='store_true',
                   help='Freeze backbone during regression stage')
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"ROM joint count: {ROM_J} (--n_joints flag is ignored for quality NPZ)")

    os.makedirs(os.path.dirname(args.save_model) or '.', exist_ok=True)

    # ── Build datasets ───────────────────────────────────────────────────
    print("\nBuilding datasets …")

    if args.split_mode == 'random':
        print("Split mode: RANDOM (combining train + test NPZ, shuffle seed=42)")

        d_tr = np.load(args.train_npz, allow_pickle=True)
        d_te = np.load(args.test_npz,  allow_pickle=True)

        seqs_all   = np.concatenate([d_tr['sequences'],      d_te['sequences']],      axis=0)
        lens_all   = np.concatenate([d_tr['lengths'],        d_te['lengths']])
        ex_all     = np.concatenate([d_tr['exercise_ids'],   d_te['exercise_ids']])
        qlabel_all = np.concatenate([d_tr['quality_labels'], d_te['quality_labels']])

        if args.use_gmm_scores:
            print("Computing GMM scores on combined data …")
            rom_means   = seqs_all.mean(axis=1)  # (N, 12)
            q_scores_all = _compute_gmm_from_arrays(
                rom_means, ex_all, qlabel_all, print_table=True)
        else:
            q_scores_all = qlabel_all.astype(np.float32)

        all_items = _build_items_from_arrays(
            seqs_all, lens_all, ex_all, q_scores_all,
            args.M, include_invalid=False)

        # Shuffle with fixed seed=42, then split 80% train / 20% test
        rng_split = np.random.default_rng(42)
        shuffled  = list(all_items)
        rng_split.shuffle(shuffled)

        n_total = len(shuffled)
        n_test  = int(0.2 * n_total)
        n_train = n_total - n_test
        n_val   = int(0.1 * n_train)

        test_items  = shuffled[:n_test]
        train_all   = shuffled[n_test:]
        val_items   = train_all[:n_val]
        train_items = train_all[n_val:]

        # VC train: add synthetic invalid sequences from train items
        train_vc_items = list(train_items)
        for (seq, eid, qs, _) in train_items:
            inv_raw  = _build_invalid_seq(seq.copy())
            inv_proc = _resize_sequence(inv_raw, args.M)
            train_vc_items.append((inv_proc, eid, qs, 1))

        ds_train_ec = UIPromdSeqDataset(M=args.M, augment=True,
                                         _prebuilt_items=train_items)
        ds_val      = UIPromdSeqDataset(M=args.M, augment=False,
                                         _prebuilt_items=val_items)
        ds_train_vc = UIPromdSeqDataset(M=args.M, augment=True,
                                         _prebuilt_items=train_vc_items)
        ds_test     = UIPromdSeqDataset(M=args.M, augment=False,
                                         _prebuilt_items=test_items)

    else:  # subject split (original behaviour)
        print(f"Split mode: SUBJECT (val_subject={args.val_subject})")
        ext_tr = None
        ext_te = None

        if args.use_gmm_scores:
            print("Computing GMM scores on train NPZ …")
            ext_tr = compute_gmm_scores(args.train_npz)
            print("Computing GMM scores on test NPZ (silent) …")
            d_te_raw   = np.load(args.test_npz, allow_pickle=True)
            rom_means_te = d_te_raw['sequences'].mean(axis=1)
            ext_te = _compute_gmm_from_arrays(
                rom_means_te,
                d_te_raw['exercise_ids'],
                d_te_raw['quality_labels'],
                print_table=False)

        ds_train_ec = UIPromdSeqDataset(
            args.train_npz, M=args.M, augment=True,
            val_subject=args.val_subject, is_val=False,
            include_invalid=False, seed=args.seed,
            ext_quality_scores=ext_tr)

        ds_val = UIPromdSeqDataset(
            args.train_npz, M=args.M, augment=False,
            val_subject=args.val_subject, is_val=True,
            include_invalid=False, seed=args.seed,
            ext_quality_scores=ext_tr)

        ds_train_vc = UIPromdSeqDataset(
            args.train_npz, M=args.M, augment=True,
            val_subject=args.val_subject, is_val=False,
            include_invalid=True, seed=args.seed,
            ext_quality_scores=ext_tr)

        ds_test = UIPromdSeqDataset(
            args.test_npz, M=args.M, augment=False,
            include_invalid=False, seed=args.seed,
            ext_quality_scores=ext_te)

    print(f"  Train (EC/QR): {len(ds_train_ec)} sequences")
    print(f"  Train (VC):    {len(ds_train_vc)} sequences  "
          f"(~{len(ds_train_vc)//2} valid + ~{len(ds_train_vc)//2} invalid)")
    print(f"  Val:           {len(ds_val)} sequences")
    print(f"  Test:          {len(ds_test)} sequences")

    kw = dict(batch_size=args.batch_size, num_workers=0, pin_memory=False)
    dl_train_ec = DataLoader(ds_train_ec, shuffle=True,  **kw)
    dl_train_vc = DataLoader(ds_train_vc, shuffle=True,  **kw)
    dl_val      = DataLoader(ds_val,      shuffle=False, **kw)
    dl_test     = DataLoader(ds_test,     shuffle=False, **kw)

    # ── Model ────────────────────────────────────────────────────────────
    model = STGCNSeq(J=ROM_J, in_dim=1, hidden_dim=args.hidden_dim,
                     M=args.M, edges=ROM_EDGES).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")
    print(f"  Backbone:           "
          f"{sum(p.numel() for p in model.backbone.parameters()):,}")
    print(f"  SpatialGCN:         "
          f"{sum(p.numel() for p in model.backbone.spatial_gcn.parameters()):,}")
    print(f"  TemporalGCN:        "
          f"{sum(p.numel() for p in model.backbone.temporal_gcn.parameters()):,}")
    print(f"  ExerciseClassifier: "
          f"{sum(p.numel() for p in model.exercise_clf.parameters()):,}")
    print(f"  ValidityClassifier: "
          f"{sum(p.numel() for p in model.validity_clf.parameters()):,}")
    print(f"  QualityHead:        "
          f"{sum(p.numel() for p in model.quality_head.parameters()):,}")

    # ── Stage 1: Exercise Classifier ─────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Stage 1 — Exercise Classifier ({args.epochs_ec} epochs)")
    print(f"{'='*60}")

    opt_ec      = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_ec_acc = 0.0
    for epoch in range(args.epochs_ec):
        tr_loss, tr_acc = _run_epoch_ec(model, dl_train_ec, opt_ec, device, train=True)
        va_loss, va_acc = _run_epoch_ec(model, dl_val,      opt_ec, device, train=False)
        if (epoch + 1) % max(1, args.epochs_ec // 5) == 0 or epoch == 0:
            print(f"  EC [{epoch+1:3d}/{args.epochs_ec}] "
                  f"train loss={tr_loss:.4f} acc={tr_acc*100:.1f}%  "
                  f"val loss={va_loss:.4f} acc={va_acc*100:.1f}%")
        if va_acc > best_ec_acc:
            best_ec_acc = va_acc

    print(f"  Best val EC accuracy: {best_ec_acc*100:.2f}%")

    # ── Stage 2: Validity Classifier ─────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Stage 2 — Validity Classifier ({args.epochs_vc} epochs)")
    print(f"{'='*60}")

    opt_vc      = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_vc_acc = 0.0
    for epoch in range(args.epochs_vc):
        tr_loss, tr_acc = _run_epoch_vc(model, dl_train_vc, opt_vc, device, train=True)
        va_loss, va_acc = _run_epoch_vc(model, dl_val,      opt_vc, device, train=False)
        if (epoch + 1) % max(1, args.epochs_vc // 5) == 0 or epoch == 0:
            print(f"  VC [{epoch+1:3d}/{args.epochs_vc}] "
                  f"train loss={tr_loss:.4f} acc={tr_acc*100:.1f}%  "
                  f"val loss={va_loss:.4f} acc={va_acc*100:.1f}%")
        if va_acc > best_vc_acc:
            best_vc_acc = va_acc

    print(f"  Best val VC accuracy: {best_vc_acc*100:.2f}%")

    # ── Stage 3: Quality Regression ───────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Stage 3 — Quality Regression ({args.epochs_reg} epochs)")
    print(f"{'='*60}")

    if args.freeze_backbone_reg:
        for p in model.backbone.parameters():
            p.requires_grad_(False)
        for p in model.exercise_clf.parameters():
            p.requires_grad_(False)
        for p in model.validity_clf.parameters():
            p.requires_grad_(False)
        opt_reg = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    else:
        opt_reg = torch.optim.Adam([
            {'params': model.backbone.parameters(),      'lr': args.lr * 0.1},
            {'params': model.exercise_clf.parameters(), 'lr': args.lr * 0.1},
            {'params': model.validity_clf.parameters(), 'lr': args.lr * 0.1},
            {'params': model.quality_head.parameters(), 'lr': args.lr},
        ])

    best_val_mad = float('inf')
    for epoch in range(args.epochs_reg):
        tr_loss = _run_epoch_reg(model, dl_train_ec, opt_reg, device, train=True)
        va_loss = _run_epoch_reg(model, dl_val,      opt_reg, device, train=False)
        if (epoch + 1) % max(1, args.epochs_reg // 10) == 0 or epoch == 0:
            print(f"  QR [{epoch+1:3d}/{args.epochs_reg}] "
                  f"train L1={tr_loss:.4f}  val L1={va_loss:.4f}")
        if va_loss < best_val_mad:
            best_val_mad = va_loss
            torch.save({'model_state_dict': model.state_dict(),
                        'args': vars(args),
                        'best_val_mad': best_val_mad},
                       args.save_model)

    print(f"  Best val MAD: {best_val_mad:.4f}")
    print(f"  Model saved -> {args.save_model}")

    # Reload best model for evaluation
    ckpt = torch.load(args.save_model, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])

    # ── Final evaluation ──────────────────────────────────────────────────
    _, ec_acc_test = _run_epoch_ec(model, dl_test, None, device, train=False)

    if args.split_mode == 'random':
        test_vc_items = list(test_items)
        for (seq, eid, qs, _) in test_items:
            inv_raw  = _build_invalid_seq(seq.copy())
            inv_proc = _resize_sequence(inv_raw, args.M)
            test_vc_items.append((inv_proc, eid, qs, 1))
        ds_test_vc = UIPromdSeqDataset(M=args.M, augment=False,
                                        _prebuilt_items=test_vc_items)
    else:
        ds_test_vc = UIPromdSeqDataset(
            args.test_npz, M=args.M, augment=False,
            include_invalid=True, seed=args.seed,
            ext_quality_scores=ext_te)

    dl_test_vc = DataLoader(ds_test_vc, shuffle=False, **kw)
    _, vc_acc_test = _run_epoch_vc(model, dl_test_vc, None, device, train=False)

    rows, overall = evaluate_quality(model, dl_test, device)

    # ── Print results ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("=== Exercise Classifier ===")
    print(f"  Accuracy: {ec_acc_test*100:.2f}%")
    print("  (compare: paper reports 0.96 on KIMORE)")

    print("\n=== Validity Classifier ===")
    print(f"  Accuracy: {vc_acc_test*100:.2f}%")
    print("  (compare: paper reports 0.93 on KIMORE)")

    print("\n=== Quality Score Regression (UI-PRMD) ===")
    header = f"{'Exercise':<12} {'MAD':>7} {'RMSE':>7} {'MAPE':>8}  {'Paper MAD':>9}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for (e, mad, rmse, mape), paper_mad in zip(rows, PAPER_MAD):
        name = f"Ex{e+1:02d} ({EXERCISE_NAMES[e]})"
        print(f"{name:<12} {mad:7.3f} {rmse:7.3f} {mape:7.2f}%  {paper_mad:9.3f}")
    print("-" * len(header))
    omad, ormse, omape = overall
    print(f"{'AVERAGE':<12} {omad:7.3f} {ormse:7.3f} {omape:7.2f}%  {'0.009':>9}")
    print("-" * len(header))

    # ── Save CSV ──────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.save_csv) or '.', exist_ok=True)
    with open(args.save_csv, 'w') as f:
        f.write("exercise,exercise_name,MAD,RMSE,MAPE,paper_MAD\n")
        for (e, mad, rmse, mape), paper_mad in zip(rows, PAPER_MAD):
            f.write(f"Ex{e+1:02d},{EXERCISE_NAMES[e]},{mad:.6f},{rmse:.6f},"
                    f"{mape:.4f},{paper_mad}\n")
        f.write(f"AVERAGE,ALL,{omad:.6f},{ormse:.6f},{omape:.4f},0.009\n")
    print(f"\nResults saved -> {args.save_csv}")


if __name__ == '__main__':
    main()
