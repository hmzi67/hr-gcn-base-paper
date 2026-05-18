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
"""

import argparse
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import interp1d

# ── Joint selection ────────────────────────────────────────────────────────
# 19 non-zero joints in the 133-joint COCO-WholeBody format that come from
# Vicon data.  We keep 17 by dropping small-toe proxies (indices 18, 21).
ACTIVE_JOINTS = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 22]
# Local indices (0-16) after selection:
#  0:head  1:l_sho  2:r_sho  3:l_elb  4:r_elb  5:l_wri  6:r_wri
#  7:l_hip  8:r_hip  9:l_kne  10:r_kne  11:l_ank  12:r_ank
#  13:l_bigtoe  14:l_heel  15:r_bigtoe  16:r_heel

SKELETON_EDGES = [
    (0, 1), (0, 2),         # head – shoulders
    (1, 2),                 # shoulder bar
    (1, 3), (3, 5),         # left arm
    (2, 4), (4, 6),         # right arm
    (1, 7), (2, 8),         # torso sides
    (7, 8),                 # hip bar
    (7, 9), (9, 11),        # left leg
    (11, 13), (11, 14),     # left foot
    (8, 10), (10, 12),      # right leg
    (12, 15), (12, 16),     # right foot
]

EXERCISE_NAMES = [
    "DS", "HS", "IL", "SL", "SS", "SLR", "SA", "SE", "SIR", "SS2"
]

PAPER_MAD = [0.006, 0.008, 0.009, 0.006, 0.003, 0.004, 0.009, 0.013, 0.006, 0.028]


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
    Per-exercise learnable adjacency GCN over joints.
    Input:  (B, J, in_dim)
    Output: (B, J, hidden_dim)
    """
    def __init__(self, in_dim: int, hidden_dim: int, J: int,
                 n_exercises: int = 10, n_layers: int = 2):
        super().__init__()
        self.n_exercises = n_exercises
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        A_init = torch.FloatTensor(build_adjacency(J, SKELETON_EDGES))
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
                A = torch.softmax(self.A_spatial[eid], dim=-1)  # (J, J)
                h = out[mask]                                    # (n, J, C)
                h = torch.einsum('ij,njc->nic', A, h)           # (n, J, C)
                h = self.W[k](h)                                # (n, J, hidden)
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
        A = torch.softmax(self.A_temporal, dim=-1)  # (M, M)
        for k in range(self.n_layers):
            out = torch.einsum('ij,bjc->bic', A, out)  # (B, M, C)
            out = self.W[k](out)                        # (B, M, hidden)
            out = F.relu(self.bn[k](out))
        out = out.mean(dim=1)   # (B, hidden_dim)
        return out


class STGCNBackbone(nn.Module):
    def __init__(self, J: int = 17, hidden_dim: int = 64, M: int = 100,
                 n_exercises: int = 10):
        super().__init__()
        self.J = J
        self.hidden_dim = hidden_dim
        self.M = M

        self.spatial_gcn = SpatialGCN(3, hidden_dim, J, n_exercises)
        self.temporal_gcn = TemporalGCN(J * hidden_dim, hidden_dim, M)

    def forward(self, x: torch.Tensor, exercise_id: torch.Tensor) -> torch.Tensor:
        # x: (B, M, J, 3)
        B, M, J, C = x.shape
        x_flat = x.reshape(B * M, J, C)
        ex_rep = exercise_id.unsqueeze(1).expand(B, M).reshape(B * M)
        sp_out = self.spatial_gcn(x_flat, ex_rep)                    # (B*M, J, hidden)
        tmp_in = sp_out.reshape(B, M, J * self.hidden_dim)
        feat = self.temporal_gcn(tmp_in)                             # (B, hidden)
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
        return self.fc(feat)  # (B, 1)


class STGCNSeq(nn.Module):
    def __init__(self, J: int = 17, hidden_dim: int = 64, M: int = 100,
                 n_exercises: int = 10):
        super().__init__()
        self.backbone = STGCNBackbone(J, hidden_dim, M, n_exercises)
        self.exercise_clf = ExerciseClassifier(hidden_dim, n_exercises)
        self.validity_clf = ValidityClassifier(hidden_dim)
        self.quality_head = QualityHead(hidden_dim)

    def forward(self, x: torch.Tensor, exercise_id: torch.Tensor):
        feat = self.backbone(x, exercise_id)
        return (
            self.exercise_clf(feat),   # (B, 10)
            self.validity_clf(feat),   # (B, 2)
            self.quality_head(feat),   # (B, 1)
        )


# ── Dataset ────────────────────────────────────────────────────────────────

def _resize_sequence(poses: np.ndarray, target_len: int) -> np.ndarray:
    T, J, C = poses.shape
    if T == target_len:
        return poses
    x_old = np.linspace(0.0, 1.0, T)
    x_new = np.linspace(0.0, 1.0, target_len)
    out = np.empty((target_len, J, C), dtype=poses.dtype)
    for j in range(J):
        for c in range(C):
            out[:, j, c] = np.interp(x_new, x_old, poses[:, j, c])
    return out


def _augment(poses: np.ndarray, M: int) -> np.ndarray:
    T, J, C = poses.shape
    # Speed augmentation
    L = np.random.randint(0, max(1, int(0.25 * T)) + 1)
    if L > 0:
        if np.random.random() < 0.5:
            extra_idx = np.random.choice(T, L, replace=False)
            poses = np.concatenate([poses, poses[extra_idx]], axis=0)
        else:
            keep = np.sort(np.random.choice(T, max(2, T - L), replace=False))
            poses = poses[keep]
    poses = _resize_sequence(poses, M)
    # Rotation around vertical (Z) axis
    angle = np.random.uniform(-15.0, 15.0) * math.pi / 180.0
    ca, sa = math.cos(angle), math.sin(angle)
    R = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]], dtype=np.float32)
    poses = poses.reshape(-1, 3) @ R.T
    return poses.reshape(M, J, C)


def _build_invalid(poses: np.ndarray) -> np.ndarray:
    T = len(poses)
    p = np.random.uniform(0.25, 0.75)
    crop_len = max(2, int(p * T))
    start = np.random.randint(0, max(1, T - crop_len))
    return poses[start: start + crop_len]


class UIPromdSeqDataset(Dataset):
    """
    Builds one sequence per (subject, exercise, quality_label) triplet.
    Optionally appends cropped (invalid) sequences for VC training.
    """

    def __init__(self, npz_path: str, M: int = 100,
                 active_joints=None,
                 augment: bool = False,
                 val_subject: int = None,
                 is_val: bool = False,
                 include_invalid: bool = False,
                 seed: int = 42):
        if active_joints is None:
            active_joints = ACTIVE_JOINTS
        self.M = M
        self.augment = augment
        self.J = len(active_joints)

        rng = np.random.default_rng(seed)

        d = np.load(npz_path, allow_pickle=True)
        poses_3d    = d['poses_3d'][:, active_joints, :]   # (N, J, 3)
        subject_ids = d['subject_ids']
        exercise_ids = d['exercise_ids']
        quality_labels = d['quality_labels']
        quality_scores = d['quality_scores']

        # Subject split
        if val_subject is not None:
            keep = (subject_ids == val_subject) if is_val else (subject_ids != val_subject)
            poses_3d    = poses_3d[keep]
            subject_ids = subject_ids[keep]
            exercise_ids = exercise_ids[keep]
            quality_labels = quality_labels[keep]
            quality_scores = quality_scores[keep]

        # Local indices of hip joints (for first-frame centering)
        hip_l = active_joints.index(11) if 11 in active_joints else 0
        hip_r = active_joints.index(12) if 12 in active_joints else 0

        self._items = []  # list of (poses_M_J_3, exercise_id, quality_score, validity_label)

        combos = sorted(set(zip(subject_ids.tolist(),
                               exercise_ids.tolist(),
                               quality_labels.tolist())))
        for (sid, eid, qlabel) in combos:
            mask = (subject_ids == sid) & (exercise_ids == eid) & (quality_labels == qlabel)
            raw = poses_3d[mask].copy()          # (T, J, 3)
            qs  = float(quality_scores[mask][0]) # binary 0.0 or 1.0

            # Normalize: subtract first-frame hip midpoint
            spine = (raw[0, hip_l] + raw[0, hip_r]) / 2.0
            raw -= spine[np.newaxis, np.newaxis, :]

            proc = _resize_sequence(raw, M).astype(np.float32)
            self._items.append((proc, eid, qs, 0))  # 0 = valid

            if include_invalid:
                inv_raw = _build_invalid(raw)
                inv_proc = _resize_sequence(inv_raw, M).astype(np.float32)
                self._items.append((inv_proc, eid, qs, 1))  # 1 = invalid

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        poses, eid, qs, validity = self._items[idx]
        if self.augment:
            poses = _augment(poses.copy(), self.M)
        return {
            'poses':         torch.from_numpy(poses),
            'exercise_id':   torch.tensor(eid,      dtype=torch.long),
            'quality_score': torch.tensor([qs],     dtype=torch.float32),
            'validity':      torch.tensor(validity, dtype=torch.long),
        }


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
    p.add_argument('--train_npz',  default='data/uiprmd_quality_train.npz')
    p.add_argument('--test_npz',   default='data/uiprmd_quality_test.npz')
    p.add_argument('--n_joints',   type=int, default=17)
    p.add_argument('--M',          type=int, default=100, help='frames per sequence')
    p.add_argument('--hidden_dim', type=int, default=64)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr',         type=float, default=1e-4)
    p.add_argument('--epochs_ec',  type=int, default=30)
    p.add_argument('--epochs_vc',  type=int, default=30)
    p.add_argument('--epochs_reg', type=int, default=100)
    p.add_argument('--val_subject', type=int, default=7,
                   help='Subject index held out for validation from train NPZ')
    p.add_argument('--save_model', default='results/stgcn_seq_best.pt')
    p.add_argument('--save_csv',   default='results/stgcn_seq_results.csv')
    p.add_argument('--seed',       type=int, default=42)
    p.add_argument('--freeze_backbone_reg', action='store_true',
                   help='Freeze backbone during regression stage')
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    os.makedirs(os.path.dirname(args.save_model) or '.', exist_ok=True)

    # ── Joint selection ──────────────────────────────────────────────────
    active_joints = ACTIVE_JOINTS[:args.n_joints]
    J = len(active_joints)
    print(f"Using {J} joints: {active_joints}")

    # ── Build datasets ───────────────────────────────────────────────────
    print("\nBuilding datasets …")

    ds_train_ec = UIPromdSeqDataset(
        args.train_npz, M=args.M, active_joints=active_joints,
        augment=True,  val_subject=args.val_subject, is_val=False,
        include_invalid=False, seed=args.seed)

    ds_val = UIPromdSeqDataset(
        args.train_npz, M=args.M, active_joints=active_joints,
        augment=False, val_subject=args.val_subject, is_val=True,
        include_invalid=False, seed=args.seed)

    ds_train_vc = UIPromdSeqDataset(
        args.train_npz, M=args.M, active_joints=active_joints,
        augment=True,  val_subject=args.val_subject, is_val=False,
        include_invalid=True, seed=args.seed)

    ds_test = UIPromdSeqDataset(
        args.test_npz, M=args.M, active_joints=active_joints,
        augment=False, include_invalid=False, seed=args.seed)

    print(f"  Train (EC/QR): {len(ds_train_ec)} sequences")
    print(f"  Train (VC):    {len(ds_train_vc)} sequences  "
          f"(≈{len(ds_train_vc)//2} valid + {len(ds_train_vc)//2} invalid)")
    print(f"  Val:           {len(ds_val)} sequences")
    print(f"  Test:          {len(ds_test)} sequences")

    kw = dict(batch_size=args.batch_size, num_workers=0, pin_memory=False)
    dl_train_ec = DataLoader(ds_train_ec, shuffle=True,  **kw)
    dl_train_vc = DataLoader(ds_train_vc, shuffle=True,  **kw)
    dl_val      = DataLoader(ds_val,      shuffle=False, **kw)
    dl_test     = DataLoader(ds_test,     shuffle=False, **kw)

    # ── Model ────────────────────────────────────────────────────────────
    model = STGCNSeq(J=J, hidden_dim=args.hidden_dim, M=args.M).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")
    print(f"  Backbone:         "
          f"{sum(p.numel() for p in model.backbone.parameters()):,}")
    print(f"  SpatialGCN:       "
          f"{sum(p.numel() for p in model.backbone.spatial_gcn.parameters()):,}")
    print(f"  TemporalGCN:      "
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

    opt_ec = torch.optim.Adam(model.parameters(), lr=args.lr)
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

    opt_vc = torch.optim.Adam(model.parameters(), lr=args.lr)
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
        # Lower LR for backbone, full LR for quality head
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
    print(f"  Model saved → {args.save_model}")

    # Reload best model for evaluation
    ckpt = torch.load(args.save_model, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])

    # ── Final evaluation ──────────────────────────────────────────────────
    # EC accuracy on test
    _, ec_acc_test = _run_epoch_ec(model, dl_test, None, device, train=False)
    # VC accuracy on test (create a test dataset with invalid seqs)
    ds_test_vc = UIPromdSeqDataset(
        args.test_npz, M=args.M, active_joints=active_joints,
        augment=False, include_invalid=True, seed=args.seed)
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
    print("─" * len(header))
    print(header)
    print("─" * len(header))
    for (e, mad, rmse, mape), paper_mad in zip(rows, PAPER_MAD):
        name = f"Ex{e+1:02d} ({EXERCISE_NAMES[e]})"
        print(f"{name:<12} {mad:7.3f} {rmse:7.3f} {mape:7.2f}%  {paper_mad:9.3f}")
    print("─" * len(header))
    omad, ormse, omape = overall
    print(f"{'AVERAGE':<12} {omad:7.3f} {ormse:7.3f} {omape:7.2f}%  {'0.009':>9}")
    print("─" * len(header))

    # ── Save CSV ──────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.save_csv) or '.', exist_ok=True)
    with open(args.save_csv, 'w') as f:
        f.write("exercise,exercise_name,MAD,RMSE,MAPE,paper_MAD\n")
        for (e, mad, rmse, mape), paper_mad in zip(rows, PAPER_MAD):
            f.write(f"Ex{e+1:02d},{EXERCISE_NAMES[e]},{mad:.6f},{rmse:.6f},"
                    f"{mape:.4f},{paper_mad}\n")
        f.write(f"AVERAGE,ALL,{omad:.6f},{ormse:.6f},{omape:.4f},0.009\n")
    print(f"\nResults saved → {args.save_csv}")


if __name__ == '__main__':
    main()
