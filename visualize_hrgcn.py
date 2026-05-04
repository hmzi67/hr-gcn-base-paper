"""
HR-GCN 3D Whole-Body Pose Visualization
Uses: authors checkpoint (ckpt_best.bin) + h3wb_test.npz
No MMPose required.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
import glob

# ── Paths — apne system ke mutabiq change karo ──────────────────────────────
TEST_NPZ    = '/home/genesys/hamza/HR-GCN/data/h3wb_test.npz'
HR_GCN_DIR  = '/home/genesys/hamza/HR-GCN'
CFG_FILE    = '/home/genesys/hamza/HR-GCN/w32_adam_lr1e-3.yaml'
BASELINE_CHECKPOINT = os.path.join(
    HR_GCN_DIR, 'checkpoint', 'HRGCN', 'dc_preagg-2026-04-24T10:48:02', 'ckpt_best.pth.tar'
)


def find_default_checkpoint():
    # Prefer the requested baseline checkpoint when it exists.
    if os.path.isfile(BASELINE_CHECKPOINT):
        return BASELINE_CHECKPOINT

    # Prefer checkpoints produced in this workspace (compatible with current code).
    local_candidates = glob.glob(os.path.join(HR_GCN_DIR, 'checkpoint', 'HRGCN', '*', 'ckpt_best.pth.tar'))
    if local_candidates:
        return max(local_candidates, key=os.path.getmtime)

    # Fallback to external/downloaded checkpoint path if no local checkpoint exists.
    return '/home/genesys/Downloads/checkpoint/ckpt_best.bin'


CHECKPOINT = find_default_checkpoint()
# ─────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, HR_GCN_DIR)

# ── Joint partition (133 total) ───────────────────────────────────────────────
BODY_IDX       = list(range(0,  23))   # 23 joints
FACE_IDX       = list(range(23, 91))   # 68 joints
LEFT_HAND_IDX  = list(range(91, 112))  # 21 joints
RIGHT_HAND_IDX = list(range(112, 133)) # 21 joints

# ── Body skeleton edges for first 23 joints (from dataset parent definition) ──
BODY_PARENTS_23 = [
    -1, 0, 0, 0, 0, 0, 0, 5, 6, 7, 8, 5, 6, 11, 12, 13, 14, 15, 15, 15, 16, 16, 16
]
BODY_EDGES = [(p, i) for i, p in enumerate(BODY_PARENTS_23) if p >= 0]

HAND_EDGES = [
    (0,1),(1,2),(2,3),(3,4),           # thumb
    (0,5),(5,6),(6,7),(7,8),           # index
    (0,9),(9,10),(10,11),(11,12),      # middle
    (0,13),(13,14),(14,15),(15,16),    # ring
    (0,17),(17,18),(18,19),(19,20),    # pinky
    (5,9),(9,13),(13,17),              # palm
]

# ─────────────────────────────────────────────────────────────────────────────
def load_model(ckpt_path, cfg_path, allow_unexpected=False):
    from lib.config import cfg
    import models.graph_hrnet_multi_branch as ghrmb
    from utils.prepare_data_h3wb import Human3WBDataset
    from common.graph_utils import adj_mx_from_skeleton

    cfg.merge_from_file(cfg_path)

    train_npz = os.path.join(HR_GCN_DIR, 'data', 'h3wb_train.npz')
    test_npz = os.path.join(HR_GCN_DIR, 'data', 'h3wb_test.npz')
    dataset = Human3WBDataset(train_npz, test_npz)
    skeleton = dataset.skeleton()
    adj = adj_mx_from_skeleton(skeleton)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = ghrmb.get_pose_net(cfg, True, adj.to(device), None, 'dc_preagg', skeleton.joints_group()).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
    load_info = model.load_state_dict(state_dict, strict=False)
    if load_info.missing_keys:
        raise RuntimeError(
            'Checkpoint/model mismatch. '
            f'Missing keys: {len(load_info.missing_keys)}, '
            f'Unexpected keys: {len(load_info.unexpected_keys)}. '
            f'Sample missing: {load_info.missing_keys[:5]} | '
            f'Sample unexpected: {load_info.unexpected_keys[:5]}\n'
            'Use a checkpoint produced by this code version, or pass a matching --checkpoint/--cfg pair.'
        )
    if load_info.unexpected_keys and not allow_unexpected:
        raise RuntimeError(
            'Checkpoint/model mismatch. '
            f'Missing keys: {len(load_info.missing_keys)}, '
            f'Unexpected keys: {len(load_info.unexpected_keys)}. '
            f'Sample missing: {load_info.missing_keys[:5]} | '
            f'Sample unexpected: {load_info.unexpected_keys[:5]}\n'
            'Use a checkpoint produced by this code version, or pass a matching --checkpoint/--cfg pair.'
        )
    if load_info.unexpected_keys and allow_unexpected:
        print(
            'WARNING: Ignoring unexpected checkpoint keys: '
            f'{len(load_info.unexpected_keys)} (sample: {load_info.unexpected_keys[:5]})'
        )
    model.eval()
    epoch = ckpt.get('epoch', 'N/A') if isinstance(ckpt, dict) else 'N/A'
    error = ckpt.get('error', None) if isinstance(ckpt, dict) else None
    if error is not None:
        print(f"✅ Model loaded — Epoch {epoch} | MPJPE {error:.2f} mm")
    else:
        print(f"✅ Model loaded — Epoch {epoch}")
    return model


def load_sample(npz_path, sample_idx=0):
    """Load one sample using the same preprocessing as HRNet_GCN_WB evaluation."""
    from utils.prepare_data_h3wb import Human3WBDataset, TEST_SUBJECTS
    from common.data_utils import read_3d_data, create_2d_data, fetch

    train_npz = os.path.join(HR_GCN_DIR, 'data', 'h3wb_train.npz')
    test_npz = npz_path

    dataset = Human3WBDataset(train_npz, test_npz)
    dataset = read_3d_data(dataset)
    keypoints = create_2d_data(dataset)

    valid_2d = []
    valid_3d = []
    for action in dataset.define_actions():
        poses_3d, poses_2d, _, _ = fetch(TEST_SUBJECTS, dataset, keypoints, [action], 1)
        if poses_3d is None or len(poses_3d) == 0:
            continue
        valid_2d.extend(poses_2d)
        valid_3d.extend(poses_3d)

    if not valid_2d or not valid_3d:
        raise ValueError('No valid paired 2D/3D samples found in test split')

    all_2d = np.concatenate(valid_2d, axis=0)  # normalized 2D, root-centered
    all_3d = np.concatenate(valid_3d, axis=0)  # meters, root-centered

    idx = min(sample_idx, len(all_2d) - 1)
    print(f"✅ Loaded sample {idx}/{len(all_2d)} from preprocessed test set")
    return all_2d[idx], all_3d[idx]


def normalize_2d(pose2d):
    """Simple normalization — center + scale"""
    pose = pose2d.copy().astype(np.float32)
    center = pose.mean(axis=0)
    pose -= center
    scale = np.abs(pose).max()
    if scale > 0:
        pose /= scale
    return pose


def assemble_pose3d(body, face, lhand, rhand):
    # Shared assembly for baseline/custom predictions.
    pose3d = np.zeros((133, 3), dtype=np.float32)
    pose3d[BODY_IDX]       = body
    pose3d[FACE_IDX]       = face
    pose3d[LEFT_HAND_IDX]  = lhand
    pose3d[RIGHT_HAND_IDX] = rhand
    return pose3d


def predict(model, pose2d_raw):
    pose2d = pose2d_raw.astype(np.float32)
    device = next(model.parameters()).device
    inp = torch.FloatTensor(pose2d).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(inp)

    body  = out[0].squeeze(0).detach().cpu().numpy()   # (23, 3)
    face  = out[1].squeeze(0).detach().cpu().numpy()   # (68, 3)
    lhand = out[2].squeeze(0).detach().cpu().numpy()   # (21, 3)
    rhand = out[3].squeeze(0).detach().cpu().numpy()   # (21, 3)

    # Sab directly pelvis-relative hain — koi anchor nahi
    return assemble_pose3d(body, face, lhand, rhand)


def predict_custom(model, pose2d_raw):
    # Custom forward placeholder; replace with your model's output mapping.
    return predict(model, pose2d_raw)


def center_on_hip(pose3d):
    # Match evaluation centering: subtract hip midpoint if available.
    centered = pose3d.copy().astype(np.float32)
    if centered.shape[0] > 12:
        centered -= (centered[11:12, :] + centered[12:13, :]) / 2.0
    else:
        centered -= centered[0:1]
    return centered


def compute_mpjpe(pred3d, gt3d):
    # Mean per-joint position error in millimeters (prediction centered as in eval).
    pred_centered = center_on_hip(pred3d)
    return np.mean(np.linalg.norm(pred_centered - gt3d, axis=-1)) * 1000.0


def center_for_display(pose3d):
    # Keep relative geometry but center around hip midpoint (same as eval code).
    return center_on_hip(pose3d)


def compute_shared_limits(poses):
    valid = [p for p in poses if p is not None]
    if not valid:
        return None
    stacked = np.concatenate(valid, axis=0)
    mins = stacked.min(axis=0)
    maxs = stacked.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max((maxs - mins).max() / 2.0, 1e-3)
    return (center, radius)


def set_equal_axes(ax, pose3d, shared=None):
    if shared is None:
        shared = compute_shared_limits([pose3d])
    center, radius = shared
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def plot_pose(pred3d, gt3d=None, custom3d=None, title="HR-GCN 3D Whole-Body Pose",
              save_path='pose_visualization.png'):
    pred_plot = center_for_display(pred3d)
    custom_plot = center_for_display(custom3d) if custom3d is not None else None
    gt_plot = center_for_display(gt3d) if gt3d is not None else None

    # Shared limits keep all panels on the same axis scale.
    shared_limits = compute_shared_limits([pred_plot, custom_plot, gt_plot])

    if custom_plot is not None and gt_plot is not None:
        cols = 3
    elif custom_plot is not None or gt_plot is not None:
        cols = 2
    else:
        cols = 1

    fig = plt.figure(figsize=(8 * cols, 8))
    fig.patch.set_facecolor('#1a1a2e')

    def draw_one(ax, pose3d, label):
        ax.set_facecolor('#16213e')
        ax.set_title(label, color='white', fontsize=13, pad=10)

        # Body — green
        b = pose3d[BODY_IDX]
        ax.scatter(b[:,0], b[:,1], b[:,2], c='#00ff88', s=36, zorder=5)
        for i, j in BODY_EDGES:
            if i < len(b) and j < len(b):
                ax.plot([b[i,0], b[j,0]], [b[i,1], b[j,1]],
                        [b[i,2], b[j,2]], c='#00ff88', lw=2.2, alpha=0.95)

        # Face — cyan dots only (too many for edges)
        # Face — larger dots + basic contour edges
        f = pose3d[FACE_IDX]
        ax.scatter(f[:,0], f[:,1], f[:,2], c='#00ccff', s=25, alpha=0.9, zorder=6)

        # Basic face contour edges (COCO-WholeBody face layout)
        face_contour = list(zip(range(0,16), range(1,17)))      # jaw line
        left_brow    = list(zip(range(17,21), range(18,22)))    # left eyebrow
        right_brow   = list(zip(range(22,26), range(23,27)))    # right eyebrow
        nose_bridge  = list(zip(range(27,30), range(28,31)))    # nose
        left_eye     = list(zip(range(36,41), range(37,42))) + [(41,36)]
        right_eye    = list(zip(range(42,47), range(43,48))) + [(47,42)]
        mouth_outer  = list(zip(range(48,59), range(49,60))) + [(59,48)]

        for edges in [face_contour, left_brow, right_brow,
                    nose_bridge, left_eye, right_eye, mouth_outer]:
            for i, j in edges:
                if i < len(f) and j < len(f):
                    ax.plot([f[i,0], f[j,0]], [f[i,1], f[j,1]],
                            [f[i,2], f[j,2]], c='#00ccff', lw=0.8, alpha=0.7)
        # Left hand — orange
        lh = pose3d[LEFT_HAND_IDX]
        ax.scatter(lh[:,0], lh[:,1], lh[:,2], c='#ff8800', s=22, zorder=5)
        for i, j in HAND_EDGES:
            if i < len(lh) and j < len(lh):
                ax.plot([lh[i,0], lh[j,0]], [lh[i,1], lh[j,1]],
                        [lh[i,2], lh[j,2]], c='#ff8800', lw=1.4, alpha=0.9)

        # Right hand — red
        rh = pose3d[RIGHT_HAND_IDX]
        ax.scatter(rh[:,0], rh[:,1], rh[:,2], c='#ff4466', s=22, zorder=5)
        for i, j in HAND_EDGES:
            if i < len(rh) and j < len(rh):
                ax.plot([rh[i,0], rh[j,0]], [rh[i,1], rh[j,1]],
                        [rh[i,2], rh[j,2]], c='#ff4466', lw=1.4, alpha=0.9)

        ax.tick_params(colors='gray')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.grid(True, color='#333355', linewidth=0.5)
        ax.set_xlabel('X', color='gray')
        ax.set_ylabel('Y', color='gray')
        ax.set_zlabel('Z', color='gray')
        set_equal_axes(ax, pose3d, shared_limits)
        ax.view_init(elev=17, azim=-62)

    ax1 = fig.add_subplot(1, cols, 1, projection='3d')
    base_label = "Baseline Prediction" if custom_plot is not None else "HR-GCN Prediction"
    draw_one(ax1, pred_plot, base_label)

    col_index = 2
    if custom_plot is not None:
        ax_custom = fig.add_subplot(1, cols, col_index, projection='3d')
        draw_one(ax_custom, custom_plot, "Custom Prediction")
        col_index += 1
    if gt_plot is not None:
        ax_gt = fig.add_subplot(1, cols, col_index, projection='3d')
        draw_one(ax_gt, gt_plot, "Ground Truth")

    # Legend
    from matplotlib.lines import Line2D
    legend = [
        Line2D([0],[0], color='#00ff88', lw=2, label='Body'),
        Line2D([0],[0], color='#00ccff', lw=2, label='Face'),
        Line2D([0],[0], color='#ff8800', lw=2, label='Left hand'),
        Line2D([0],[0], color='#ff4466', lw=2, label='Right hand'),
    ]
    fig.legend(handles=legend, loc='lower center', ncol=4,
               facecolor='#1a1a2e', labelcolor='white', fontsize=11,
               framealpha=0.8)

    plt.suptitle(title, color='white', fontsize=15, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150,
                bbox_inches='tight', facecolor='#1a1a2e')
    print(f"✅ Saved: {save_path}")
    plt.show()


def plot_overlay(pred3d, custom3d, title="Baseline vs Custom Overlay",
                 save_path='comparison_pose_overlay.png'):
    # Overlay baseline and custom predictions to highlight deviation.
    pred_plot = center_for_display(pred3d)
    custom_plot = center_for_display(custom3d)
    shared_limits = compute_shared_limits([pred_plot, custom_plot])

    fig = plt.figure(figsize=(8, 8))
    fig.patch.set_facecolor('#1a1a2e')
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_facecolor('#16213e')
    ax.set_title(title, color='white', fontsize=13, pad=10)

    ax.scatter(pred_plot[:, 0], pred_plot[:, 1], pred_plot[:, 2],
               c='#00ff88', s=16, alpha=0.9, zorder=5)
    ax.scatter(custom_plot[:, 0], custom_plot[:, 1], custom_plot[:, 2],
               c='#ff4466', s=16, alpha=0.9, zorder=6)

    # Deviation lines between baseline and custom joints.
    joint_count = min(len(pred_plot), len(custom_plot))
    for i in range(joint_count):
        ax.plot([pred_plot[i, 0], custom_plot[i, 0]],
                [pred_plot[i, 1], custom_plot[i, 1]],
                [pred_plot[i, 2], custom_plot[i, 2]],
                c='#ffffaa', lw=0.6, alpha=0.6)

    ax.tick_params(colors='gray')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, color='#333355', linewidth=0.5)
    ax.set_xlabel('X', color='gray')
    ax.set_ylabel('Y', color='gray')
    ax.set_zlabel('Z', color='gray')
    set_equal_axes(ax, pred_plot, shared_limits)
    ax.view_init(elev=17, azim=-62)

    from matplotlib.lines import Line2D
    legend = [
        Line2D([0], [0], color='#00ff88', lw=2, label='Baseline'),
        Line2D([0], [0], color='#ff4466', lw=2, label='Custom'),
        Line2D([0], [0], color='#ffffaa', lw=2, label='Deviation'),
    ]
    fig.legend(handles=legend, loc='lower center', ncol=3,
               facecolor='#1a1a2e', labelcolor='white', fontsize=11,
               framealpha=0.8)

    plt.suptitle(title, color='white', fontsize=15, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150,
                bbox_inches='tight', facecolor='#1a1a2e')
    print(f"✅ Saved: {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=int, default=100,
                        help='Sample index from test set')
    parser.add_argument('--no_model', action='store_true',
                        help='Skip model — only show ground truth')
    parser.add_argument('--checkpoint', type=str, default=CHECKPOINT,
                        help='Checkpoint path (.pth.tar/.bin)')
    parser.add_argument('--cfg', type=str, default=CFG_FILE,
                        help='YAML config path used to build model')
    parser.add_argument('--test_npz', type=str, default=TEST_NPZ,
                        help='Test npz path')
    parser.add_argument('--compare', action='store_true',
                        help='Compare baseline vs custom predictions')
    parser.add_argument('--overlay', action='store_true',
                        help='Overlay baseline vs custom predictions')
    parser.add_argument('--allow_unexpected', action='store_true',
                        help='Allow unexpected keys when loading checkpoints')
    args = parser.parse_args()

    print("\n==> Loading test sample...")
    pose2d, pose3d_gt = load_sample(args.test_npz, sample_idx=args.sample)

    if args.no_model:
        print("\n==> Showing ground truth only...")
        plot_pose(pose3d_gt, title="Ground Truth 3D Whole-Body Pose")
    else:
        if args.overlay and not args.compare:
            print("\n==> --overlay requires --compare; enabling compare mode.")
            args.compare = True

        print("\n==> Loading baseline model...")
        print(f"Using checkpoint: {args.checkpoint}")
        print(f"Using config: {args.cfg}")
        baseline_model = load_model(
            args.checkpoint, args.cfg, allow_unexpected=args.allow_unexpected
        )

        if args.compare:
            print("\n==> Loading custom model...")
            # Custom model placeholder; replace with your model loader/checkpoint.
            custom_model = load_model(
                args.checkpoint, args.cfg, allow_unexpected=args.allow_unexpected
            )

            print("\n==> Running inference...")
            baseline_pred = predict(baseline_model, pose2d)
            custom_pred = predict_custom(custom_model, pose2d)

            baseline_mpjpe = compute_mpjpe(baseline_pred, pose3d_gt)
            custom_mpjpe = compute_mpjpe(custom_pred, pose3d_gt)
            print(f"✅ Baseline MPJPE: {baseline_mpjpe:.1f} mm")
            print(f"✅ Custom MPJPE: {custom_mpjpe:.1f} mm")

            print("\n==> Plotting comparison...")
            plot_pose(
                baseline_pred,
                pose3d_gt,
                custom3d=custom_pred,
                title=(
                    f"Baseline vs Custom vs Ground Truth | MPJPE: "
                    f"{baseline_mpjpe:.1f}/{custom_mpjpe:.1f} mm"
                ),
                save_path='comparison_pose.png'
            )

            if args.overlay:
                print("\n==> Plotting overlay...")
                plot_overlay(
                    baseline_pred,
                    custom_pred,
                    title="Baseline vs Custom Overlay",
                    save_path='comparison_pose_overlay.png'
                )
        else:
            print("\n==> Running inference...")
            pose3d_pred = predict(baseline_model, pose2d)

            mpjpe = compute_mpjpe(pose3d_pred, pose3d_gt)
            print(f"✅ Sample MPJPE: {mpjpe:.1f} mm")

            print("\n==> Plotting...")
            plot_pose(
                pose3d_pred,
                pose3d_gt,
                title=f"HR-GCN vs Ground Truth | MPJPE: {mpjpe:.1f} mm"
            )
