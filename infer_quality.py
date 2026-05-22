"""
Visual real-time inference script for the Dual-Stream Quality Assessment Network.
Renders a 6-panel dashboard per sliding window and saves as MP4 or shows live.
"""

import argparse
import os
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np
import torch
import torch.nn.functional as F

from dual_stream_quality import (
    DualStreamQualityNet,
    BODY_JOINT_IDX,
    SKELETON_EDGES,
    EXERCISE_NAMES,
    J,
    build_topology_adjacency,
)

# ─── Clinical ROM reference limits (degrees) ──────────────────────────────────

ROM_NAMES = [
    'Cerv Pitch', 'Trunk Flex',
    'L Sho Flex', 'R Sho Flex',
    'L Sho Abd',  'R Sho Abd',
    'L Hip',      'R Hip',
    'L Knee',     'R Knee',
    'L Ankle',    'R Ankle',
]

ROM_LIMITS = [
    (0,  60),   # cerv pitch
    (0,  90),   # trunk flex
    (0, 180),   # l sho flex
    (0, 180),   # r sho flex
    (0, 180),   # l sho abd
    (0, 180),   # r sho abd
    (0, 120),   # l hip
    (0, 120),   # r hip
    (0, 150),   # l knee
    (0, 150),   # r knee
    (60, 120),  # l ankle (90 = neutral)
    (60, 120),  # r ankle
]

# Joint colour map indexed by joint index
_JOINT_COLORS = ['white'] * J
_JOINT_COLORS[0]  = 'red'                                   # head
_JOINT_COLORS[1]  = _JOINT_COLORS[2] = _JOINT_COLORS[3] = 'dodgerblue'   # spine
_JOINT_COLORS[5]  = _JOINT_COLORS[6] = _JOINT_COLORS[7] = 'limegreen'    # l arm
_JOINT_COLORS[8]  = _JOINT_COLORS[9] = _JOINT_COLORS[10] = 'orange'      # r arm
_JOINT_COLORS[11] = _JOINT_COLORS[12] = _JOINT_COLORS[13] = 'mediumpurple'  # l leg
_JOINT_COLORS[14] = _JOINT_COLORS[15] = _JOINT_COLORS[16] = 'saddlebrown'   # r leg


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Dual-Stream Quality Inference Dashboard')
    p.add_argument('--checkpoint',  default='results/dual_stream_improve/dual_segmented_best.pt')
    p.add_argument('--npz',         default='data/uiprmd_segmented_gmm_test.npz')
    p.add_argument('--subject',     type=int,   default=9,
                   help='Subject ID to visualise (0-indexed)')
    p.add_argument('--exercise',    type=int,   default=0,
                   help='Exercise ID 0-9')
    p.add_argument('--quality',     type=int,   default=1,
                   help='0=incorrect, 1=correct')
    p.add_argument('--window_size', type=int,   default=100)
    p.add_argument('--stride',      type=int,   default=50)
    p.add_argument('--output_dir',  default='results/dual_stream_improve/inference')
    p.add_argument('--save_video',  action='store_true',
                   help='Save MP4 instead of showing an interactive window')
    p.add_argument('--fps',         type=int,   default=10)
    p.add_argument('--seed',        type=int,   default=42)
    return p.parse_args()


# ─── Model loading ────────────────────────────────────────────────────────────

def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sa = ckpt.get('args', {})
    hidden_dim = sa.get('hidden_dim', 64)
    M          = sa.get('M', 100)

    A_topology = build_topology_adjacency(J)
    model = DualStreamQualityNet(
        hidden_dim=hidden_dim, M=M,
        n_joints=J, n_exercises=10,
        A_topology=A_topology,
        rom_guided_inits=None,
    ).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    rom_mean = ckpt['rom_mean']   # (10, 12)  np.ndarray
    rom_std  = ckpt['rom_std']    # (10, 12)
    return model, rom_mean, rom_std


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_subject_sequence(npz_path, subject_id, exercise_id, quality_label):
    d = np.load(npz_path, allow_pickle=True)

    mask = (
        (d['subject_ids']   == subject_id)  &
        (d['exercise_ids']  == exercise_id) &
        (d['quality_labels'] == quality_label)
    )
    if mask.sum() == 0:
        avail_s = np.unique(d['subject_ids'])
        avail_e = np.unique(d['exercise_ids'])
        avail_q = np.unique(d['quality_labels'])
        raise ValueError(
            f"No frames for subject={subject_id}, exercise={exercise_id}, "
            f"quality={quality_label}.\n"
            f"Available  subjects={avail_s}  exercises={avail_e}  quality={avail_q}"
        )

    poses_3d   = d['poses_3d'][mask]     # (N, 133, 3)
    rom_angles = d['rom_angles'][mask]   # (N, 12)
    gmm_scores = (d['quality_scores'][mask]
                  if 'quality_scores' in d
                  else np.zeros(mask.sum(), dtype=np.float32))

    q_str = 'Correct' if quality_label == 1 else 'Incorrect'
    print(
        f"Loaded {mask.sum()} frames for "
        f"Subject {subject_id}, Exercise {exercise_id} "
        f"({EXERCISE_NAMES[exercise_id]}), Quality: {q_str}"
    )
    return poses_3d, rom_angles, gmm_scores


# ─── Per-window inference ─────────────────────────────────────────────────────

@torch.no_grad()
def run_inference_on_window(model, joints_window, rom_window,
                            rom_mean, rom_std, exercise_id, device):
    """
    joints_window : (W, 17, 3)
    rom_window    : (W, 12)  raw degrees
    Returns (quality_score, exercise_probs[10], valid_label, attention[W])
    """
    ex = exercise_id
    rom_norm = (rom_window - rom_mean[ex]) / (rom_std[ex] + 1e-8)

    jw = joints_window.copy()
    jw -= jw[0, 0, :]                    # subtract spine origin

    j_t  = torch.tensor(jw,       dtype=torch.float32).unsqueeze(0).to(device)
    r_t  = torch.tensor(rom_norm, dtype=torch.float32).unsqueeze(0).to(device)
    ex_t = torch.tensor([ex],     dtype=torch.long).to(device)

    ex_logits, vc_logits, qual_score, attn = model(j_t, r_t, ex_t)

    quality_score  = float(qual_score[0].item())
    exercise_probs = F.softmax(ex_logits[0], dim=0).cpu().numpy()
    valid_label    = int(vc_logits[0].argmax().item())
    attention      = attn[0].cpu().numpy()
    return quality_score, exercise_probs, valid_label, attention


# ─── Drawing helpers ──────────────────────────────────────────────────────────

def _dark_style(ax):
    ax.set_facecolor('#0d0d1a')
    ax.tick_params(colors='#aaaaaa', labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor('#333355')


def draw_skeleton(ax, joints_frame, exercise_name, subject_id):
    """joints_frame : (17, 3)"""
    ax.cla()
    ax.set_facecolor('#0d0d1a')
    ax.grid(False)

    # edges
    for (i, j) in SKELETON_EDGES:
        if i < J and j < J:
            ax.plot(
                [joints_frame[i, 0], joints_frame[j, 0]],
                [joints_frame[i, 1], joints_frame[j, 1]],
                [joints_frame[i, 2], joints_frame[j, 2]],
                color='#556688', linewidth=1.4, alpha=0.85,
            )

    # joints
    ax.scatter(
        joints_frame[:, 0],
        joints_frame[:, 1],
        joints_frame[:, 2],
        c=_JOINT_COLORS, s=28, depthshade=True, zorder=5,
    )

    ax.set_title(
        f'Exercise: {exercise_name}  |  Subject: {subject_id}',
        color='#dddddd', fontsize=8, pad=3, fontweight='bold',
    )
    ax.tick_params(colors='#888888', labelsize=5)
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#1e1e3a')
    ax.yaxis.pane.set_edgecolor('#1e1e3a')
    ax.zaxis.pane.set_edgecolor('#1e1e3a')


def draw_gauge(ax, score):
    ax.cla()
    ax.set_facecolor('#0d0d1a')
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-0.25, 1.3)
    ax.set_aspect('equal', adjustable='datalim')
    ax.axis('off')

    # Coloured arc bands
    for (t0_frac, t1_frac, col) in [
        (0.0, 0.4, '#993333'),
        (0.4, 0.7, '#997700'),
        (0.7, 1.0, '#2d7a3a'),
    ]:
        t_seg = np.linspace(np.pi * (1 - t0_frac), np.pi * (1 - t1_frac), 80)
        x_in  = 0.56 * np.cos(t_seg);  y_in  = 0.56 * np.sin(t_seg)
        x_out = 0.92 * np.cos(t_seg);  y_out = 0.92 * np.sin(t_seg)
        ax.fill(
            np.concatenate([x_out, x_in[::-1]]),
            np.concatenate([y_out, y_in[::-1]]),
            color=col, alpha=0.55,
        )

    # Needle
    angle = np.pi * (1.0 - score)
    nx = 0.80 * np.cos(angle)
    ny = 0.80 * np.sin(angle)
    ncol = '#ff4444' if score < 0.4 else ('#ffcc00' if score < 0.7 else '#44cc66')
    ax.annotate('', xy=(nx, ny), xytext=(0.0, 0.0),
                arrowprops=dict(arrowstyle='->', color=ncol, lw=2.8))
    ax.plot(0, 0, 'o', color='white', markersize=5, zorder=6)

    ax.text(0, 0.36, f'{score:.3f}',
            ha='center', va='center', color=ncol, fontsize=19, fontweight='bold')
    ax.text(0, 0.13, 'Quality Score',
            ha='center', va='center', color='#aaaaaa', fontsize=8)

    border = '#44cc66' if score >= 0.7 else ('#ff4444' if score < 0.4 else '#ffcc00')
    rect = plt.Rectangle((-1.22, -0.22), 2.44, 1.5,
                         fill=False, edgecolor=border, linewidth=2.5,
                         transform=ax.transData, clip_on=False)
    ax.add_patch(rect)


def draw_exercise_probs(ax, probs, predicted_ex, confidence):
    ax.cla()
    _dark_style(ax)
    colors = ['#2a8a4a' if i == predicted_ex else '#334488'
              for i in range(10)]
    ax.barh(range(10), probs, color=colors, height=0.62, edgecolor='none')
    ax.set_yticks(range(10))
    ax.set_yticklabels(EXERCISE_NAMES, fontsize=7, color='#cccccc')
    ax.set_xlim(0, 1.0)
    ax.set_xlabel('Probability', color='#aaaaaa', fontsize=7)
    pred_name = EXERCISE_NAMES[predicted_ex]
    ax.set_title(f'Predicted: {pred_name} ({confidence:.0%})',
                 color='#dddddd', fontsize=9, pad=4, fontweight='bold')


def draw_rom_bars(ax, rom_frame_raw):
    """rom_frame_raw : (12,) raw degrees"""
    ax.cla()
    _dark_style(ax)
    colors = []
    for i, val in enumerate(rom_frame_raw):
        lo, hi = ROM_LIMITS[i]
        margin = max((hi - lo) * 0.15, 5)
        if val < lo - margin or val > hi + margin:
            colors.append('#cc3333')
        elif val < lo + margin or val > hi - margin:
            colors.append('#ccaa00')
        else:
            colors.append('#33aa55')

    ax.barh(range(12), rom_frame_raw, color=colors, height=0.62, edgecolor='none')
    for i, (lo, hi) in enumerate(ROM_LIMITS):
        ax.plot([lo, lo], [i - 0.38, i + 0.38], color='white', lw=0.9, alpha=0.45)
        ax.plot([hi, hi], [i - 0.38, i + 0.38], color='white', lw=0.9, alpha=0.45)

    ax.set_yticks(range(12))
    ax.set_yticklabels(ROM_NAMES, fontsize=6, color='#cccccc')
    ax.set_xlabel('Degrees', color='#aaaaaa', fontsize=7)
    ax.set_title('ROM Angles (degrees)', color='#dddddd', fontsize=9,
                 pad=4, fontweight='bold')


def draw_attention(ax, attention):
    """attention : (W,)"""
    ax.cla()
    _dark_style(ax)
    W   = len(attention)
    img = attention.reshape(1, -1)
    ax.imshow(img, aspect='auto', cmap='YlOrRd',
              extent=[0, W, 0, 1], vmin=0, vmax=attention.max() + 1e-8)
    peak = int(np.argmax(attention))
    ax.axvline(peak, color='white', lw=1.2, linestyle='--', alpha=0.85)
    ax.set_xlabel('Frame', color='#aaaaaa', fontsize=7)
    ax.set_yticks([])
    ax.set_xlim(0, W)
    ax.set_title(f'BiLSTM Attention Weights  (peak @ frame {peak})',
                 color='#dddddd', fontsize=9, pad=4, fontweight='bold')


def draw_timeline(ax, scores_history, current_win):
    ax.cla()
    _dark_style(ax)
    xs = list(range(len(scores_history)))
    ax.plot(xs, scores_history, color='#5599ff', linewidth=1.5)
    ax.fill_between(xs, 0.7, 1.0, alpha=0.10, color='#44cc66')
    ax.fill_between(xs, 0.0, 0.4, alpha=0.10, color='#cc3333')
    ax.axhline(0.7, color='#44cc66', lw=0.9, linestyle='--', alpha=0.6)
    ax.axhline(0.4, color='#cc3333', lw=0.9, linestyle='--', alpha=0.6)
    if 0 <= current_win < len(scores_history):
        ax.axvline(current_win, color='white', lw=1.0, linestyle=':', alpha=0.7)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, max(len(scores_history) + 1, 5))
    ax.set_xlabel('Window', color='#aaaaaa', fontsize=7)
    ax.set_ylabel('Quality Score', color='#aaaaaa', fontsize=7)
    ax.set_title('Quality Score Over Time',
                 color='#dddddd', fontsize=9, pad=4, fontweight='bold')


def update_banner(fig, valid_label, subject_id, exercise_id, quality_label, win_idx, n_wins):
    if valid_label == 0:
        bg_col, symbol = '#1d5c2e', '✓  VALID EXECUTION'
    else:
        bg_col, symbol = '#6b1a1a', '✗  INVALID EXECUTION'
    q_str = 'Correct' if quality_label == 1 else 'Incorrect'
    title = (
        f'{symbol}     |     Subject {subject_id}  ·  '
        f'Exercise {EXERCISE_NAMES[exercise_id]}  ·  {q_str}  '
        f'·  Window {win_idx + 1}/{n_wins}'
    )
    fig.suptitle(
        title, fontsize=11, color='white', fontweight='bold', y=0.97,
        bbox=dict(facecolor=bg_col, edgecolor='none',
                  boxstyle='round,pad=0.35', alpha=0.92),
    )


# ─── Build figure with 6-panel layout ────────────────────────────────────────
#
#  ┌─────────────────┬──────────────┬──────────────┐
#  │  3D Skeleton    │  Quality     │  Exercise    │
#  │  Animation      │  Gauge       │  Probability │
#  ├─────────────────┼──────────────┴──────────────┤
#  │  ROM Angles     │  Attention Heatmap           │
#  │  Bar Chart      │  (spans cols 1-2)            │
#  ├─────────────────┴──────────────────────────────┤
#  │  Quality Score Timeline  (spans all cols)      │
#  └────────────────────────────────────────────────┘

def build_figure():
    fig = plt.figure(figsize=(20, 11), facecolor='#111122')

    gs = gridspec.GridSpec(
        3, 3,
        figure=fig,
        left=0.055, right=0.975,
        top=0.90,   bottom=0.06,
        hspace=0.50, wspace=0.32,
        height_ratios=[1.6, 1.4, 0.8],
    )

    ax_skel     = fig.add_subplot(gs[0, 0], projection='3d')
    ax_gauge    = fig.add_subplot(gs[0, 1])
    ax_exercise = fig.add_subplot(gs[0, 2])
    ax_rom      = fig.add_subplot(gs[1, 0])
    ax_attn     = fig.add_subplot(gs[1, 1:])    # attention spans cols 1-2
    ax_timeline = fig.add_subplot(gs[2, :])     # timeline spans all cols

    for ax in (ax_gauge, ax_exercise, ax_rom, ax_attn, ax_timeline):
        ax.set_facecolor('#0d0d1a')
    ax_skel.set_facecolor('#0d0d1a')

    return fig, ax_skel, ax_gauge, ax_exercise, ax_rom, ax_attn, ax_timeline


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    print('Loading model ...')
    model, rom_mean, rom_std = load_model(args.checkpoint, device)

    poses_3d, rom_angles, gmm_scores = load_subject_sequence(
        args.npz, args.subject, args.exercise, args.quality
    )

    N = len(poses_3d)
    W = args.window_size

    # Build sliding windows
    windows = []
    start = 0
    while start + W <= N:
        jw = poses_3d[start:start + W][:, BODY_JOINT_IDX, :]   # (W, 17, 3)
        rw = rom_angles[start:start + W]                         # (W, 12)
        windows.append((jw, rw, start))
        start += args.stride

    if not windows:
        raise RuntimeError(
            f'Sequence too short ({N} frames) for window_size={W}. '
            'Reduce --window_size or pick a longer sequence.'
        )
    n_wins = len(windows)
    print(f'Built {n_wins} windows  (stride={args.stride})')

    # Pre-compute inference for all windows
    print('Running inference on all windows ...')
    results = []
    for jw, rw, frame_start in windows:
        qs, ex_probs, valid, attn = run_inference_on_window(
            model, jw, rw, rom_mean, rom_std, args.exercise, device
        )
        results.append({
            'quality_score':  qs,
            'exercise_probs': ex_probs,
            'valid_label':    valid,
            'attention':      attn,
            'joints_window':  jw,
            'frame_start':    frame_start,
        })

    quality_scores = [r['quality_score'] for r in results]
    pred_exercises = [int(np.argmax(r['exercise_probs'])) for r in results]
    ex_acc = float(np.mean([p == args.exercise for p in pred_exercises]))
    print(f'Quality scores — min: {min(quality_scores):.3f}  '
          f'max: {max(quality_scores):.3f}  '
          f'mean: {np.mean(quality_scores):.3f}')
    print(f'Exercise classification accuracy on this sequence: {ex_acc:.0%}')

    # Switch to an interactive backend BEFORE creating the figure so the
    # canvas is compatible with plt.show().  Agg was locked in by the
    # dual_stream_quality import; plt.switch_backend() overrides it at
    # runtime.  For save_video we keep Agg (headless-safe).
    if not args.save_video:
        try:
            plt.switch_backend('TkAgg')
        except ImportError:
            print('[warn] No interactive display found — switching to --save_video mode.')
            args.save_video = True

    # ── Build animated figure ─────────────────────────────────────────────────
    fig, ax_skel, ax_gauge, ax_exercise, ax_rom, ax_attn, ax_timeline = build_figure()
    ex_name       = EXERCISE_NAMES[args.exercise]
    scores_history = []

    def animate(win_idx):
        r   = results[win_idx]
        scores_history.append(r['quality_score'])

        # Representative frame = middle of the window
        mid          = W // 2
        joints_frame = r['joints_window'][mid]                # (17, 3)
        rom_raw      = rom_angles[r['frame_start'] + mid]    # (12,) raw degrees

        pred_ex    = int(np.argmax(r['exercise_probs']))
        confidence = float(r['exercise_probs'][pred_ex])

        draw_skeleton(ax_skel, joints_frame, ex_name, args.subject)
        draw_gauge(ax_gauge, r['quality_score'])
        draw_exercise_probs(ax_exercise, r['exercise_probs'], pred_ex, confidence)
        draw_rom_bars(ax_rom, rom_raw)
        draw_attention(ax_attn, r['attention'])
        draw_timeline(ax_timeline, scores_history, win_idx)
        update_banner(fig, r['valid_label'], args.subject,
                      args.exercise, args.quality, win_idx, n_wins)
        return []

    anim = FuncAnimation(
        fig, animate,
        frames=n_wins,
        interval=int(1000 / args.fps),
        blit=False,
        repeat=False,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    q_str      = 'correct' if args.quality == 1 else 'incorrect'
    video_name = f's{args.subject:02d}_ex{args.exercise + 1:02d}_{q_str}.mp4'
    video_path = os.path.join(args.output_dir, video_name)

    if args.save_video:
        print(f'Saving video → {video_path} ...')
        writer = FFMpegWriter(fps=args.fps,
                              metadata={'title': 'Dual-Stream Quality Dashboard'})
        anim.save(video_path, writer=writer, dpi=88,
                  savefig_kwargs={'facecolor': '#111122'})
        print(f'Video saved: {video_path}')
    else:
        plt.show()

    plt.close(fig)

    # ── Final summary ─────────────────────────────────────────────────────────
    print('\n=== Inference Summary ===')
    print(f'  File written:       infer_quality.py')
    print(f'  Windows processed:  {n_wins}')
    print(f'  Quality range:      [{min(quality_scores):.3f}, {max(quality_scores):.3f}]')
    print(f'  Exercise acc:       {ex_acc:.0%}')
    if args.save_video:
        print(f'  Video saved:        {video_path}')

    print('\nDisplay command (interactive — run on a machine with a monitor):')
    print(f'  python infer_quality.py \\')
    print(f'    --checkpoint {args.checkpoint} \\')
    print(f'    --npz {args.npz} \\')
    print(f'    --subject {args.subject} \\')
    print(f'    --exercise 6 \\')
    print(f'    --quality 1')


if __name__ == '__main__':
    main()
