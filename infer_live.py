"""
Live inference for the base HR-GCN (HRGCN, model=1) on a webcam or video file.

Produces a real-time skeleton overlay showing all 133 joints:
  body (23), face (68), left hand (21), right hand (21).

2D detection is handled by RTMPose-W (requires mmpose) or can be bypassed
with --skip_pose_detector for testing the 3D pipeline with zero keypoints.

Usage:
    # Webcam with RTMPose
    python infer_live.py \
        --checkpoint checkpoint/ckpt_best.pth.tar \
        --cfg w32_adam_lr1e-3.yaml \
        --rtmpose_config <config.py> \
        --rtmpose_checkpoint <weights.pth>

    # Video file, skip 2D detector
    python infer_live.py \
        --source video.mp4 \
        --checkpoint checkpoint/ckpt_best.pth.tar \
        --cfg w32_adam_lr1e-3.yaml \
        --skip_pose_detector \
        --save_video out.mp4

    # CPU-only (no CUDA)
    python infer_live.py \
        --checkpoint checkpoint/ckpt_best.pth.tar \
        --cfg w32_adam_lr1e-3.yaml \
        --skip_pose_detector \
        --device cpu
"""

from __future__ import print_function, absolute_import, division

import argparse
import os
import os.path as path
import time

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from lib.config import cfg
from common.graph_utils import adj_mx_from_skeleton
from utils.prepare_data_h3wb import Human3WBDataset
import models.graph_hrnet_multi_branch as ghrmb

try:
    from mmpose.apis import init_model as mmpose_init_model
    from mmpose.apis import inference_topdown
    MMPOSE_AVAILABLE = True
except ImportError:
    MMPOSE_AVAILABLE = False

N_JOINTS = 133

# ---------------------------------------------------------------------------
# Skeleton drawing config
# ---------------------------------------------------------------------------

# BGR colors
_BODY_COLOR  = (0, 200, 100)   # green
_FACE_COLOR  = (200, 140, 60)  # blue-grey
_HAND_COLOR  = (60, 140, 240)  # orange

# COCO-style body edges (indices into 23-joint body array)
_BODY_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # head chain
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), # arms
    (5, 11), (6, 12), (11, 12),               # torso
    (11, 13), (13, 15),                       # left leg
    (12, 14), (14, 16),                       # right leg
]

# Finger edges inside each 21-joint hand (relative offsets from wrist=0)
_FINGER_EDGES = [
    (0,1),(1,2),(2,3),(3,4),       # thumb
    (0,5),(5,6),(6,7),(7,8),       # index
    (0,9),(9,10),(10,11),(11,12),  # middle
    (0,13),(13,14),(14,15),(15,16),# ring
    (0,17),(17,18),(18,19),(19,20),# pinky
    (5,9),(9,13),(13,17),          # palm arch
]

# Approximate face contour: just draw lines along the 68-pt dlib landmark ring
# Landmark layout: jaw 0-16, right brow 17-21, left brow 22-26,
#   nose 27-35, right eye 36-41, left eye 42-47, outer mouth 48-59, inner 60-67
_FACE_CONTOURS = (
    list(zip(range(0, 16), range(1, 17))),      # jaw
    list(zip(range(17, 21), range(18, 22))),    # right brow
    list(zip(range(22, 26), range(23, 27))),    # left brow
    list(zip(range(27, 30), range(28, 31))),    # nose bridge
    list(zip(range(31, 35), range(32, 36))),    # nose base
    list(zip(range(36, 41), range(37, 42))) + [(41, 36)],  # right eye
    list(zip(range(42, 47), range(43, 48))) + [(47, 42)],  # left eye
    list(zip(range(48, 59), range(49, 60))) + [(59, 48)],  # outer mouth
    list(zip(range(60, 67), range(61, 68))) + [(67, 60)],  # inner mouth
)
_FACE_EDGES = [e for group in _FACE_CONTOURS for e in group]


def _project(joints_3d: np.ndarray, h: int, w: int, scale: float = 300.0) -> np.ndarray:
    """Orthographic projection: X→right, Y→up, discard Z."""
    pts = joints_3d[:, :2].copy()
    pts[:, 0] =  pts[:, 0] * scale + w // 2
    pts[:, 1] = -pts[:, 1] * scale + h // 2
    return pts.astype(int)


def _clip(pt, w, h):
    return (int(np.clip(pt[0], 0, w - 1)), int(np.clip(pt[1], 0, h - 1)))


def draw_skeleton(frame: np.ndarray,
                  body: np.ndarray,   # (23, 3)
                  face: np.ndarray,   # (68, 3)
                  lhand: np.ndarray,  # (21, 3)
                  rhand: np.ndarray,  # (21, 3)
                  show_face: bool = True,
                  show_hands: bool = True) -> np.ndarray:
    H, W = frame.shape[:2]

    # Body
    j2d = _project(body, H, W)
    for a, b in _BODY_EDGES:
        if a < len(j2d) and b < len(j2d):
            cv2.line(frame, _clip(j2d[a], W, H), _clip(j2d[b], W, H),
                     _BODY_COLOR, 2, cv2.LINE_AA)
    for pt in j2d:
        cv2.circle(frame, _clip(pt, W, H), 4, _BODY_COLOR, -1, cv2.LINE_AA)

    # Face
    if show_face and face is not None:
        f2d = _project(face, H, W)
        for a, b in _FACE_EDGES:
            if a < len(f2d) and b < len(f2d):
                cv2.line(frame, _clip(f2d[a], W, H), _clip(f2d[b], W, H),
                         _FACE_COLOR, 1, cv2.LINE_AA)
        for pt in f2d:
            cv2.circle(frame, _clip(pt, W, H), 2, _FACE_COLOR, -1, cv2.LINE_AA)

    # Hands
    if show_hands:
        for hand_3d in (lhand, rhand):
            if hand_3d is None:
                continue
            h2d = _project(hand_3d, H, W)
            for a, b in _FINGER_EDGES:
                if a < len(h2d) and b < len(h2d):
                    cv2.line(frame, _clip(h2d[a], W, H), _clip(h2d[b], W, H),
                             _HAND_COLOR, 1, cv2.LINE_AA)
            for pt in h2d:
                cv2.circle(frame, _clip(pt, W, H), 2, _HAND_COLOR, -1, cv2.LINE_AA)

    return frame


def draw_hud(frame: np.ndarray, fps: float, frame_idx: int) -> np.ndarray:
    cv2.putText(frame, f'FPS: {fps:.1f}  Frame: {frame_idx}',
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (220, 220, 220), 1, cv2.LINE_AA)
    return frame


# ---------------------------------------------------------------------------
# Preprocessing (mirrors H3WB pipeline)
# ---------------------------------------------------------------------------

def preprocess_keypoints(kpts_2d: np.ndarray) -> np.ndarray:
    """kpts_2d: (133, 2) pixel coords → (133, 2) normalised, hip-centred."""
    hip_center = (kpts_2d[11] + kpts_2d[12]) / 2.0
    centered = kpts_2d - hip_center
    norms = np.linalg.norm(centered, axis=-1)
    scale = np.percentile(norms[norms > 1e-3], 95) if (norms > 1e-3).any() else 1.0
    return (centered / max(scale, 1e-6)).astype(np.float32)


# ---------------------------------------------------------------------------
# RTMPose detector (optional)
# ---------------------------------------------------------------------------

class RTMPoseDetector:
    def __init__(self, config_path: str, ckpt_path: str, device: str):
        if not MMPOSE_AVAILABLE:
            raise RuntimeError(
                'mmpose not installed. Run:\n'
                '  pip install openmim && python -m mim install mmpose\n'
                'Or pass --skip_pose_detector.'
            )
        self.model = mmpose_init_model(config_path, ckpt_path, device=device)

    def detect(self, frame_bgr: np.ndarray) -> np.ndarray:
        result = inference_topdown(self.model, frame_bgr)
        if result and len(result) > 0:
            kpts = result[0].pred_instances.keypoints[0]
            return np.array(kpts, dtype=np.float32)
        return np.zeros((N_JOINTS, 2), dtype=np.float32)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='HR-GCN live inference (base HRGCN)')
    p.add_argument('--source', default='0',
                   help='Video source: 0 = webcam, or path to video file')
    p.add_argument('--checkpoint', required=True,
                   help='Path to H3WB checkpoint (ckpt_best.pth.tar)')
    p.add_argument('-cfg', '--cfg', default='w32_adam_lr1e-3.yaml',
                   help='Model config YAML (same one used during training)')
    p.add_argument('--gcn', default='dc_preagg',
                   help='GCN variant used when training the checkpoint')
    p.add_argument('--dropout', default=0.0, type=float)
    p.add_argument('--device', default='cuda:0',
                   help='Torch device string, e.g. cuda:0 or cpu')
    p.add_argument('--save_video', default='',
                   help='Write annotated output to this path (empty = display)')
    p.add_argument('--skip_pose_detector', action='store_true',
                   help='Feed zero 2D keypoints — useful for testing without RTMPose')
    p.add_argument('--rtmpose_config', default='',
                   help='mmpose config .py for RTMPose-W wholebody')
    p.add_argument('--rtmpose_checkpoint', default='',
                   help='RTMPose checkpoint path')
    p.add_argument('--no_face', action='store_true',
                   help='Skip face skeleton overlay')
    p.add_argument('--no_hands', action='store_true',
                   help='Skip hand skeleton overlay')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    device = torch.device(args.device)
    if 'cuda' in args.device:
        cudnn.benchmark = True

    # ---- Build skeleton + adjacency matrix ----
    print('==> Loading skeleton structure...')
    dataset = Human3WBDataset('data/h3wb_train.npz', 'data/h3wb_test.npz')
    skeleton = dataset.skeleton()
    adj = adj_mx_from_skeleton(skeleton).to(device)

    # ---- Build model ----
    print('==> Building HR-GCN backbone...')
    cfg.merge_from_file(args.cfg)
    p_dropout = None if args.dropout == 0.0 else args.dropout
    model = ghrmb.get_pose_net(
        cfg, True, adj, p_dropout, args.gcn, skeleton.joints_group()
    ).to(device)
    print(f'    Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')

    # ---- Load checkpoint ----
    print(f'==> Loading checkpoint: {args.checkpoint}')
    if not path.isfile(args.checkpoint):
        raise FileNotFoundError(f'Checkpoint not found: {args.checkpoint}')
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['state_dict'], strict=False)
    epoch = ckpt.get('epoch', '?')
    error = ckpt.get('error', '?')
    print(f'    Loaded epoch={epoch}  MPJPE={error}')
    model.eval()

    # ---- 2D detector ----
    detector = None
    if not args.skip_pose_detector:
        if not args.rtmpose_config or not args.rtmpose_checkpoint:
            raise ValueError(
                'Provide --rtmpose_config and --rtmpose_checkpoint, '
                'or add --skip_pose_detector to bypass RTMPose.'
            )
        print('==> Initialising RTMPose...')
        detector = RTMPoseDetector(
            args.rtmpose_config, args.rtmpose_checkpoint, args.device
        )
    else:
        print('==> Skipping 2D pose detector — using zero keypoints.')

    # ---- Video source ----
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video source: {args.source}')

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'==> Source: {args.source}  {frame_w}x{frame_h} @ {fps_src:.1f} fps')

    # ---- Optional video writer ----
    writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.save_video, fourcc, fps_src, (frame_w, frame_h))
        print(f'==> Saving to: {args.save_video}')

    frame_idx = 0
    fps_avg = 0.0
    t_start = time.time()

    print('==> Running inference. Press Q to quit.')
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t_frame = time.time()

            # -- 2D keypoints --
            if detector is not None:
                kpts_2d = detector.detect(frame)
            else:
                kpts_2d = np.zeros((N_JOINTS, 2), dtype=np.float32)

            kpts_norm = preprocess_keypoints(kpts_2d)
            inp = torch.from_numpy(kpts_norm).unsqueeze(0).to(device)  # (1, 133, 2)

            # -- 3D inference --
            body_3d, face_3d, lhand_3d, rhand_3d = model(inp)

            # Hip-center the output (mirrors evaluate() in HRNet_GCN_WB.py)
            body_np  = body_3d[0].cpu().numpy()   # (23, 3)
            hip_mid  = (body_np[11] + body_np[12]) / 2.0
            body_np  = body_np  - hip_mid
            face_np  = face_3d[0].cpu().numpy()  - hip_mid   # (68, 3)
            lhand_np = lhand_3d[0].cpu().numpy() - hip_mid   # (21, 3)
            rhand_np = rhand_3d[0].cpu().numpy() - hip_mid   # (21, 3)

            # -- Visualise --
            vis = draw_skeleton(
                frame.copy(),
                body_np, face_np, lhand_np, rhand_np,
                show_face=not args.no_face,
                show_hands=not args.no_hands,
            )

            elapsed_frame = time.time() - t_frame
            fps_avg = 0.9 * fps_avg + 0.1 * (1.0 / max(elapsed_frame, 1e-6))
            draw_hud(vis, fps_avg, frame_idx)

            if writer is not None:
                writer.write(vis)
            else:
                cv2.imshow('HR-GCN Live Inference', vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_idx += 1
            if frame_idx % 60 == 0:
                total = time.time() - t_start
                print(f'  Frame {frame_idx}  FPS: {fps_avg:.1f}  Elapsed: {total:.1f}s')

    cap.release()
    if writer is not None:
        writer.release()
        print(f'Saved to {args.save_video}')
    cv2.destroyAllWindows()
    print(f'Done. Total frames: {frame_idx}')


if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    main()
