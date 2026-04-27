"""
Real-time rehabilitation inference pipeline.

RGB source → RTMPose-W (2D keypoints) → HR-GCN (3D joints) →
One-Euro filter (smoothing) → ClinicalAngleHead (ROM angles) →
Anatomical clamp → Skeleton overlay + ROM dashboard

Usage:
    python infer_rehab.py \
        --source 0 \
        --checkpoint checkpoint_rehab/ckpt_best_rehab.pth.tar \
        --cfg w32_adam_lr1e-3.yaml

    # Save to video file
    python infer_rehab.py --source video.mp4 --save_video out.mp4 \
        --checkpoint checkpoint_rehab/ckpt_best_rehab.pth.tar \
        --cfg w32_adam_lr1e-3.yaml

Note: RTMPose requires mmpose + mmcv. Install via:
    pip install openmim && python -m mim install mmpose
If unavailable, --skip_pose_detector feeds zeros as the 2D keypoints
(useful for unit-testing the 3D and angle pipeline only).
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
from common.anatomical_constraints import clamp_angles_to_valid_range
from common.one_euro_filter import SkeletonFilter
from common.visualize import draw_skeleton_on_frame
from models.clinical_angle_head import ClinicalAngleHead
from utils.prepare_data_h3wb import Human3WBDataset

import models.graph_hrnet_multi_branch as ghrmb
import models.graph_resnet as GraphRes
import models.graph_hrnet as ghr
from models.graph_sh import GraphSH

# RTMPose is optional — only needed for full end-to-end inference
try:
    from mmpose.apis import init_model as mmpose_init_model
    from mmpose.apis import inference_topdown
    MMPOSE_AVAILABLE = True
except ImportError:
    MMPOSE_AVAILABLE = False

N_JOINTS = 133


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='GCADA real-time inference')
    parser.add_argument('--source', default='0',
                        help='Video source: 0 for webcam, or path to video file')
    parser.add_argument('--checkpoint', required=True,
                        help='Path to rehab checkpoint (ckpt_best_rehab.pth.tar)')
    parser.add_argument('-cfg', '--cfg', default='w32_adam_lr1e-3.yaml',
                        help='Model config yaml')
    parser.add_argument('--gcn', default='dc_preagg')
    parser.add_argument('-m', '--model', default=1, type=int)
    parser.add_argument('--hid_dim', default=64, type=int)
    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--save_video', default='', type=str,
                        help='Output video path (empty = display only)')
    parser.add_argument('--skip_pose_detector', action='store_true',
                        help='Feed zero 2D keypoints (test 3D + angle pipeline '
                             'without RTMPose)')
    parser.add_argument('--rtmpose_config', default='', type=str,
                        help='mmpose config for RTMPose model')
    parser.add_argument('--rtmpose_checkpoint', default='', type=str,
                        help='RTMPose checkpoint path')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _get_skeleton():
    dataset = Human3WBDataset('data/h3wb_train.npz', 'data/h3wb_test.npz')
    return dataset.skeleton()


def build_backbone(args, cfg, adj, device):
    p_dropout = None if args.dropout == 0.0 else args.dropout
    skeleton = _get_skeleton()

    if args.model == 1:
        model = ghrmb.get_pose_net(cfg, True, adj, p_dropout, args.gcn,
                                   skeleton.joints_group())
    elif args.model == 2:
        model = GraphRes.get_pose_net(True, adj, p_dropout, args.gcn, 50, True)
    elif args.model == 3:
        model = ghr.get_pose_net(cfg, True, adj, p_dropout, args.gcn,
                                 skeleton.joints_group())
    elif args.model == 4:
        model = GraphSH(adj, args.hid_dim, skeleton.joints_group(),
                        num_layers=args.num_layers, p_dropout=p_dropout,
                        gcn_type=args.gcn)
    else:
        raise ValueError(f'Unknown model index: {args.model}')

    return model.to(device)


def load_rehab_checkpoint(ckpt_path, model, angle_head, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['state_dict'], strict=False)
    angle_head.load_state_dict(ckpt['angle_head_state_dict'])
    print(f'Loaded rehab checkpoint epoch={ckpt.get("epoch","?")} '
          f'best_mae={ckpt.get("best_mae","?"):.2f} deg')


# ---------------------------------------------------------------------------
# 2D keypoint preprocessing (mirrors H3WB pipeline)
# ---------------------------------------------------------------------------

def preprocess_keypoints(kpts_2d: np.ndarray) -> np.ndarray:
    """
    kpts_2d: (133, 2) raw pixel coordinates from RTMPose
    Returns: (133, 2) normalized + hip-centred, matching H3WB preprocessing
    """
    # Root-center at hip midpoint (COCO joints 11 and 12)
    hip_center = (kpts_2d[11] + kpts_2d[12]) / 2.0
    centered = kpts_2d - hip_center

    # Scale: normalize by the 95th percentile distance from root
    norms = np.linalg.norm(centered, axis=-1)
    scale = np.percentile(norms[norms > 1e-3], 95) if (norms > 1e-3).any() else 1.0
    return (centered / max(scale, 1e-6)).astype(np.float32)


# ---------------------------------------------------------------------------
# RTMPose 2D detector wrapper
# ---------------------------------------------------------------------------

class RTMPoseDetector:
    """Wraps mmpose RTMPose-W for whole-body 133-joint detection."""

    def __init__(self, config_path: str, checkpoint_path: str, device: str):
        if not MMPOSE_AVAILABLE:
            raise RuntimeError(
                'mmpose is not installed. Install via:\n'
                '  pip install openmim && python -m mim install mmpose\n'
                'Or use --skip_pose_detector for testing without RTMPose.'
            )
        self.model = mmpose_init_model(config_path, checkpoint_path,
                                      device=device)

    def detect(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        frame_bgr: (H, W, 3) BGR numpy
        Returns: (133, 2) keypoints in pixel coordinates, or zeros if no person
        """
        result = inference_topdown(self.model, frame_bgr)
        if result and len(result) > 0:
            kpts = result[0].pred_instances.keypoints[0]  # (133, 2)
            return np.array(kpts, dtype=np.float32)
        return np.zeros((N_JOINTS, 2), dtype=np.float32)


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device('cuda:0')
    cudnn.benchmark = True

    # ---- Config + model ----
    cfg.merge_from_file(args.cfg)
    print('==> Building model...')
    skeleton = _get_skeleton()
    adj = adj_mx_from_skeleton(skeleton).to(device)
    model = build_backbone(args, cfg, adj, device)
    angle_head = ClinicalAngleHead(in_features=69, hidden=128).to(device)

    print(f'==> Loading rehab checkpoint: {args.checkpoint}')
    load_rehab_checkpoint(args.checkpoint, model, angle_head, device)

    model.eval()
    angle_head.eval()

    # ---- 2D detector ----
    detector = None
    if not args.skip_pose_detector:
        if not args.rtmpose_config or not args.rtmpose_checkpoint:
            raise ValueError(
                'Provide --rtmpose_config and --rtmpose_checkpoint, '
                'or use --skip_pose_detector to bypass RTMPose.'
            )
        print('==> Initialising RTMPose...')
        detector = RTMPoseDetector(
            args.rtmpose_config,
            args.rtmpose_checkpoint,
            device='cuda:0',
        )
    else:
        print('==> Skipping pose detector — feeding zero 2D keypoints.')

    # ---- One-Euro filter (create once, never recreate per frame) ----
    smoother = SkeletonFilter(n_joints=N_JOINTS, n_coords=3, freq=30.0)

    # ---- Video source ----
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video source: {args.source}')

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'==> Source: {args.source}  {frame_w}x{frame_h} @ {fps_src:.1f} fps')

    # ---- Video writer ----
    writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.save_video, fourcc, fps_src,
                                 (frame_w, frame_h))

    frame_idx = 0
    t_start = time.time()

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # ---- 2D keypoints ----
            if detector is not None:
                kpts_2d = detector.detect(frame)    # (133, 2) pixels
            else:
                kpts_2d = np.zeros((N_JOINTS, 2), dtype=np.float32)

            kpts_norm = preprocess_keypoints(kpts_2d)  # (133, 2) normalised

            # ---- 3D inference ----
            inp = torch.from_numpy(kpts_norm).unsqueeze(0).to(device)  # (1, 133, 2)
            body_3d, face_3d, lhand_3d, rhand_3d = model(inp)

            # Concatenate and hip-center (mirrors HRNet_GCN_WB.py evaluate)
            out_3d = torch.cat([body_3d, face_3d, lhand_3d, rhand_3d], dim=1)
            out_3d = out_3d - (out_3d[:, 11:12] + out_3d[:, 12:13]) / 2.0

            # ---- ROM angles ----
            pred_angles = angle_head(body_3d)                      # (1, 8)
            pred_angles = clamp_angles_to_valid_range(pred_angles)
            angles_np = pred_angles[0].cpu().numpy()               # (8,)

            # ---- One-Euro smoothing on body joints ----
            joints_np = out_3d[0, :23].cpu().numpy()              # (23, 3)
            joints_np = smoother(joints_np)

            # ---- Visualise ----
            annotated = draw_skeleton_on_frame(frame.copy(), joints_np, angles_np)

            if writer is not None:
                writer.write(annotated)
            else:
                cv2.imshow('GCADA — Rehab Inference', annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # ---- FPS logging ----
            frame_idx += 1
            if frame_idx % 30 == 0:
                elapsed = time.time() - t_start
                fps = frame_idx / elapsed
                print(f'  Frame {frame_idx}  FPS: {fps:.1f}')

    cap.release()
    if writer is not None:
        writer.release()
        print(f'Saved video to {args.save_video}')
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
