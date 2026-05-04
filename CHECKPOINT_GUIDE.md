# HR-GCN Checkpoint Guide

## Available Checkpoints for Live Inference

All checkpoints are located in `checkpoint/HRGCN/` with subdirectories organized by GCN variant and training date.

### Best HRGCN (Model=1, dc_preagg) Checkpoint:
```bash
checkpoint/HRGCN/dc_preagg-2026-04-24T14:41:39/ckpt_best.pth.tar
```
- **GCN Variant**: `dc_preagg` (Decoupled Pre-Aggregation)
- **Training Date**: 2026-04-24 14:41:39
- **File Size**: 252 MB
- **Training Params**:
  - Epochs: 100
  - Batch size: 256
  - Learning rate: 0.01
  - Model: 1 (HRGCN with multi-branch)

### How to Use with infer_live.py

#### Testing without RTMPose (zero keypoints):
```bash
python infer_live.py \
    --checkpoint checkpoint/HRGCN/dc_preagg-2026-04-24T14:41:39/ckpt_best.pth.tar \
    --cfg w32_adam_lr1e-3.yaml \
    --gcn dc_preagg \
    --skip_pose_detector
```

#### With RTMPose (full pipeline from webcam):
```bash
python infer_live.py \
    --checkpoint checkpoint/HRGCN/dc_preagg-2026-04-24T14:41:39/ckpt_best.pth.tar \
    --cfg w32_adam_lr1e-3.yaml \
    --gcn dc_preagg \
    --rtmpose_config <path_to_rtmpose_config> \
    --rtmpose_checkpoint <path_to_rtmpose_weights>
```

#### From video file:
```bash
python infer_live.py \
    --source video.mp4 \
    --checkpoint checkpoint/HRGCN/dc_preagg-2026-04-24T14:41:39/ckpt_best.pth.tar \
    --cfg w32_adam_lr1e-3.yaml \
    --gcn dc_preagg \
    --skip_pose_detector \
    --save_video output.mp4
```

#### CPU-only (no GPU):
```bash
python infer_live.py \
    --checkpoint checkpoint/HRGCN/dc_preagg-2026-04-24T14:41:39/ckpt_best.pth.tar \
    --cfg w32_adam_lr1e-3.yaml \
    --gcn dc_preagg \
    --skip_pose_detector \
    --device cpu
```

## Other Available Checkpoints

### dc_vanilla variant (for comparison):
- `checkpoint/HRGCN/dc_vanilla-2026-04-23T17:14:45/ckpt_best.pth.tar` (188 MB)
- `checkpoint/HRGCN/dc_vanilla-2026-04-23T17:16:56/ckpt_best.pth.tar` (188 MB)

Use with: `--gcn dc_vanilla`

## Dependencies Required

```bash
pip install opencv-python
pip install openmim
python -m mim install mmpose  # Only if using --rtmpose_config
```

## RTMPose Setup

For end-to-end inference with 2D pose detection, download RTMPose-W files:

```python
# Auto-download and cache (recommended)
from mmpose.apis import init_model
model = init_model('rtmpose-w_8xb256-270e_coco-wholebody-384x288.py', 
                   'rtmpose-w_sim-coco_270e_384x288-4d6dfc6d_20230124.pth',
                   device='cuda:0')
# Files cached to ~/.cache/mmpose/checkpoints/
```

Or manually download:
- Config: `rtmpose-w_8xb256-270e_coco-wholebody-384x288.py`
- Weights: `https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-w_sim-coco_270e_384x288-4d6dfc6d_20230124.pth`

## Inference Output

The `infer_live.py` script displays:
- **133 joints** (whole-body): body (23), face (68), hands (21+21)
- **Colors**: green (body), blue-grey (face), orange (hands)
- **Visualization**: Real-time skeleton overlay on frames
- **FPS counter**: Exponential moving average displayed on-screen
- **Controls**: Press **Q** to quit

## Troubleshooting

| Error | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'cv2'` | `pip install opencv-python` |
| `RuntimeError: state_dict mismatch` | Ensure `--gcn` matches the checkpoint (check params.json in checkpoint directory) |
| `CUDA out of memory` | Use `--device cpu` or reduce batch processing |
| RTMPose not found | Install mmpose: `python -m mim install mmpose` |
