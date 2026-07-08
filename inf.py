"""
Cookie YOLOv11 Inference Script
================================
Usage:
  # Single image
  python3 inference.py --source image.jpg

  # Folder of images
  python3 inference.py --source test_images/

  # Webcam
  python3 inference.py --source 0

  # With custom confidence
  python3 inference.py --source image.jpg --conf 0.15

  # Save results
  python3 inference.py --source test_images/ --save
"""

import argparse
import cv2
import numpy as np
import os
import time
from pathlib import Path
from ultralytics import YOLO

# ─── CONFIG ───────────────────────────────────────────────────────────────────

MODEL_PATH = 'runs/segment/runs/segment/cookie_v5_fixed/weights/best.pt'
DEVICE     = 1          # GPU 1 (RTX 4090)
IMG_SIZE   = 1024
CONF_THRES = 0.15       # Low threshold — catches more class 8 detections
IOU_THRES  = 0.45

CLASS_NAMES = [
    'midnight_crc',        # 0
    'choc_walnut',         # 1
    'cinnamon_roll',       # 2
    'peanut_butter',       # 3
    'red_velvet',          # 4
    'classic_choc',        # 5
    'lotus_lava',          # 6
    'kunafa_dblchocolate', # 7
    'chocolate_hazelnut',  # 8
    'double_chocolate',    # 9
]

# Per-class colors (BGR)
CLASS_COLORS = [
    (180,  30,  30),   # 0  midnight_crc      — dark red
    (200, 120,  30),   # 1  choc_walnut        — brown
    (220, 180,  50),   # 2  cinnamon_roll      — golden
    (240, 200, 120),   # 3  peanut_butter      — light tan
    (180,  30, 100),   # 4  red_velvet         — crimson
    ( 60, 120, 200),   # 5  classic_choc       — blue
    ( 50, 180,  50),   # 6  lotus_lava         — green
    (130,  50, 180),   # 7  kunafa_dblchocolate— purple
    ( 30, 180, 180),   # 8  chocolate_hazelnut — teal
    ( 50,  50, 220),   # 9  double_chocolate   — navy
]

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def draw_results(image, result):
    """Draw segmentation masks + labels on image."""
    annotated = image.copy()
    h, w = annotated.shape[:2]

    if result.masks is None:
        return annotated, []

    detections = []

    masks  = result.masks.data.cpu().numpy()   # (N, H, W)
    boxes  = result.boxes
    clsids = boxes.cls.cpu().numpy().astype(int)
    confs  = boxes.conf.cpu().numpy()
    xyxys  = boxes.xyxy.cpu().numpy().astype(int)

    # Overlay masks
    overlay = annotated.copy()
    for i, (mask, cls_id, conf) in enumerate(zip(masks, clsids, confs)):
        color = CLASS_COLORS[cls_id]

        # Resize mask to image size
        mask_resized = cv2.resize(
            mask.astype(np.uint8),
            (w, h),
            interpolation=cv2.INTER_NEAREST
        )
        # Fill mask area
        overlay[mask_resized == 1] = color

    # Blend overlay
    cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0, annotated)

    # Draw boxes + labels
    for i, (cls_id, conf, xyxy) in enumerate(zip(clsids, confs, xyxys)):
        x1, y1, x2, y2 = xyxy
        color = CLASS_COLORS[cls_id]
        name  = CLASS_NAMES[cls_id]
        label = f"{name} {conf:.2f}"

        # Box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)

        # Label text
        cv2.putText(
            annotated, label,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (255, 255, 255), 1, cv2.LINE_AA
        )

        detections.append({
            'class_id':   cls_id,
            'class_name': name,
            'confidence': float(conf),
            'bbox':       [int(x1), int(y1), int(x2), int(y2)],
        })

    return annotated, detections


def print_detection_summary(detections, img_name=''):
    """Print per-image detection summary."""
    if img_name:
        print(f"\n{'─'*50}")
        print(f"Image: {img_name}")
        print(f"{'─'*50}")

    if not detections:
        print("  No cookies detected.")
        return

    # Count per class
    from collections import Counter
    counts = Counter(d['class_name'] for d in detections)

    print(f"  Total detections: {len(detections)}")
    for name, count in sorted(counts.items()):
        # Get confidence range for this class
        class_confs = [d['confidence'] for d in detections if d['class_name'] == name]
        avg_conf = sum(class_confs) / len(class_confs)

        # Flag class 5/8 confusion risk
        flag = ''
        if name == 'chocolate_hazelnut' and avg_conf < 0.40:
            flag = '  ⚠ Low conf — possible class 5/8 ambiguity'
        elif name == 'classic_choc' and avg_conf < 0.40:
            flag = '  ⚠ Low conf — possible class 5/8 ambiguity'

        print(f"  {name:<25} x{count}  (avg conf: {avg_conf:.2f}){flag}")


# ─── MAIN INFERENCE ───────────────────────────────────────────────────────────

def run_inference(
    source,
    model_path=MODEL_PATH,
    conf=CONF_THRES,
    iou=IOU_THRES,
    imgsz=IMG_SIZE,
    device=DEVICE,
    save=False,
    save_dir='outputs/inference',
    show=False,
):
    print(f"\nLoading model: {model_path}")
    model = YOLO(model_path)
    print(f"Model loaded. Running on device {device}")
    print(f"Conf threshold: {conf} | IoU: {iou} | imgsz: {imgsz}\n")

    if save:
        os.makedirs(save_dir, exist_ok=True)
        print(f"Results will be saved to: {save_dir}/\n")

    # ── Collect sources ──────────────────────────────────────────────────────
    source = str(source)

    # Webcam
    if source.isdigit():
        _run_webcam(model, int(source), conf, iou, imgsz, device, show)
        return

    # Single image
    if os.path.isfile(source):
        sources = [source]
    # Directory
    elif os.path.isdir(source):
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        sources = [
            str(p) for p in Path(source).iterdir()
            if p.suffix.lower() in exts
        ]
        sources.sort()
        print(f"Found {len(sources)} images in {source}")
    else:
        print(f"ERROR: Source not found: {source}")
        return

    # ── Process images ───────────────────────────────────────────────────────
    all_detections = []
    total_time = 0

    for img_path in sources:
        image = cv2.imread(img_path)
        if image is None:
            print(f"WARNING: Could not read {img_path}, skipping.")
            continue

        t0 = time.time()
        results = model.predict(
            source=image,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )
        elapsed = time.time() - t0
        total_time += elapsed

        result    = results[0]
        annotated, detections = draw_results(image, result)

        fname = Path(img_path).name
        print_detection_summary(detections, fname)
        print(f"  Inference time: {elapsed*1000:.1f} ms")

        all_detections.extend(detections)

        if save:
            out_path = os.path.join(save_dir, f"pred_{fname}")
            cv2.imwrite(out_path, annotated)

        if show:
            cv2.imshow('Cookie Detection', annotated)
            key = cv2.waitKey(0)
            if key == ord('q'):
                break

    cv2.destroyAllWindows()

    # ── Final summary ────────────────────────────────────────────────────────
    if len(sources) > 1:
        print(f"\n{'='*50}")
        print(f"OVERALL SUMMARY — {len(sources)} images")
        print(f"{'='*50}")
        print_detection_summary(all_detections)
        print(f"\nAvg inference time: {(total_time/len(sources))*1000:.1f} ms/image")

        # Class 5/8 confusion warning
        cls5 = [d for d in all_detections if d['class_id'] == 5]
        cls8 = [d for d in all_detections if d['class_id'] == 8]
        if cls5 or cls8:
            low5 = [d for d in cls5 if d['confidence'] < 0.40]
            low8 = [d for d in cls8 if d['confidence'] < 0.40]
            if low5 or low8:
                print(f"\n⚠  Class 5/8 low-confidence detections:")
                print(f"   classic_choc      < 0.40 conf: {len(low5)} detections")
                print(f"   chocolate_hazelnut < 0.40 conf: {len(low8)} detections")
                print(f"   These may be misclassified — manual review recommended.")

    if save:
        print(f"\nSaved results: {save_dir}/")


# ─── WEBCAM MODE ──────────────────────────────────────────────────────────────

def _run_webcam(model, cam_id, conf, iou, imgsz, device, show):
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {cam_id}")
        return

    print("Webcam running — press Q to quit\n")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(
            source=frame,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )
        annotated, detections = draw_results(frame, results[0])

        # FPS overlay
        fps_label = f"Detections: {len(detections)}"
        cv2.putText(annotated, fps_label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow('Cookie Detection — Live', annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='Cookie YOLOv11 Inference')
    parser.add_argument('--source',     type=str, required=True,
                        help='Image path, folder path, or camera index (0)')
    parser.add_argument('--model',      type=str, default=MODEL_PATH,
                        help='Path to best.pt weights')
    parser.add_argument('--conf',       type=float, default=CONF_THRES,
                        help='Confidence threshold (default: 0.15)')
    parser.add_argument('--iou',        type=float, default=IOU_THRES,
                        help='IoU threshold (default: 0.45)')
    parser.add_argument('--imgsz',      type=int,   default=IMG_SIZE,
                        help='Inference image size (default: 1024)')
    parser.add_argument('--device',     type=int,   default=DEVICE,
                        help='GPU device (default: 1)')
    parser.add_argument('--save',       action='store_true',
                        help='Save annotated images to outputs/inference/')
    parser.add_argument('--save-dir',   type=str, default='outputs/inference',
                        help='Output directory for saved results')
    parser.add_argument('--show',       action='store_true',
                        help='Display results in window (requires display)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_inference(
        source    = args.source,
        model_path= args.model,
        conf      = args.conf,
        iou       = args.iou,
        imgsz     = args.imgsz,
        device    = args.device,
        save      = args.save,
        save_dir  = args.save_dir,
        show      = args.show,
    )