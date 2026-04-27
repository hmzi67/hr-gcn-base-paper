"""
Draw 3D skeleton projected to 2D on an RGB frame.
Color code: purple = cervical, teal = torso, coral = lower body.
Shows ROM angle labels next to relevant joints and a dashboard panel.
"""
import cv2
import numpy as np

# Colors BGR
COLOR_CERVICAL = (221, 119, 127)   # purple #7F77DD
COLOR_TORSO    = (117, 158,  29)   # teal   #1D9E75
COLOR_LOWER    = ( 48,  90, 216)   # coral  #D85A30

JOINT_COLORS = {}
for _i in range(5):     JOINT_COLORS[_i]  = COLOR_CERVICAL   # head/neck
for _i in range(5, 17): JOINT_COLORS[_i]  = COLOR_TORSO      # torso + arms
for _i in range(17, 23):JOINT_COLORS[_i]  = COLOR_LOWER      # lower body

# COCO body skeleton edges (joint_a, joint_b)
SKELETON_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # face/head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), # arms
    (5, 11), (6, 12), (11, 12),               # torso
    (11, 13), (13, 15),                       # left leg
    (12, 14), (14, 16),                       # right leg
]

ANGLE_NAMES = [
    'Cerv Yaw', 'Cerv Pitch', 'Cerv Roll',
    'Trunk', 'L Hip', 'R Hip', 'L Knee', 'R Knee',
]

ROM_LABEL_CONFIG = {
    # joint_idx: (label, angle_index_in_output)
    13: ('L Knee', 6),
    14: ('R Knee', 7),
    11: ('L Hip',  4),
    12: ('R Hip',  5),
    0:  ('Cerv P', 1),   # cervical pitch shown at nose
}


def project_3d_to_2d(joints_3d: np.ndarray, frame_h: int, frame_w: int,
                     scale: float = 300.0) -> np.ndarray:
    """Simple orthographic projection centered on frame."""
    joints_2d = joints_3d[:, :2].copy()
    joints_2d[:, 0] = joints_2d[:, 0] * scale + frame_w // 2
    joints_2d[:, 1] = -joints_2d[:, 1] * scale + frame_h // 2
    return joints_2d.astype(int)


def draw_skeleton_on_frame(frame: np.ndarray,
                           joints_3d_body: np.ndarray,
                           rom_angles: np.ndarray = None) -> np.ndarray:
    """
    frame:           BGR numpy (H, W, 3)
    joints_3d_body:  (23, 3) body joints from HR-GCN (meters, hip-centred)
    rom_angles:      (8,) float degrees, or None
    returns:         annotated frame (in-place)
    """
    H, W = frame.shape[:2]
    j2d = project_3d_to_2d(joints_3d_body, H, W)

    # Draw bones
    for (a, b) in SKELETON_EDGES:
        if a >= len(j2d) or b >= len(j2d):
            continue
        color = JOINT_COLORS.get(a, COLOR_TORSO)
        pt1 = tuple(np.clip(j2d[a], [0, 0], [W - 1, H - 1]).tolist())
        pt2 = tuple(np.clip(j2d[b], [0, 0], [W - 1, H - 1]).tolist())
        cv2.line(frame, pt1, pt2, color, 2, cv2.LINE_AA)

    # Draw joints
    for idx, (x, y) in enumerate(j2d):
        x = int(np.clip(x, 0, W - 1))
        y = int(np.clip(y, 0, H - 1))
        color = JOINT_COLORS.get(idx, COLOR_TORSO)
        cv2.circle(frame, (x, y), 5, color, -1, cv2.LINE_AA)

    # Per-joint ROM labels
    if rom_angles is not None:
        for j_idx, (label, a_idx) in ROM_LABEL_CONFIG.items():
            if j_idx >= len(j2d):
                continue
            x = int(j2d[j_idx][0]) - 50
            y = int(j2d[j_idx][1])
            x = max(x, 5)
            cv2.putText(frame, f'{label}: {float(rom_angles[a_idx]):.1f}',
                        (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # Dashboard panel (top-right corner)
    if rom_angles is not None:
        panel_x, panel_y = W - 200, 10
        for i, (name, val) in enumerate(zip(ANGLE_NAMES, rom_angles)):
            y = panel_y + i * 22
            cv2.putText(frame, f'{name}: {val:+.1f}',
                        (panel_x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, (200, 200, 200), 1, cv2.LINE_AA)

    return frame
