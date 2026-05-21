"""
Dual-Stream Rehabilitation Architecture Diagram  v2
Master's Thesis — Publication Quality
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import os

# ── Canvas ────────────────────────────────────────────────────────────────────
FW, FH = 28, 17
fig, ax = plt.subplots(figsize=(FW, FH))
ax.set_xlim(0, FW); ax.set_ylim(0, FH)
ax.axis('off')
fig.patch.set_facecolor('white')

# ── Palette ───────────────────────────────────────────────────────────────────
C = dict(
    blue_fc='#DBEAFE',  blue_ec='#1D4ED8',
    purple_fc='#EDE9FE', purple_ec='#6D28D9',
    amber_fc='#FEF3C7',  amber_ec='#B45309',
    teal_fc='#CCFBF1',   teal_ec='#0F766E',
    green_fc='#DCFCE7',  green_ec='#15803D',
    gray_fc='#F1F5F9',   gray_ec='#475569',
    red_fc='#FEE2E2',    red_ec='#B91C1C',
)

# ── Helpers ───────────────────────────────────────────────────────────────────
def rbox(ax, x, y, w, h, title, sub='',
         fc='#DBEAFE', ec='#1D4ED8', lw=1.8,
         tfs=9, sfs=7.5, title_color=None, rad=0.22):
    patch = FancyBboxPatch((x, y), w, h,
                           boxstyle=f'round,pad=0,rounding_size={rad}',
                           fc=fc, ec=ec, lw=lw, zorder=3, clip_on=False)
    ax.add_patch(patch)
    tc = title_color or ec
    yo = y + h/2 + (0.17 if sub else 0)
    ax.text(x+w/2, yo, title, ha='center', va='center',
            fontsize=tfs, fontweight='bold', color=tc, zorder=4, clip_on=False)
    if sub:
        ax.text(x+w/2, y+h/2-0.25, sub, ha='center', va='center',
                fontsize=sfs, color='#374151', zorder=4, style='italic',
                multialignment='center', clip_on=False)

def arrow(ax, x0, y0, x1, y1, label='', ec='#475569',
          lw=1.5, cs='arc3,rad=0.0', lfs=7.2, lpad=0.12):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0), zorder=5,
                arrowprops=dict(arrowstyle='->', color=ec, lw=lw,
                                connectionstyle=cs))
    if label:
        mx = (x0+x1)/2; my = (y0+y1)/2
        ax.text(mx+lpad, my, label, ha='left', va='center',
                fontsize=lfs, color=ec, zorder=6,
                bbox=dict(boxstyle='round,pad=0.12', fc='white',
                          ec='none', alpha=0.9))

def slabel(ax, x, y, txt, fc='#475569'):
    ax.text(x, y, txt, ha='center', va='center', fontsize=7.8,
            color='white', fontweight='bold', zorder=6,
            bbox=dict(boxstyle='round,pad=0.28', fc=fc, ec='none', alpha=0.95))

# ══════════════════════════════════════════════════════════════════════════════
# TITLE BAR
# ══════════════════════════════════════════════════════════════════════════════
ax.text(FW/2, 16.55,
        'Dual-Stream Rehabilitation Exercise Quality Assessment via GCADA-Enhanced Spatiotemporal GCN',
        ha='center', va='center', fontsize=14, fontweight='bold', color='#0F172A', zorder=6)
ax.text(FW/2, 16.1,
        "Master's Thesis  ·  Full System Architecture",
        ha='center', va='center', fontsize=9.5, color='#64748B', zorder=6, style='italic')
ax.plot([0.5, FW-0.5], [15.75, 15.75], color='#CBD5E1', lw=1.0, zorder=2)

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — INPUT   x=0.35 … 3.05
# ══════════════════════════════════════════════════════════════════════════════
slabel(ax, 1.70, 15.42, 'STAGE 1 — INPUT', '#475569')

rbox(ax, 0.35, 12.5, 2.7, 2.7,
     'UI-PRMD Dataset',
     'Vicon Motion Capture\n10 exercises × 10 subjects\n× 10 reps = 1,000 reps\n133 whole-body joints',
     fc=C['blue_fc'], ec=C['blue_ec'])

# ── two arrows from input box ──────────────────────────────────────────────
arrow(ax, 3.05, 14.45, 4.2, 14.45,
      '2D Keypoints (N, 133, 2)', ec=C['blue_ec'])
arrow(ax, 3.05, 13.2, 4.2, 13.2, ec=C['blue_ec'])
ax.text(3.1, 12.92, '117-dim Angle Files', ha='left', va='center',
        fontsize=7.2, color=C['blue_ec'],
        bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='none', alpha=0.9))

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — QUALITY SCORE GENERATION   (top lane, x=4.2 … 11.5, y≈13.5)
# ══════════════════════════════════════════════════════════════════════════════
slabel(ax, 7.5, 15.42, 'STAGE 2 — QUALITY SCORE GENERATION  (Offline)', C['amber_ec'])

rbox(ax, 4.2, 12.7, 2.6, 1.35,
     'Segmented Reps',
     '990 reps\n(10 ex × 10 subj × ~9 avg)',
     fc=C['amber_fc'], ec=C['amber_ec'])
arrow(ax, 6.8, 13.38, 7.8, 13.38, ec=C['amber_ec'])

rbox(ax, 7.8, 12.7, 2.6, 1.35,
     'PCA + GMM',
     '20 PCA components\nn_components=4 / exercise\ndiag covariance, 300 iter',
     fc=C['amber_fc'], ec=C['amber_ec'])
arrow(ax, 10.4, 13.38, 11.4, 13.38, ec=C['amber_ec'])

rbox(ax, 11.4, 12.7, 2.6, 1.35,
     'Quality Scores  [0, 1]',
     'GMM log-likelihood\nnorm. & clipped\nAvg gap = 0.091 (train)',
     fc=C['amber_fc'], ec=C['amber_ec'])

# Quality score drops down to fusion
arrow(ax, 12.7, 12.7, 12.7, 8.8,
      'gmm_score', ec=C['amber_ec'])

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — DUAL STREAM
# ══════════════════════════════════════════════════════════════════════════════
slabel(ax, 8.3, 12.2, 'STAGE 3 — DUAL STREAM', '#475569')

# ─── STREAM 1 — Spatiotemporal GCN   x≈4.2 … 7.0, y=5.5 … 11.6 ─────────────
sx1 = 4.2
slabel(ax, sx1+1.35, 11.85, 'Stream 1 — Spatiotemporal GCN', C['purple_ec'])

# Keypoint input arrives at top of stream 1
arrow(ax, sx1+1.35, 14.45, sx1+1.35, 11.55, ec=C['blue_ec'],
      cs='arc3,rad=0.0')

rbox(ax, sx1, 10.4, 2.7, 1.0,
     'Sliding Window',
     'window = 100 frames\nstride = 50',
     fc=C['purple_fc'], ec=C['purple_ec'])
arrow(ax, sx1+1.35, 10.4, sx1+1.35, 10.18, ec=C['purple_ec'])

rbox(ax, sx1, 8.6, 2.7, 1.65,
     'Spatial GCN  ★',
     '10 per-exercise learnable\nadjacency matrices\nROM-guided initialisation\nk=2 layers  ·  3→64→128 ch',
     fc=C['purple_fc'], ec=C['purple_ec'])
arrow(ax, sx1+1.35, 8.6, sx1+1.35, 8.38, ec=C['purple_ec'])

rbox(ax, sx1, 6.8, 2.7, 1.65,
     'Temporal GCN',
     'Learnable Gaussian adj.\nA[i,j]=exp(−|i−j|²/σ²)\nM×M window matrix\n128-dim output',
     fc=C['purple_fc'], ec=C['purple_ec'])
arrow(ax, sx1+1.35, 6.8, sx1+1.35, 6.58, ec=C['purple_ec'])

rbox(ax, sx1, 5.9, 2.7, 0.6,
     '3D Feature  (128-dim)',
     '', fc='#EDE9FE', ec=C['purple_ec'], tfs=8.8)

# ─── STREAM 2 — ROM BiLSTM   x≈8.7 … 11.5, y=5.5 … 11.6 ────────────────────
sx2 = 8.7
slabel(ax, sx2+1.35, 11.85, 'Stream 2 — ROM BiLSTM', C['amber_ec'])

# Angle file input arrives at top of stream 2
arrow(ax, 3.05, 13.2, sx2+1.35, 13.2, ec=C['blue_ec'],
      cs='arc3,rad=0.0')
arrow(ax, sx2+1.35, 13.2, sx2+1.35, 11.55, ec=C['blue_ec'],
      cs='arc3,rad=0.0')

rbox(ax, sx2, 10.4, 2.7, 1.0,
     'ROM Angles  (N, 12)',
     '12 clinical angles\nknee, hip, shoulder, ankle…',
     fc=C['amber_fc'], ec=C['amber_ec'])
arrow(ax, sx2+1.35, 10.4, sx2+1.35, 10.18, ec=C['amber_ec'])

rbox(ax, sx2, 8.7, 2.7, 1.55,
     'Linear Projection',
     '12 → 64\nReLU + Dropout(0.3)\n(per-frame encoding)',
     fc=C['amber_fc'], ec=C['amber_ec'])
arrow(ax, sx2+1.35, 8.7, sx2+1.35, 8.48, ec=C['amber_ec'])

rbox(ax, sx2, 6.8, 2.7, 1.75,
     'BiLSTM + Attention  ★',
     'hidden = 64, bidirectional\n2 stacked layers\nBahdanau attention pooling\n→ 128-dim context vector',
     fc=C['amber_fc'], ec=C['amber_ec'])
arrow(ax, sx2+1.35, 6.8, sx2+1.35, 6.58, ec=C['amber_ec'])

rbox(ax, sx2, 5.9, 2.7, 0.6,
     'ROM Feature  (128-dim)',
     '', fc='#FEF3C7', ec=C['amber_ec'], tfs=8.8)

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — FUSION   x=12.0, y=6.4 … 9.4
# ══════════════════════════════════════════════════════════════════════════════
fx, fy = 12.0, 6.4
slabel(ax, fx+1.45, 9.55, 'STAGE 4 — FUSION  ★', C['teal_ec'])

rbox(ax, fx, fy, 2.9, 2.2,
     'Fusion Layer  ★',
     'Concat(128+128) = 256-dim\nFC(256→128) + ReLU\nLayerNorm + Dropout(0.2)\n⊕ gmm_score (scalar inject)',
     fc=C['teal_fc'], ec=C['teal_ec'])

# Stream outputs → fusion
arrow(ax, sx1+2.7, 6.2, fx, 7.5, ec=C['purple_ec'], cs='arc3,rad=-0.1')
arrow(ax, sx2+2.7, 6.2, fx, 7.5, ec=C['amber_ec'], cs='arc3,rad=0.1')

rbox(ax, fx, 5.45, 2.9, 0.75,
     'Shared Feature  (128-dim)',
     '', fc='#CCFBF1', ec=C['teal_ec'], tfs=9)
arrow(ax, fx+1.45, 6.4, fx+1.45, 6.2, ec=C['teal_ec'])

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 5 — OUTPUT HEADS   x=15.6 …
# ══════════════════════════════════════════════════════════════════════════════
slabel(ax, 19.5, 9.55, 'STAGE 5 — OUTPUT HEADS', C['green_ec'])

# Fan-out: shared feature → 3 heads
# We draw an explicit vertical distribution bar
dist_x = 15.4
arrow(ax, fx+2.9, 5.82, dist_x, 7.8,  ec=C['teal_ec'], cs='arc3,rad=-0.2')
arrow(ax, fx+2.9, 5.82, dist_x, 5.82, ec=C['teal_ec'], cs='arc3,rad=0.0')
arrow(ax, fx+2.9, 5.82, dist_x, 3.85, ec=C['teal_ec'], cs='arc3,rad=0.2')

# --- Head 1: Exercise Classifier ---
rbox(ax, dist_x, 7.15, 2.9, 1.4,
     'Exercise Classifier',
     'FC(128→64→10)\nCrossEntropy + label_smooth\nDropout(0.4)',
     fc=C['green_fc'], ec=C['green_ec'])
arrow(ax, dist_x+2.9, 7.85, dist_x+4.0, 7.85, ec=C['green_ec'])
rbox(ax, dist_x+4.0, 7.4, 3.1, 0.95,
     'Exercise ID  (0–9)',
     '10 classes  ·  98.88% Accuracy',
     fc=C['green_fc'], ec=C['green_ec'], tfs=9)

# --- Head 2: Validity Classifier ---
rbox(ax, dist_x, 5.2, 2.9, 1.4,
     'Validity Classifier',
     'FC(128→64→2)\nBinary CrossEntropy\nValid / Invalid',
     fc=C['green_fc'], ec=C['green_ec'])
arrow(ax, dist_x+2.9, 5.9, dist_x+4.0, 5.9, ec=C['green_ec'])
rbox(ax, dist_x+4.0, 5.45, 3.1, 0.95,
     'Valid / Invalid',
     '2 classes  ·  99.58% Accuracy',
     fc=C['green_fc'], ec=C['green_ec'], tfs=9)

# --- Head 3: Quality Regression ---
rbox(ax, dist_x, 3.25, 2.9, 1.4,
     'Quality Head',
     'FC(128→64→1)\nSigmoid activation\nL1 regression loss',
     fc=C['green_fc'], ec=C['green_ec'])
arrow(ax, dist_x+2.9, 3.95, dist_x+4.0, 3.95, ec=C['green_ec'])
rbox(ax, dist_x+4.0, 3.5, 3.1, 0.95,
     'Quality Score  [0 → 1]',
     'Regression  ·  MAD = 0.009',
     fc=C['green_fc'], ec=C['green_ec'], tfs=9)

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 6 — COMBINED LOSS   (bottom strip across head section)
# ══════════════════════════════════════════════════════════════════════════════
rbox(ax, 15.4, 0.75, 12.1, 2.2,
     'End-to-End Joint Training  ★',
     'L_total  =  L_quality  +  0.5 × L_exercise  +  0.3 × L_validity\n'
     'All three heads optimised simultaneously in a single forward pass\n'
     'Novel contribution: paper (STGCN-Seq) used staged per-task training',
     fc=C['gray_fc'], ec=C['gray_ec'], tfs=9.5, sfs=8.2)
arrow(ax, 16.85, 3.25, 16.85, 2.95, ec=C['gray_ec'])
arrow(ax, 18.85, 3.25, 18.85, 2.95, ec=C['gray_ec'])
arrow(ax, 20.95, 3.25, 20.95, 2.95, ec=C['gray_ec'])

# ══════════════════════════════════════════════════════════════════════════════
# NOVEL CONTRIBUTIONS LEGEND   (bottom-left)
# ══════════════════════════════════════════════════════════════════════════════
lx, ly, lw, lh = 0.35, 0.5, 9.2, 4.6
p = FancyBboxPatch((lx, ly), lw, lh,
                   boxstyle='round,pad=0,rounding_size=0.22',
                   fc='#F8FAFC', ec='#94A3B8', lw=1.3, zorder=3)
ax.add_patch(p)
ax.text(lx+lw/2, ly+lh-0.28, 'NOVEL CONTRIBUTIONS',
        ha='center', va='center', fontsize=9.5, fontweight='bold',
        color='#1E293B', zorder=4)
ax.plot([lx+0.2, lx+lw-0.2], [ly+lh-0.52, ly+lh-0.52],
        color='#CBD5E1', lw=0.9, zorder=4)

items = [
    ('★  ROM-guided adjacency initialisation',
     'Clinical joint correlations pre-initialise the 10 per-exercise learnable\n'
     'adjacency matrices in the Spatial GCN layer.'),
    ('★  Dual-stream fusion',
     'First method to jointly fuse 3D spatiotemporal GCN features with\n'
     'explicit clinical ROM angle sequences for UI-PRMD assessment.'),
    ('★  BiLSTM + Bahdanau attention on ROM stream',
     'Temporal attention mechanism over 12 clinical angle sequences\n'
     'produces a context-aware 128-dim ROM feature vector.'),
    ('★  End-to-end joint training',
     'Exercise classification, validity detection, and quality regression\n'
     'are trained simultaneously; prior work used staged training.'),
]
for i, (ttl, desc) in enumerate(items):
    iy = ly + lh - 0.72 - i * 0.98
    ax.text(lx+0.22, iy, ttl,
            ha='left', va='top', fontsize=8.2, fontweight='bold',
            color=C['teal_ec'], zorder=4)
    ax.text(lx+0.32, iy-0.27, desc,
            ha='left', va='top', fontsize=7.2, color='#475569',
            zorder=4, style='italic')

# ══════════════════════════════════════════════════════════════════════════════
# COMPARISON TABLE   (top-right)
# ══════════════════════════════════════════════════════════════════════════════
tx, ty = 23.1, 14.7
ax.text(tx, ty+0.35, 'Comparison vs. STGCN-Seq',
        ha='left', va='bottom', fontsize=9, fontweight='bold', color='#1E293B')

rows = [
    ('Method',        'MAD',   'EC Acc'),
    ('STGCN-Seq',     '0.009', '96.0%'),
    ('Ours  (Dual)',  '0.009', '98.9%'),
]
cw = [2.0, 0.95, 1.0]; rh = 0.42
for r, row in enumerate(rows):
    for c, cell in enumerate(row):
        cx2 = tx + sum(cw[:c])
        cy2 = ty - r*rh
        hdr = r == 0; win = r == 2
        fc2 = '#CBD5E1' if hdr else ('#DCFCE7' if win else 'white')
        ec2 = '#64748B'
        rp = FancyBboxPatch((cx2, cy2-rh+0.05), cw[c]-0.06, rh-0.07,
                            boxstyle='square,pad=0', fc=fc2, ec=ec2,
                            lw=0.8, zorder=3)
        ax.add_patch(rp)
        fw = 'bold' if (hdr or (win and c > 0)) else 'normal'
        cc = C['green_ec'] if (win and c > 0) else '#1E293B'
        ax.text(cx2+(cw[c]-0.06)/2, cy2-rh/2+0.05, cell,
                ha='center', va='center', fontsize=8,
                fontweight=fw, color=cc, zorder=4)

ax.text(tx, ty - 3*rh - 0.1,
        '▲ Beat paper:  Ex07, Ex08, Ex10\n● Tied paper:  Overall MAD',
        ha='left', va='top', fontsize=7.5, color=C['green_ec'], zorder=4)

# ══════════════════════════════════════════════════════════════════════════════
# COLOUR KEY   (bottom strip between legend and comparison)
# ══════════════════════════════════════════════════════════════════════════════
chips = [
    (C['blue_fc'],   C['blue_ec'],   'Input Data'),
    (C['purple_fc'], C['purple_ec'], 'Spatiotemporal GCN'),
    (C['amber_fc'],  C['amber_ec'],  'ROM / Quality Stream'),
    (C['teal_fc'],   C['teal_ec'],   'Fusion'),
    (C['green_fc'],  C['green_ec'],  'Output Heads'),
    (C['gray_fc'],   C['gray_ec'],   'Loss / Training'),
]
chip_x0, chip_y0 = 9.6, 0.55
for i, (fc2, ec2, lbl) in enumerate(chips):
    cx2 = chip_x0 + i*2.15
    rp = FancyBboxPatch((cx2, chip_y0), 0.35, 0.3,
                        boxstyle='round,pad=0,rounding_size=0.07',
                        fc=fc2, ec=ec2, lw=1.0, zorder=3)
    ax.add_patch(rp)
    ax.text(cx2+0.42, chip_y0+0.15, lbl,
            ha='left', va='center', fontsize=7.5, color='#374151', zorder=4)

# ══════════════════════════════════════════════════════════════════════════════
# LIGHT SECTION DIVIDERS
# ══════════════════════════════════════════════════════════════════════════════
for xv, y0, y1 in [(3.8, 0.5, 15.7), (14.6, 0.5, 15.7)]:
    ax.plot([xv, xv], [y0, y1], color='#E2E8F0', lw=0.9,
            linestyle='--', zorder=1)

# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════
os.makedirs('results', exist_ok=True)
out = 'results/thesis_architecture_diagram.png'
plt.savefig(out, dpi=220, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f'Saved → {out}')
