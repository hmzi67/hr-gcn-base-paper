"""
Build novel_vs_baseline_comparison.csv from two sets of evaluate_comparison.py outputs.

Usage:
    python make_novel_vs_baseline_comparison.py \
        --baseline_dir  results_comparison_baseline \
        --novel_dir     results_comparison_novel \
        --output        novel_vs_baseline_comparison.csv

The script merges per_joint_mae.csv and per_exercise_mae.csv from each directory,
adds delta columns, and prints a summary table.
"""
import argparse
import csv
import os
import sys


def load_per_joint(csv_path):
    rows = {}
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            rows[row['Joint']] = row
    return rows


def load_per_exercise(csv_path):
    rows = {}
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            rows[row['Exercise']] = row
    return rows


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline_dir',  default='results_comparison_baseline')
    p.add_argument('--novel_dir',     default='results_comparison_novel')
    p.add_argument('--output',        default='novel_vs_baseline_comparison.csv')
    return p.parse_args()


def main():
    args = parse_args()

    base_joint_path = os.path.join(args.baseline_dir, 'per_joint_mae.csv')
    nov_joint_path  = os.path.join(args.novel_dir,    'per_joint_mae.csv')
    base_ex_path    = os.path.join(args.baseline_dir, 'per_exercise_mae.csv')
    nov_ex_path     = os.path.join(args.novel_dir,    'per_exercise_mae.csv')

    for p in [base_joint_path, nov_joint_path, base_ex_path, nov_ex_path]:
        if not os.path.isfile(p):
            sys.exit(f'ERROR: missing file: {p}')

    base_joint = load_per_joint(base_joint_path)
    nov_joint  = load_per_joint(nov_joint_path)
    base_ex    = load_per_exercise(base_ex_path)
    nov_ex     = load_per_exercise(nov_ex_path)

    rows_out = []

    # ── per-joint section ────────────────────────────────────────────────────
    joint_order = [
        'Cervical Pitch', 'Trunk Flexion',
        'L Shoulder Flex', 'R Shoulder Flex',
        'L Shoulder Abd',  'R Shoulder Abd',
        'L Hip',  'R Hip',
        'L Knee', 'R Knee',
        'L Ankle','R Ankle',
        'Mean',
    ]
    # tolerate minor name differences (short vs long form)
    all_joints = list(base_joint.keys())

    def find_key(d, want):
        if want in d:
            return want
        for k in d:
            if k.lower() == want.lower():
                return k
        return None

    rows_out.append(['Section', 'Name', 'Baseline_MAE_deg', 'Novel_MAE_deg',
                     'Delta_deg', 'Improvement_pct', 'Baseline_Clinical',
                     'Novel_Clinical'])

    bar = '=' * 80
    sep = '-' * 80
    print('\n' + bar)
    print('Per-Joint MAE — Baseline vs Novel (GCADA)')
    print(bar)
    hdr = f"{'Joint':<22} | {'Baseline':>10} | {'Novel':>10} | {'Delta':>8} | {'Δ%':>7} | Clin"
    print(hdr)
    print(sep)

    for jname in joint_order:
        bk = find_key(base_joint, jname)
        nk = find_key(nov_joint,  jname)
        if bk is None or nk is None:
            print(f'  WARNING: joint "{jname}" not found — skipping')
            continue

        b_mae = float(base_joint[bk]['MAE_deg'])
        n_mae = float(nov_joint[nk]['MAE_deg'])
        delta = n_mae - b_mae
        pct   = (delta / b_mae * 100) if b_mae else 0.0
        b_clin = base_joint[bk].get('Clinical_pass', '?')
        n_clin = nov_joint[nk].get('Clinical_pass', '?')

        arrow = '↓' if delta < 0 else ('↑' if delta > 0 else '=')
        print(f'{jname:<22} | {b_mae:>9.2f}° | {n_mae:>9.2f}° | '
              f'{delta:>+7.2f}° | {pct:>+6.1f}% | {n_clin}  {arrow}')

        rows_out.append(['per_joint', jname, f'{b_mae:.4f}', f'{n_mae:.4f}',
                         f'{delta:+.4f}', f'{pct:+.1f}', b_clin, n_clin])
    print(sep)

    # ── per-exercise section ─────────────────────────────────────────────────
    ex_order = [f'Ex{i+1:02d}' for i in range(10)] + ['AVERAGE']
    ex_names = {
        'Ex01': 'Deep Squat',      'Ex02': 'Hurdle Step',
        'Ex03': 'Inline Lunge',    'Ex04': 'Side Lunge',
        'Ex05': 'Sit to Stand',    'Ex06': 'Straight Leg Raise',
        'Ex07': 'Shoulder Abduction', 'Ex08': 'Shoulder Extension',
        'Ex09': 'Shoulder Int-Ext Rot', 'Ex10': 'Shoulder Scaption',
        'AVERAGE': 'All Exercises',
    }

    print('\n' + bar)
    print('Per-Exercise ROM MAE — Baseline vs Novel')
    print(bar)
    hdr2 = f"{'Exercise':<8} | {'Name':<28} | {'Baseline':>10} | {'Novel':>10} | {'Delta':>8} | {'Δ%':>7}"
    print(hdr2)
    print(sep)

    for ex in ex_order:
        bk = find_key(base_ex, ex)
        nk = find_key(nov_ex,  ex)
        if bk is None or nk is None:
            continue
        b_mae = float(base_ex[bk]['ROM_MAE_deg'])
        n_mae = float(nov_ex[nk]['ROM_MAE_deg'])
        delta = n_mae - b_mae
        pct   = (delta / b_mae * 100) if b_mae else 0.0
        name  = ex_names.get(ex, ex)

        arrow = '↓' if delta < 0 else ('↑' if delta > 0 else '=')
        print(f'{ex:<8} | {name:<28} | {b_mae:>9.2f}° | {n_mae:>9.2f}° | '
              f'{delta:>+7.2f}° | {pct:>+6.1f}% {arrow}')

        rows_out.append(['per_exercise', f'{ex} {name}', f'{b_mae:.4f}',
                         f'{n_mae:.4f}', f'{delta:+.4f}', f'{pct:+.1f}',
                         base_ex[bk].get('MPJPE_mm', ''), nov_ex[nk].get('MPJPE_mm', '')])
    print(sep)

    # ── write CSV ────────────────────────────────────────────────────────────
    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows_out)

    print(f'\nComparison saved to: {args.output}')
    print(bar)


if __name__ == '__main__':
    main()
