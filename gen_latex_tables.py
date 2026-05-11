"""
Generate two publication-quality LaTeX tables for the GCADA thesis.

  FILE 1: results_comparison/table_per_exercise.tex
  FILE 2: results_comparison/table_comparison_main.tex

Usage:
    python gen_latex_tables.py --output_dir results_comparison/
"""
import argparse
import os
import textwrap

# ── data (from evaluate_comparison.py results) ────────────────────────────────

EXERCISE_DATA = [
    # (id_str, full_name,               mae_deg, mpjpe_mm)
    ('Ex01', 'Deep Squat',                4.2067,  56.86),
    ('Ex02', 'Hurdle Step',               5.1618,  55.51),
    ('Ex03', 'Inline Lunge',              5.7252,  46.88),
    ('Ex04', 'Side Lunge',                4.9691,  72.30),
    ('Ex05', 'Sit to Stand',              5.1122,  50.81),
    ('Ex06', 'Straight Leg Raise',        4.2173,  76.85),
    ('Ex07', 'Shoulder Abduction',        3.3530,  76.17),
    ('Ex08', 'Shoulder Extension',        6.4673, 109.11),
    ('Ex09', 'Sho.\ Int-Ext Rotation',   2.9717,  73.90),
    ('Ex10', 'Shoulder Scaption',         3.9354,  67.43),
]
EXERCISE_AVERAGE = ('AVERAGE', 'All Exercises', 4.5891, 67.65)

# Joint-group means for comparison table (L/R averaged)
# Derived from per_joint_mae.csv
GCADA_GROUPS = {
    'Cervical Pitch':  3.13,
    'Trunk Flexion':   1.68,
    'Shoulder Flex':   (4.8367 + 7.3282) / 2,   # 6.08
    'Shoulder Abd':    (4.7524 + 9.3425) / 2,   # 7.05
    'Hip':             (2.5807 + 3.1268) / 2,   # 2.85
    'Knee':            (3.8050 + 3.7747) / 2,   # 3.79
    'Ankle':           (4.3638 + 6.3499) / 2,   # 5.36
}
GCADA_MEAN_MAE  = 4.59   # degrees
GCADA_MPJPE_MM  = 67.65  # mm

CLINICAL_THRESHOLD = 5.0  # McGinley et al. 2009


# ── helpers ───────────────────────────────────────────────────────────────────

def check(v, threshold=CLINICAL_THRESHOLD):
    """Return LaTeX checkmark/cross symbol string."""
    return r'\checkmark' if v < threshold else r'\texttimes'


def bold(s):
    return rf'\textbf{{{s}}}'


def nd():
    """Not reported / not directly comparable."""
    return r'--'


# ─────────────────────────────────────────────────────────────────────────────
# FILE 1 — Per-exercise table
# ─────────────────────────────────────────────────────────────────────────────

def build_per_exercise_tex():
    rows = []
    for ex_id, name, mae, mpjpe in EXERCISE_DATA:
        mark = check(mae)
        rows.append(
            f'    {ex_id} & {name} & {mae:.2f} & {mpjpe:.1f} & ${mark}$ \\\\'
        )

    # Bold average row
    avg_id, avg_name, avg_mae, avg_mpjpe = EXERCISE_AVERAGE
    avg_mark = check(avg_mae)
    avg_row = (
        f'    {bold(avg_id)} & {bold(avg_name)} & '
        f'{bold(f"{avg_mae:.2f}")} & {bold(f"{avg_mpjpe:.1f}")} & '
        f'${bold(f"{avg_mark}")}$ \\\\'
    )

    body = '\n'.join(rows)

    tex = rf"""\begin{{table}}[ht]
\centering
\caption{{Per-exercise ROM MAE and body MPJPE of GCADA on the UI-PRMD
  test set (subjects S09--S10, $n=2$). ROM MAE is the mean absolute
  error averaged over all 12 joints and all frames of that exercise.
  MPJPE is the mean per-joint position error on the 23 body joints,
  in millimetres. The clinical threshold of $5^\circ$ follows
  McGinley et al.\ \cite{{mcginley2009}};
  $\checkmark$\,=\,pass, $\times$\,=\,fail.}}
\label{{tab:per_exercise_mae}}
\setlength{{\tabcolsep}}{{6pt}}
\renewcommand{{\arraystretch}}{{1.15}}
\begin{{tabular}}{{llccc}}
\toprule
\textbf{{ID}} &
\textbf{{Exercise}} &
\textbf{{ROM MAE ($^\circ$)}} &
\textbf{{MPJPE (mm)}} &
\textbf{{$<5^\circ$?}} \\
\midrule
{body}
\midrule
{avg_row}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    return tex


# ─────────────────────────────────────────────────────────────────────────────
# FILE 2 — Main comparison table
# ─────────────────────────────────────────────────────────────────────────────

def build_comparison_tex():
    """
    Columns: Joint/Metric | GCADA (ours) | Physio2.2M | Aguilar-Ortega | Mercadal-Baudart
    Rows grouped by joint group + MPJPE row.

    Bold GCADA value where it strictly beats all reported competitor values.
    """

    # Physio2.2M reports best/worst range → show as "X.X -- Y.Y"
    # Aguilar-Ortega: mean body joint MAE 10.30°
    # Mercadal-Baudart: RMSE ≤ 10--15° (range)
    # "--" where not reported

    def gcada_cell(v, beats_all=False):
        s = f'{v:.2f}'
        return bold(s) if beats_all else s

    # For each row decide if GCADA beats all reported competitors
    # (only compare against numeric values; -- rows excluded from comparison)
    rows = []

    # ── Joint MAE rows ────────────────────────────────────────────────────────
    # Cervical — no competitor reports this
    rows.append((
        'Cervical Pitch',
        gcada_cell(GCADA_GROUPS['Cervical Pitch'], False),
        nd(), nd(), nd(),
        r'GCADA only',
    ))

    # Trunk — no competitor reports this
    rows.append((
        'Trunk Flexion',
        gcada_cell(GCADA_GROUPS['Trunk Flexion'], False),
        nd(), nd(), nd(),
        r'GCADA only',
    ))

    # Shoulder Flex — Physio2.2M does not report shoulder; A-O mean 10.30
    gcada_shoflex = GCADA_GROUPS['Shoulder Flex']
    beats_shoflex = gcada_shoflex < 10.30   # True
    rows.append((
        r'Shoulder Flex$^\dagger$',
        gcada_cell(gcada_shoflex, beats_shoflex),
        nd(),
        '10.30 (mean)',
        nd(),
        r'L/R avg.; A-O: mean body',
    ))

    # Shoulder Abd
    gcada_shoadd = GCADA_GROUPS['Shoulder Abd']
    beats_shoadd = gcada_shoadd < 10.30
    rows.append((
        r'Shoulder Abd$^\dagger$',
        gcada_cell(gcada_shoadd, beats_shoadd),
        nd(),
        '10.30 (mean)',
        nd(),
        r'L/R avg.; high R asymmetry',
    ))

    # Hip
    gcada_hip = GCADA_GROUPS['Hip']
    beats_hip = gcada_hip < 10.30
    rows.append((
        r'Hip$^\dagger$',
        gcada_cell(gcada_hip, beats_hip),
        nd(),
        '10.30 (mean)',
        nd(),
        'L/R avg.',
    ))

    # Knee — Physio2.2M reports 9.3 (best) / 21.9 (worst)
    gcada_knee = GCADA_GROUPS['Knee']
    beats_knee = gcada_knee < 9.3   # True — beat best Physio2.2M
    rows.append((
        r'Knee Flex$^\dagger$',
        gcada_cell(gcada_knee, beats_knee),
        '9.3 -- 21.9',
        '10.30 (mean)',
        r'$\leq$10--15 (RMSE)',
        r'GCADA $2.5\times$ better than P2.2M-best',
    ))

    # Ankle
    gcada_ankle = GCADA_GROUPS['Ankle']
    beats_ankle = gcada_ankle < 10.30
    rows.append((
        r'Ankle$^\dagger$',
        gcada_cell(gcada_ankle, beats_ankle),
        nd(),
        '10.30 (mean)',
        nd(),
        'L/R avg.',
    ))

    # ── Mean MAE row ──────────────────────────────────────────────────────────
    beats_mean = GCADA_MEAN_MAE < 9.3
    rows.append((
        r'\textbf{Mean ROM MAE}',
        gcada_cell(GCADA_MEAN_MAE, beats_mean),
        nd(),
        '10.30',
        r'$\leq$10--15 (RMSE)',
        r'All joints averaged',
    ))

    # ── MPJPE row ─────────────────────────────────────────────────────────────
    beats_mpjpe = GCADA_MPJPE_MM < 72.0
    rows.append((
        r'\textbf{Body MPJPE (mm)}',
        gcada_cell(GCADA_MPJPE_MM, beats_mpjpe),
        '72 -- 122',
        nd(),
        nd(),
        r'GCADA on UI-PRMD vs P2.2M on Physio2.2M',
    ))

    # Build LaTeX rows (6-column: joint, gcada, p22m, a-o, m-b, note)
    latex_rows = []
    for i, (joint, gcada, p22m, ao, mb, note) in enumerate(rows):
        # Insert \midrule before Mean and MPJPE rows
        if joint.startswith(r'\textbf{Mean') or joint.startswith(r'\textbf{Body'):
            latex_rows.append(r'    \midrule')
        latex_rows.append(
            f'    {joint} & ${gcada}$ & {p22m} & {ao} & {mb} \\\\'
        )

    body = '\n'.join(latex_rows)

    tex = rf"""\begin{{table}}[ht]
\centering
\caption{{Comparison of GCADA with related monocular rehabilitation pose
  estimation methods. ROM MAE in degrees ($^\circ$); MPJPE in millimetres.
  $\dagger$\,=\,left/right side averaged.
  \textbf{{Bold}} = GCADA strictly outperforms all reported competitor values.
  Physio2.2M \cite{{rode2025}} reports best/worst model range;
  Aguilar-Ortega et al.\ \cite{{aguilar2023}} report a single mean body MAE;
  Mercadal-Baudart et al.\ \cite{{mercadal2024}} report RMSE ranges.
  $^*$\,Physio2.2M and Aguilar-Ortega are evaluated on different datasets
  (zero-shot on their respective corpora); GCADA is fine-tuned on UI-PRMD.}}
\label{{tab:comparison_main}}
\setlength{{\tabcolsep}}{{5pt}}
\renewcommand{{\arraystretch}}{{1.18}}
\begin{{tabular}}{{lcccc}}
\toprule
\textbf{{Joint / Metric}} &
\textbf{{GCADA (ours)}} &
\textbf{{Physio2.2M~\cite{{rode2025}}}} &
\textbf{{Aguilar-Ortega~\cite{{aguilar2023}}}} &
\textbf{{Mercadal-Baudart~\cite{{mercadal2024}}}} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    return tex, rows


# ─────────────────────────────────────────────────────────────────────────────
# Text previews (terminal)
# ─────────────────────────────────────────────────────────────────────────────

def print_exercise_preview():
    sep  = '=' * 74
    hsep = '-' * 74
    hdr  = f"{'ID':<8} {'Exercise':<28} {'ROM MAE':>10} {'MPJPE':>10} {'<5°?':>6}"
    print('\n' + sep)
    print('TABLE 1 — Per-Exercise ROM MAE (UI-PRMD Test Set)')
    print(sep)
    print(hdr)
    print(hsep)
    for ex_id, name, mae, mpjpe in EXERCISE_DATA:
        mark = '✓' if mae < CLINICAL_THRESHOLD else '✗'
        print(f"{ex_id:<8} {name:<28} {mae:>9.2f}° {mpjpe:>9.1f}mm {mark:>6}")
    print(hsep)
    avg_id, avg_name, avg_mae, avg_mpjpe = EXERCISE_AVERAGE
    avg_mark = '✓' if avg_mae < CLINICAL_THRESHOLD else '✗'
    print(f"{avg_id:<8} {avg_name:<28} {avg_mae:>9.2f}° {avg_mpjpe:>9.1f}mm {avg_mark:>6}")
    print(sep)


def print_comparison_preview(rows):
    sep  = '=' * 96
    hsep = '-' * 96
    hdr  = (f"{'Joint / Metric':<26} {'GCADA':>10} "
            f"{'Physio2.2M':>16} {'Aguilar-Ortega':>16} {'Mercadal-Baudart':>18}")
    print('\n' + sep)
    print('TABLE 2 — GCADA vs Literature Comparison')
    print(sep)
    print(hdr)
    print(hsep)
    for joint, gcada, p22m, ao, mb, note in rows:
        # Strip LaTeX markup for display
        def plain(s):
            for cmd in [r'\textbf{', r'\checkmark', r'\texttimes',
                        r'\leq', r'$', '}', r'\dagger', r'\cite{rode2025}',
                        r'\cite{aguilar2023}', r'\cite{mercadal2024}']:
                s = s.replace(cmd, '')
            return s.replace(r'^\dagger', '†').strip()

        j = plain(joint)
        g = plain(gcada)
        p = plain(p22m)
        a = plain(ao)
        m = plain(mb)
        # Mark bold values with *
        if 'textbf' in gcada or r'\textbf' in gcada:
            g = f'*{g}*'
        print(f"{j:<26} {g:>10} {p:>16} {a:>16} {m:>18}")
    print(sep)
    print('* = GCADA beats all reported competitors (bold in LaTeX)')
    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--output_dir', default='results_comparison/')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Table 1 ───────────────────────────────────────────────────────────────
    tex1 = build_per_exercise_tex()
    path1 = os.path.join(args.output_dir, 'table_per_exercise.tex')
    with open(path1, 'w') as f:
        f.write(tex1)
    print_exercise_preview()
    print(f'\nSaved: {path1}')

    # ── Table 2 ───────────────────────────────────────────────────────────────
    tex2, rows = build_comparison_tex()
    path2 = os.path.join(args.output_dir, 'table_comparison_main.tex')
    with open(path2, 'w') as f:
        f.write(tex2)
    print_comparison_preview(rows)
    print(f'\nSaved: {path2}')

    # ── raw LaTeX preview ─────────────────────────────────────────────────────
    print('\n' + '─' * 60)
    print('RAW LaTeX — table_per_exercise.tex')
    print('─' * 60)
    print(tex1)

    print('─' * 60)
    print('RAW LaTeX — table_comparison_main.tex')
    print('─' * 60)
    print(tex2)


if __name__ == '__main__':
    main()
