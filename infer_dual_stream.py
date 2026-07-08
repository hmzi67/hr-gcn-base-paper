"""
Inference script for Dual-Stream Quality model (dual_stream_quality.py).

This script replicates the preprocessing used during training (GMM scoring, ROM-guided adjacency,
window building) and runs evaluation on the test NPZ. It saves a summary CSV and attention plot.

Usage example:

python infer_dual_stream.py \
    --checkpoint checkpoint/dual_stream_best.pt \
    --train_npz data/uiprmd_quality_train.npz \
    --test_npz data/uiprmd_quality_test.npz \
    --save_csv results/dual_stream_results_eval.csv \
    --save_attention results/dual_stream_attention_eval.png

"""

from __future__ import print_function

import argparse
import os
import torch
import numpy as np

# Import helper functions and classes from training module
from dual_stream_quality import (
    DualStreamQualityNet,
    build_topology_adjacency,
    compute_rom_guided_init,
    fit_gmm_models,
    apply_gmm_scores,
    build_windows,
    compute_rom_stats,
    DualStreamDataset,
    evaluate,
    save_attention_viz,
    J,
)


def parse_args():
    p = argparse.ArgumentParser(description='Dual-Stream model inference/eval')
    p.add_argument('--checkpoint', required=True, help='Path to model checkpoint (.pt/.pth)')
    p.add_argument('--train_npz', default='data/uiprmd_quality_train.npz')
    p.add_argument('--test_npz',  default='data/uiprmd_quality_test.npz')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--hidden_dim', type=int, default=64)
    p.add_argument('--M', type=int, default=100)
    p.add_argument('--window_size', type=int, default=100)
    p.add_argument('--stride', type=int, default=50)
    p.add_argument('--min_len', type=int, default=50)
    p.add_argument('--lambda_ec', type=float, default=0.5,
                   help='Exercise classification loss weight used at eval (for completeness)')
    p.add_argument('--lambda_vc', type=float, default=0.3,
                   help='Validity classifier loss weight used at eval (for completeness)')
    p.add_argument('--num_workers', type=int, default=4,
                   help='Number of DataLoader workers')
    p.add_argument('--export_npz', default='',
                   help='If set, save per-window predictions to this NPZ path')
    p.add_argument('--save_csv', default='results/dual_stream_results_eval.csv')
    p.add_argument('--save_per_ex_csv', default='results/dual_stream_per_exercise_eval.csv')
    p.add_argument('--save_attention', default='results/dual_stream_attention_eval.png')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--use_npz_scores', action='store_true', help='Use quality_scores from NPZ instead of fitting GMMs')
    return p.parse_args()


def load_npz(path):
    d = np.load(path, allow_pickle=True)
    return d


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.save_csv) if os.path.dirname(args.save_csv) else '.', exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print('Loading NPZ files...')
    tr = load_npz(args.train_npz)
    te = load_npz(args.test_npz)

    tr_poses3d = tr['poses_3d']
    tr_rom     = tr['rom_angles']
    tr_ql      = tr['quality_labels'] if 'quality_labels' in tr else (tr['quality_scores'] >= 0.5).astype(np.int32)
    tr_ex      = tr['exercise_ids']
    tr_subj    = tr['subject_ids']

    te_poses3d = te['poses_3d']
    te_rom     = te['rom_angles']
    te_ql      = te['quality_labels'] if 'quality_labels' in te else (te['quality_scores'] >= 0.5).astype(np.int32)
    te_ex      = te['exercise_ids']
    te_subj    = te['subject_ids']

    # Use subject split as in training: train subjects mask
    train_mask = tr_subj <= 6

    # Load checkpoint early (if present) so we can reuse saved args and rom stats
    ckpt = None
    if os.path.exists(args.checkpoint):
        try:
            ckpt = torch.load(args.checkpoint, map_location='cpu')
        except Exception:
            ckpt = None

    # Fit or use GMM quality scores
    if args.use_npz_scores:
        print('Using NPZ quality_scores (no GMM fit)')
        tr_gmm = tr['quality_scores'].astype(np.float32) if 'quality_scores' in tr else np.zeros(len(tr_poses3d), np.float32)
        te_gmm = te['quality_scores'].astype(np.float32) if 'quality_scores' in te else np.zeros(len(te_poses3d), np.float32)
    else:
        print('Fitting PCA+GMM models on training frames (correct only)...')
        gmm_models = fit_gmm_models(tr_poses3d[train_mask], tr_rom[train_mask], tr_ex[train_mask], tr_ql[train_mask])
        print('Scoring test frames with fitted models...')
        te_gmm = apply_gmm_scores(te_rom, te_ex, te_ql, gmm_models, split_name='TEST')
        tr_gmm = apply_gmm_scores(tr_rom, tr_ex, tr_ql, gmm_models, split_name='TRAIN')

    # If checkpoint saved training args and used random split, recreate same random split
    # so windows match the training evaluation.
    if ckpt is not None and isinstance(ckpt, dict) and 'args' in ckpt:
        saved_args = ckpt['args']
        if saved_args.get('split_mode') == 'random':
            seed = int(saved_args.get('seed', 42))
            print('Recreating random split from checkpoint args (seed=%d)' % seed)

            # combine train+test like training
            all_poses3d = np.concatenate([tr_poses3d, te_poses3d], axis=0)
            all_rom     = np.concatenate([tr_rom,     te_rom],     axis=0)
            all_ql      = np.concatenate([tr_ql,      te_ql],      axis=0)
            all_ex      = np.concatenate([tr_ex,      te_ex],      axis=0)
            all_subj    = np.concatenate([tr_subj,    te_subj],    axis=0)
            _tr_sc = tr['quality_scores'].astype(np.float32) if 'quality_scores' in tr else np.zeros(len(tr_poses3d), np.float32)
            _te_sc = te['quality_scores'].astype(np.float32) if 'quality_scores' in te else np.zeros(len(te_poses3d), np.float32)
            all_gmm = np.concatenate([_tr_sc, _te_sc], axis=0)

            rng = np.random.default_rng(seed)
            idx = rng.permutation(len(all_poses3d))
            n_train = int(0.8 * len(idx))
            n_val   = int(0.1 * len(idx))

            train_idx = idx[:n_train]
            val_idx   = idx[n_train:n_train+n_val]
            test_idx  = idx[n_train+n_val:]

            tr_poses3d = all_poses3d[train_idx]
            tr_rom     = all_rom[train_idx]
            tr_ql      = all_ql[train_idx]
            tr_ex      = all_ex[train_idx]
            tr_subj    = all_subj[train_idx]
            tr_gmm     = all_gmm[train_idx]

            val_poses3d = all_poses3d[val_idx]
            val_rom     = all_rom[val_idx]
            val_ql      = all_ql[val_idx]
            val_ex      = all_ex[val_idx]
            val_subj    = all_subj[val_idx]
            val_gmm     = all_gmm[val_idx]

            te_poses3d = all_poses3d[test_idx]
            te_rom     = all_rom[test_idx]
            te_ql      = all_ql[test_idx]
            te_ex      = all_ex[test_idx]
            te_subj    = all_subj[test_idx]
            te_gmm     = all_gmm[test_idx]

            train_mask = np.ones(len(tr_poses3d), dtype=bool)
            print(f'Random split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}')

    # ROM-guided adjacency (train only)
    print('Computing ROM-guided adjacency init...')
    rom_guided_inits = compute_rom_guided_init(tr_poses3d[train_mask], tr_rom[train_mask], tr_ex[train_mask], tr_ql[train_mask])

    # Topology adjacency
    A_topology = build_topology_adjacency(J)

    # ROM normalization stats from train windows
    rom_mean = None
    rom_std = None
    if ckpt is not None and isinstance(ckpt, dict) and 'rom_mean' in ckpt and 'rom_std' in ckpt:
        rom_mean = ckpt['rom_mean']
        rom_std = ckpt['rom_std']
        print('Using rom_mean/rom_std from checkpoint')

    if rom_mean is None or rom_std is None:
        print('Computing ROM normalization stats from train windows...')
        dummy_windows = build_windows(
            tr_poses3d[train_mask], tr_rom[train_mask], tr_gmm[train_mask], tr_ql[train_mask],
            tr_ex[train_mask], tr_subj[train_mask],
            args.window_size, args.stride, args.min_len,
            rom_mean=np.zeros((10, 12), dtype=np.float32), rom_std=np.ones((10, 12), dtype=np.float32),
            augment=False, generate_invalid=False,
        )
        rom_mean, rom_std = compute_rom_stats(dummy_windows)
        del dummy_windows

    # Build test windows
    print('Building test windows...')
    test_windows = build_windows(
        te_poses3d, te_rom, te_gmm, te_ql, te_ex, te_subj,
        args.window_size, args.stride, args.min_len,
        rom_mean=rom_mean, rom_std=rom_std,
        augment=False, generate_invalid=False,
    )

    print(f'  Test windows: {len(test_windows)}')

    test_ds = DualStreamDataset(test_windows)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                              num_workers=max(0, args.num_workers), pin_memory=True)

    # Build model
    print('Building model...')
    model = DualStreamQualityNet(hidden_dim=args.hidden_dim, M=args.M, n_joints=J, n_exercises=10, A_topology=A_topology, rom_guided_inits=rom_guided_inits)
    model.to(device)

    # Load checkpoint (support several common dict shapes)
    print(f'Loading checkpoint: {args.checkpoint}')
    ckpt = torch.load(args.checkpoint, map_location=device)
    loaded = False
    if isinstance(ckpt, dict):
        for key in ('state_dict', 'model_state_dict', 'net'):
            if key in ckpt:
                model.load_state_dict(ckpt[key])
                loaded = True
                break
        if not loaded:
            # maybe the dict *is* the state_dict
            try:
                model.load_state_dict(ckpt)
                loaded = True
            except Exception:
                loaded = False
    if not loaded:
        raise RuntimeError(f"Could not find model weights in checkpoint: {args.checkpoint}")
    print('Checkpoint loaded')

    # Evaluate
    print('Running evaluation on test set...')
    test_metrics = evaluate(model, test_loader, device, lambda_ec=args.lambda_ec, lambda_vc=args.lambda_vc)

    # Print and save results
    from dual_stream_quality import print_results_table
    print_results_table(test_metrics, args.save_csv, args.save_per_ex_csv)

    # Save attention plot
    print('Saving attention visualization...')
    save_attention_viz(model, test_ds, device, args.save_attention)

    # Export per-window predictions (optional)
    if args.export_npz:
        print(f"Saving per-window predictions to {args.export_npz} ...")
        np.savez_compressed(args.export_npz,
                            qual_pred=test_metrics.get('qual_pred'),
                            qual_true=test_metrics.get('qual_true'),
                            ex_true=test_metrics.get('ex_true'))
        print('NPZ saved')

    # Also write a small JSON summary next to CSV
    try:
        import json
        summary = {
            'mad': float(test_metrics.get('mad', np.nan)),
            'ec_acc': float(test_metrics.get('ec_acc', np.nan)),
            'vc_acc': float(test_metrics.get('vc_acc', np.nan)),
            'n_windows': int(len(test_ds)),
        }
        json_path = os.path.splitext(args.save_csv)[0] + '.json'
        with open(json_path, 'w') as jf:
            json.dump(summary, jf, indent=2)
        print(f'Summary JSON: {json_path}')
    except Exception:
        pass

    print('\nDone.')


if __name__ == '__main__':
    main()
