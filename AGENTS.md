# AGENTS.md

## Scope

This repository implements HR-GCN variants for 3D whole-body pose estimation from 2D keypoints on the H3WB dataset.

Use this file as the fast-start guide, then follow [README.md](README.md) for full usage details.

## Primary Entry Points

- Training and evaluation: [HRNet_GCN_WB.py](HRNet_GCN_WB.py)
- Inference script: [infer.py](infer.py)
- Dataset loader and subject splits: [utils/prepare_data_h3wb.py](utils/prepare_data_h3wb.py)
- Core data pipeline utilities: [common/data_utils.py](common/data_utils.py), [common/generators.py](common/generators.py)
- Model implementations: [models/](models/)
- GCN layer variants: [models/gconv/](models/gconv/)
- Config defaults (YACS): [lib/config/default.py](lib/config/default.py)

## Environment And Setup

- Python and dependency baseline is documented in [README.md](README.md).
- Install dependencies:

```bash
pip install -r requirements.txt
```

- Optional native extension build (Linux NMS helper used by the HRNet code under `lib/`):

```bash
cd lib && make && cd ..
```

## Common Commands

- Train (example):

```bash
python HRNet_GCN_WB.py --gcn dc_preagg --model 1
```

- Evaluate from checkpoint:

```bash
python HRNet_GCN_WB.py --model 1 --gcn dc_preagg --evaluate checkpoint/ckpt_best.pth.tar -cfg checkpoint/w32_adam_lr1e-3.yaml
```

- Inference:

```bash
python infer.py --evaluate checkpoint_58.4/ckpt_best.pth.tar -cfg checkpoint_58.4/w32_adam_lr1e-3.yaml
```

For variant-specific commands, use [README.md](README.md).

## Project Conventions

- Default dataset prefix is `h3wb_`; scripts expect:
  - `data/h3wb_train.npz`
  - `data/h3wb_test.npz`
- Config values are loaded from YAML files (for example [w32_adam_lr1e-3.yaml](w32_adam_lr1e-3.yaml)) via `cfg.merge_from_file(...)`.
- `--gcn` selects graph convolution family (`dc_preagg`, `dc_vanilla`, `semantic`, `dc_postagg`, `convst`, `nosharing`, `modulated`).
- Model selection in code currently supports IDs `1..4` (despite some README examples showing `5`).

## Known Pitfalls

- GPU is hardcoded to `cuda:0` in main scripts; CPU-only environments will fail without code changes.
- `--resume` and `--evaluate` are mutually exclusive.
- In [HRNet_GCN_WB.py](HRNet_GCN_WB.py), checkpoint directory generation is inconsistent:
  - `model==3` and `model==4` paths are not rooted under `args.checkpoint`.
- In [infer.py](infer.py), `models.graph_hrnet_multi_branch_58` is imported but no corresponding file exists in `models/`.

## Change Guidance For Agents

- Keep edits minimal and local; avoid broad refactors unless requested.
- If you add or rename model or GCN options, update both:
  - CLI parsing and branching in [HRNet_GCN_WB.py](HRNet_GCN_WB.py)
  - Command examples in [README.md](README.md)
- Preserve checkpoint compatibility (`state_dict`, optimizer state, and naming conventions) when changing training loops.
- There is no dedicated test suite in this repo; validate with a focused smoke run (help/eval/infer path) relevant to your change.
