# Target Coupling (alignment/)

This module implements **target coupling**, the mechanism behind **HOST (Human-to-robot
One-shot Skill Transfer)** that aligns a visual demonstration to a robot trajectory on a shared
task-progress manifold, so the robot's prediction target is coupled to the corresponding point in
the demonstration rather than to raw clock time. It trains a frame-embedding model (Qwen3-VL
backbone) with Temporal Cycle-Consistency (TCC) and Smooth DTW losses so that semantically
corresponding moments across two videos of the same task map to nearby points in embedding space;
`policy_training/`'s dataset construction uses this alignment (via `coupling/`) to build its
coupled prediction targets.

## Index

- [File Structure](#file-structure)
- [Environment Setup](#environment-setup)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)

## File Structure

```text
alignment/
├── train.py                  # Training entrypoint
├── evaluate_v2.py             # Evaluation entrypoint
├── models.py                  # BaseModel, embedders
├── datasets.py                # AlignmentDataset / AlignmentVideoDataset, AlignmentCollator
├── config.py                  # CONFIG.* — data/training/model hyperparameters
├── tcc/                       # Smooth DTW + TCC loss implementations
├── algos/                     # Alignment / Algorithm module-level wrappers
├── monkey_patch_forward.py    # Qwen3-VL packed-sequence attention/position-id patches
├── train_scripts/
│   └── run_ds.sh              # DeepSpeed training entrypoint
└── scripts/                   # Checkpoint conversion utilities
```

## Environment Setup

```bash
conda env create -f environment.yml
conda activate HOST_Alignment
```

`environment.yml` is a full conda environment export; if you'd rather build one incrementally,
the key dependencies are PyTorch, DeepSpeed, `transformers` (for Qwen3-VL), and the Qwen3-VL
image/video processor stack.

## Data Preparation

`alignment/` and `policy_training/` consume the same on-disk data convention — see
[`data_preprocessing/README.md`](../data_preprocessing/README.md) at the repo root for the full,
single-source-of-truth schema (video-paths list, episode directory layout, camera mapping,
joint/action normalization) and the task-grouping scripts that build `task_paths.json`. Pass your
own episode-list JSON through the `VIDEO_PATHS` environment variable when launching `run_ds.sh`.

`alignment/` specifically needs only video frames plus task grouping (`task_paths.json`) — no
robot actions/joint data are required unless you enable joint conditioning
(`CONFIG.JOINTS.USE_JOINTS`); see
[§2.6 of `data_preprocessing/README.md`](../data_preprocessing/README.md#26-per-module-differences)
for the exact per-module requirements. Video-paths entries can also address a segment of a longer
recording as `path:segment_id:start-end` (inclusive frame range), parsed by
`AlignmentDataset._parse_video_path` in `datasets.py`.

Relevant `config.py` fields:

```python
CONFIG.DATA.NUM_STEPS = 3            # frames per chunk
CONFIG.DATA.FRAME_STRIDE = 10        # stride between context frames
CONFIG.TRAIN.NUM_FRAMES = 24         # anchor frames sampled per video
CONFIG.TRAIN.NUM_ALIGN_FRAMES = 24   # reference frames used for the alignment window
CONFIG.DATA.CAM_MAPPING_DIR = ''     # set to your cam_mapping directory if multi-view
CONFIG.DATA.SEGMENTED_PATH_DATASETS = ('AgiBotWorld', '10042')  # dataset ids using the segmented path:id:start-end format
CONFIG.JOINTS.USE_JOINTS = False     # set True + JOINT_ACTION_MAPPING_DIR to condition on joint state
```

The first camera in each mapping entry is the primary view. The loader pads one
view to `[primary, primary, primary]` and two views to `[primary, secondary, primary]`.
Main and Reference are padded independently for both context and alignment frames;
three or more native views keep their order. Padding reuses frame paths without
copying videos or editing camera mappings. With up to three native views, keep
`NUM_STEPS=3`, `NUM_FRAMES=24`, and `NUM_ALIGN_FRAMES=24`.

## Training

Training defaults live in `config.py`: `CONFIG.DATA.VIDEO_PATHS` lists all six generated
HumanGen manifests, `CONFIG.MODEL.BASE_MODEL.MODEL_PATH` selects the local model,
`CONFIG.DATA.CAM_MAPPING_DIR` selects camera mappings, and `CONFIG.JOINTS.USE_JOINTS=False`
enables visual-only training. Change `CONFIG.DATA.HUMANGEN_ROOT` to relocate the
default manifests and camera mappings. Model weights, configuration, tokenizer, and
processor files must all be present in the model directory. Command-line flags override
the `VIDEO_PATHS`, `MODEL_PATH`, `CAM_MAPPING_DIR`, and `USE_JOINTS` environment variables;
both override `config.py`. The launcher contains no defaults for these four settings.

```bash
bash train_scripts/run_ds.sh
```

This wraps `train.py` with `torchrun` + DeepSpeed ZeRO-2 and CPU optimizer offload, using
`scripts/ds_config_zero2.json`. Set `DS_CONFIG` to select a different profile. Key flags (see `train.py --help` for the
full list): `--video_paths` (overrides `CONFIG.DATA.VIDEO_PATHS`), `--network` (`Qwen3-VL-Embedding-8B` is the verified path),
`--gradient_accumulation_steps`, `--ds_config`, `--save_interval`, `--max_iters`, `--resume_dir`,
`--pretrain_weights`. `WANDB_API_KEY` falls back to offline logging if unset.

Run the CPU regression checks in `HOST_Alignment` without datasets or model weights:

```bash
USE_TF=0 python -m unittest test_training_throughput -v
```

These check Qwen feature/gradient equivalence, multi-view frame ordering, reference-cache
behavior, and data-loader worker reuse. Throughput and memory still require an actual GPU run.

## Evaluation

```bash
python evaluate_v2.py --video_paths <your_video_paths.json> --network Qwen3-VL-Embedding-8B --resume_dir <checkpoint_dir>
```

Evaluates alignment quality (embedding-space nearest-neighbor correspondence) between Main and
Reference videos using the same data format as training. `--resume_dir` must contain one of
`checkpoint.pth.tar`, `fp32_converted/pytorch_model.bin`, or `pytorch_model.bin` (see
`scripts/convert_ds_to_hf.sh` / `scripts/convert_ckpt_to_hf.py` for converting a DeepSpeed
checkpoint into one of these formats).
