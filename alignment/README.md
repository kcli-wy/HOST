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

### Local model path

Set `CONFIG.MODEL.BASE_MODEL.MODEL_PATH` in `config.py` to the local
Qwen3-VL-Embedding-8B directory. It defaults to `Qwen3-VL-Embedding-8B/` at the
repository root, resolved relative to `config.py` rather than the working directory.
Model loading, training/evaluation processors, and checkpoint conversion all use
this setting. Conversion scripts still accept an explicit model path through
`--model_name_or_path` (Python) or the second positional argument (shell).

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
CONFIG.DATA.CAM_MAPPING_DIR = ...    # defaults to humangen_pairs/output/cam_mapping
CONFIG.DATA.SEGMENTED_PATH_DATASETS = ('AgiBotWorld', '10042')  # dataset ids using the segmented path:id:start-end format
CONFIG.JOINTS.USE_JOINTS = False     # set True + JOINT_ACTION_MAPPING_DIR to condition on joint state
```

## Training

```bash
VIDEO_PATHS=/path/to/video_paths.json bash train_scripts/run_ds.sh
```

This wraps `train.py` with `torchrun` + DeepSpeed ZeRO-3. Key flags (see `train.py --help` for the
full list): `--video_paths` (required), `--network` (`Qwen3-VL-Embedding-8B` is the verified path),
`--gradient_accumulation_steps`, `--ds_config`, `--save_interval`, `--max_iters`, `--resume_dir`,
`--pretrain_weights`. `WANDB_API_KEY` falls back to offline logging if unset.

For the prepared HumanGen pairs, the default camera mapping and video-only settings in
`config.py` allow the same launcher to be used directly:

```bash
VIDEO_PATHS=humangen_pairs/output/HumanGen_video_paths.json bash train_scripts/run_ds.sh
```

See [`humangen_pairs/README.md`](./humangen_pairs/README.md) for data preparation and validation.
For other datasets, update `CONFIG.DATA.CAM_MAPPING_DIR` and, if joint conditioning is needed,
`CONFIG.JOINTS.USE_JOINTS` and `CONFIG.JOINTS.JOINT_ACTION_MAPPING_DIR`.
The ZeRO-3 configuration offloads optimizer states to CPU, and training uses four persistent
`spawn` data workers with one prefetched batch per worker. Spawning workers avoids inheriting
the large CPU optimizer state through `fork`. Single-GPU full-model training still requires
substantial host RAM. It uses PyTorch AdamW to avoid compiling DeepSpeed optimizer extensions.
`MAX_ITERS` and `SAVE_INTERVAL` can override the launcher's defaults
(3,000 and 500 optimizer steps); use `MAX_ITERS=2` for a short training check.

### Configuration files

The original launcher uses [`config.py`](./config.py) for model/data/loss hyperparameters and
[`scripts/ds_config_zero3.json`](./scripts/ds_config_zero3.json) for DeepSpeed precision, ZeRO
and offload settings. [`train_scripts/run_ds.sh`](./train_scripts/run_ds.sh) supplies launch
arguments and environment-variable overrides. It does not read
[`humangen_pairs/config.json`](./humangen_pairs/config.json), which controls data preparation
and the optional HumanGen wrapper. See the
[configuration reference](./humangen_pairs/README.md#配置文件位置与作用) for all file locations,
effective defaults, generated data mappings, path resolution and override precedence.

### Input pipeline performance

`CONFIG.DATA.NUM_WORKERS=4` and `CONFIG.DATA.PREFETCH_FACTOR=1` in `config.py` control training
data preparation. Workers use `spawn`, receive the active configuration, and remain alive across
epochs. Set `CONFIG.DATA.NUM_WORKERS=0` for synchronous debugging. Increasing the worker count also increases
host memory use and concurrent storage access; measure throughput before increasing it further.
The first batch includes worker startup and is not representative of steady-state data wait.

The collator decodes each requested video frame once in sorted order, reuses raw alignment
frames across the paired rows, and builds only the paired Qwen input during training. Each row
still receives its original independent augmentation. Evaluation continues to build the main-only
input used by reference caching. CPU optimizer offload still introduces pauses at update boundaries.
See [the HumanGen performance notes](./humangen_pairs/README.md#训练读取性能) for validation results.

`CONFIG.TRAIN.CPU_NUM_THREADS=22` limits each training process's PyTorch CPU thread count.
This matches the current single-GPU container's CPU quota more closely than the original
96-thread default; adjust it to the CPU resources available per rank on other machines or
multi-GPU runs. Data worker thread counts are independent. Embedding forwards also use
`logits_to_keep=1` to avoid projecting every token into the vocabulary while preserving all
hidden features. Reference features are cached only during evaluation.
Training no longer populates that cache with tensors that retain computation graphs.
Packed CLS features are gathered with `index_select` to avoid reading each GPU index as a
Python scalar. CPU optimizer offload still causes periodic utilization drops; see
[the utilization notes](./humangen_pairs/README.md#为什么-gpu-利用率仍会高低交替) for the
measured update cost, remaining bottleneck and next optimization prerequisites.

## Evaluation

```bash
python evaluate_v2.py --video_paths <your_video_paths.json> --network Qwen3-VL-Embedding-8B --resume_dir <checkpoint_dir>
```

Evaluates alignment quality (embedding-space nearest-neighbor correspondence) between Main and
Reference videos using the same data format as training. `--resume_dir` must contain one of
`checkpoint.pth.tar`, `fp32_converted/pytorch_model.bin`, or `pytorch_model.bin` (see
`scripts/convert_ds_to_hf.sh` / `scripts/convert_ckpt_to_hf.py` for converting a DeepSpeed
checkpoint into one of these formats).
