"""Shared paths and video-only alignment settings for HumanGen."""

import json
from pathlib import Path


DEFAULT_CONFIG = Path(__file__).resolve().with_name('config.json')
ALIGNMENT_ROOT = Path(__file__).resolve().parent.parent
DATASET_NAME = 'HumanGen'


def load_settings(config_path=DEFAULT_CONFIG):
    config_path = Path(config_path).resolve()
    settings = json.loads(config_path.read_text(encoding='utf-8'))
    for key in ('source_root', 'output_root'):
        path = Path(settings[key]).expanduser()
        if not path.is_absolute():
            path = config_path.parent / path
        settings[key] = path.resolve()
    if settings['workers'] < 1 or settings['training']['num_frames'] < 1:
        raise ValueError('workers and training.num_frames must be positive')
    if settings['training']['num_align_frames'] % 3:
        raise ValueError('training.num_align_frames must be divisible by 3')
    return settings


def video_paths_file(settings):
    return settings['output_root'] / f'{DATASET_NAME}_video_paths.json'


def apply_alignment_settings(settings):
    from config import CONFIG

    CONFIG.DATA.CAM_MAPPING_DIR = str(settings['output_root'] / 'cam_mapping')
    CONFIG.DATA.USE_CAM_MAPPING = True
    CONFIG.DATA.NUM_STEPS = 3
    CONFIG.JOINTS.USE_JOINTS = False
    CONFIG.TRAIN.NUM_FRAMES = settings['training']['num_frames']
    CONFIG.TRAIN.NUM_ALIGN_FRAMES = settings['training']['num_align_frames']
    CONFIG.DS_CONFIG.train_micro_batch_size_per_gpu = settings['training']['batch_size']
    CONFIG.TRAIN.BATCH_SIZE = settings['training']['batch_size']
