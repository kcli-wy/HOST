"""Run the existing alignment trainer with the HumanGen video-only configuration."""

import argparse
import os
import sys

from common import (
    ALIGNMENT_ROOT, DEFAULT_CONFIG, apply_alignment_settings, load_settings,
    video_paths_file,
)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--humangen_config', default=str(DEFAULT_CONFIG))
    args, remaining = parser.parse_known_args()
    settings = load_settings(args.humangen_config)
    sys.path.insert(0, str(ALIGNMENT_ROOT))
    apply_alignment_settings(settings)
    os.environ.setdefault('USE_TF', '0')
    os.environ.setdefault('USE_FLAX', '0')
    cache_root = settings['output_root'] / '.cache'
    os.environ.setdefault('TRITON_CACHE_DIR', str(cache_root / 'triton'))
    os.environ.setdefault('TORCH_EXTENSIONS_DIR', str(cache_root / 'torch_extensions'))

    import train as training
    from absl import app

    defaults = {
        'video_paths': str(video_paths_file(settings)),
        'ds_config': str(ALIGNMENT_ROOT / 'scripts/ds_config_zero3.json'),
        'logdir': str(ALIGNMENT_ROOT / 'logs/humangen'),
        'gradient_accumulation_steps': settings['training']['gradient_accumulation_steps'],
        'max_iters': settings['training']['max_iters'],
        'save_interval': settings['training']['save_interval'],
    }
    for key, value in defaults.items():
        training.FLAGS.set_default(key, value)
    sys.argv = [sys.argv[0], *remaining]
    app.run(training.train)


if __name__ == '__main__':
    main()
