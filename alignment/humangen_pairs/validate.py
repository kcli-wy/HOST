"""Validate exported pairs using the actual alignment dataset and optional processor."""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import random
import sys

from common import (
    ALIGNMENT_ROOT, DEFAULT_CONFIG, DATASET_NAME, apply_alignment_settings,
    load_settings, video_paths_file,
)


def decode_paths(paths):
    import decord

    groups = defaultdict(set)
    for path in paths:
        video, frame, _ = path.removeprefix('video://').rsplit('::', 2)
        groups[video].add(int(frame))
    for video, frames in groups.items():
        reader = decord.VideoReader(video, num_threads=1)
        decoded = reader.get_batch(sorted(frames)).asnumpy()
        if len(decoded) != len(frames) or decoded.shape[-1] != 3:
            raise ValueError(f'Frame decoding failed for {video}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--samples-per-dataset', type=int, default=2)
    parser.add_argument('--with-processor', action='store_true', help='Also build one real Qwen input batch on CPU; does not load model weights')
    args = parser.parse_args()
    if args.samples_per_dataset < 1:
        parser.error('--samples-per-dataset must be positive')
    settings = load_settings(args.config)
    if args.output_root:
        settings['output_root'] = args.output_root.resolve()
    output = settings['output_root']
    paths = json.loads(video_paths_file(settings).read_text())
    cameras = json.loads((output / 'cam_mapping' / f'{DATASET_NAME}_cam_mapping.json').read_text())
    with (output / 'pairs.jsonl').open(encoding='utf-8') as handle:
        pairs = [json.loads(line) for line in handle]
    if not pairs or len(pairs) != len(paths) or len(set(paths)) != len(paths):
        raise ValueError('Training index must contain one unique robot episode per pair')
    grouped = defaultdict(list)
    for index, pair in enumerate(pairs):
        robot, human = Path(pair['robot_episode']), Path(pair['human_episode'])
        if paths[index] != str(robot) or robot == human or robot.parent != human.parent:
            raise ValueError(f'Invalid pair layout at index {index}')
        if cameras.get(str(robot.parent)) != ['images']:
            raise ValueError(f'Missing camera mapping for {robot.parent}')
        if min(pair['robot_frames'], pair['human_frames']) < settings['training']['num_frames']:
            raise ValueError(f'Pair is too short at index {index}')
        grouped[pair['dataset']].append(index)

    sys.path.insert(0, str(ALIGNMENT_ROOT))
    apply_alignment_settings(settings)
    os.environ.setdefault('USE_TF', '0')
    os.environ.setdefault('USE_FLAX', '0')
    from config import CONFIG
    from datasets import AlignmentCollator, AlignmentDataset

    random.seed(0)
    dataset = AlignmentDataset(mode='train', video_paths_json=str(video_paths_file(settings)))
    selected = []
    for indices in grouped.values():
        count = min(len(indices), args.samples_per_dataset)
        selected.extend(indices[round(i * (len(indices) - 1) / max(count - 1, 1))] for i in range(count))
    results = []
    first_item = None
    for index in selected:
        pair = pairs[index]
        for role, peer in (('robot', 'human'), ('human', 'robot')):
            episode = Path(pair[f'{role}_episode'])
            video = episode / 'images.mp4'
            if not video.is_symlink() or str(video.readlink()) != pair[f'{role}_source'] or not video.is_file():
                raise ValueError(f'Incorrect source link: {video}')
            for name in ('task_paths.json', 'task_paths_eval.json'):
                peers = json.loads((episode / name).read_text())
                if peers != {'same': [pair[f'{peer}_episode']]}:
                    raise ValueError(f'Incorrect pair reference: {episode / name}')
        # Call the implementation directly: __getitem__ retries indefinitely and can hide invalid pairs.
        for mode in ('train', 'eval'):
            dataset.mode = mode
            item = dataset._get_item_impl(index)
            if item['name'] != pair['robot_episode'] or item['ref_name'] != pair['human_episode']:
                raise ValueError(f'Unexpected self-alignment or pair mismatch at {index}')
            if int(item['seq_lens']) != pair['robot_frames'] or int(item['ref_seq_lens']) != pair['human_frames']:
                raise ValueError(f'Frame length mismatch at {index}')
            decode_paths(item['frame_paths'] + item['ref_frame_paths'] + item['align_frame_paths'])
            if first_item is None:
                first_item = item
        result = {'dataset': pair['dataset'], 'manifest_position': pair['manifest_position'],
                  'robot_frames': pair['robot_frames'], 'human_frames': pair['human_frames']}
        results.append(result)
        print(f'PASS train/eval pair and frame decoding: {result}', flush=True)
    processor_shape = None
    if args.with_processor:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(
            CONFIG.MODEL.BASE_MODEL.MODEL_PATH, trust_remote_code=True, local_files_only=True,
        )
        collator = AlignmentCollator(processor=processor, mode='train')
        batch = collator([first_item])
        processor_shape = list(batch['qwen_input']['input_ids'].shape)
        print(f'PASS Qwen processor + real training collator: input_ids={processor_shape}', flush=True)
    report = {'pairs_in_index': len(pairs), 'sampled_pairs': results,
              'processor_input_ids_shape': processor_shape,
              'model_forward_or_training_run': False}
    (output / 'validation_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'Validated {len(pairs)} index records and {len(results)} sampled train/eval pairs.', flush=True)


if __name__ == '__main__':
    main()
