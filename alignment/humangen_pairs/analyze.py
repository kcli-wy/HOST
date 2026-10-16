"""Recompute HumanGen pair statistics, source-overlap audit, and report figures."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

import av
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from common import DEFAULT_CONFIG, load_settings, video_paths_file


QUANTILES = [0, 10, 25, 50, 75, 90, 95, 99, 100]


def describe(values):
    values = np.asarray(values, dtype=float)
    result = {f'p{q}': float(np.percentile(values, q)) for q in QUANTILES}
    result.update(mean=float(values.mean()), sum=float(values.sum()))
    return result


def normalize(text):
    return ' '.join(text.casefold().split())


def source_identity(dataset, sample):
    status = sample['status_info']
    relative = status.get('original_video_rel_path') or status['video_rel_path']
    prefix, suffix = relative.split('/videos/', 1)
    video = PurePosixPath(suffix)
    # Exclude camera names and frame ranges when identifying the original trajectory.
    episode = re.match(r'episode_\d+', video.stem)
    if episode is None:
        raise ValueError(f'Unknown episode name: {relative}')
    trajectory = f'{dataset}/{prefix}/{video.parts[0]}/{episode.group()}'
    return f'{dataset}/{status["video_rel_path"]}', trajectory, f'{dataset}/{prefix}'


def source_overlap(rows, test_indices):
    train = [row for i, row in enumerate(rows) if i not in test_indices]
    test = [row for i, row in enumerate(rows) if i in test_indices]
    human_train = {row['human_source'] for row in train}
    trajectory_train = {row['_trajectory'] for row in train}
    human_overlap = sum(row['human_source'] in human_train for row in test)
    robot_overlap = sum(row['_trajectory'] in trajectory_train for row in test)
    either = sum(row['human_source'] in human_train or row['_trajectory'] in trajectory_train for row in test)
    return {'train_pairs': len(train), 'test_pairs': len(test),
            'test_human_seen_in_train': human_overlap,
            'test_robot_trajectory_seen_in_train': robot_overlap,
            'test_either_source_seen_in_train': either}


def components(rows):
    parents = {}

    def find(node):
        parents.setdefault(node, node)
        while parents[node] != node:
            parents[node] = parents[parents[node]]
            node = parents[node]
        return node

    for row in rows:
        human, robot = find(('human', row['human_source'])), find(('robot', row['_trajectory']))
        if human != robot:
            parents[human] = robot
    sizes = Counter(find(('human', row['human_source'])) for row in rows)
    return {'count': len(sizes), 'pair_sizes': describe(list(sizes.values())),
            'singleton_components': sum(size == 1 for size in sizes.values()),
            'pairs_in_non_singleton_components': sum(size for size in sizes.values() if size > 1)}


def analyze_group(rows):
    human = Counter(row['human_source'] for row in rows)
    robot_clip = Counter(row['_clip'] for row in rows)
    trajectory = Counter(row['_trajectory'] for row in rows)
    task = Counter(row['_task'] for row in rows)
    ratios = [row['robot_frames'] / row['human_frames'] for row in rows]
    tiers = Counter(96 if min(row['robot_frames'], row['human_frames']) >= 96 else
                    48 if min(row['robot_frames'], row['human_frames']) >= 48 else 24 for row in rows)
    return {
        'pairs': len(rows),
        'robot_frames': describe([row['robot_frames'] for row in rows]),
        'human_frames': describe([row['human_frames'] for row in rows]),
        'robot_human_frame_ratio': describe(ratios),
        'frame_ratio_below_half': sum(value < 0.5 for value in ratios),
        'frame_ratio_above_two': sum(value > 2 for value in ratios),
        'initial_sampling_tiers': dict(tiers),
        'unique_robot_files': len({row['robot_source'] for row in rows}),
        'unique_human_files': len(human), 'unique_original_robot_clips': len(robot_clip),
        'unique_original_robot_trajectories': len(trajectory), 'source_task_directories': len(task),
        'distinct_robot_instruction_texts': len({normalize(row['robot_instruction']) for row in rows}),
        'distinct_human_instruction_texts': len({normalize(row['human_instruction']) for row in rows}),
        'matching_instruction_text_pairs': sum(normalize(row['robot_instruction']) == normalize(row['human_instruction']) for row in rows),
        'duplicate_clip_human_pair_rows': len(rows) - len({(row['_clip'], row['human_source']) for row in rows}),
        'human_reuse': describe(list(human.values())),
        'human_files_used_more_than_once': sum(count > 1 for count in human.values()),
        'pairs_using_reused_human_files': sum(count for count in human.values() if count > 1),
        'robot_clip_reuse': describe(list(robot_clip.values())),
        'robot_trajectory_reuse': describe(list(trajectory.values())),
        'robot_trajectories_used_more_than_once': sum(count > 1 for count in trajectory.values()),
        'pairs_using_reused_robot_trajectories': sum(count for count in trajectory.values() if count > 1),
        'pairs_per_task_directory': describe(list(task.values())),
        'physics_scores': dict(Counter(str(row['_physics']) for row in rows)),
        'semantic_scores': dict(Counter(str(row['_semantic']) for row in rows)),
        'generators_pair_weighted': dict(Counter(row['_generator'] for row in rows)),
        'generators_unique_human_files': dict(Counter(Path(path).stem.removeprefix('generated_video_') for path in human)),
        'top_human_reuse': [
            {'path': path, 'pairs': count} for path, count in human.most_common(3)
        ],
    }


def probe_pair(row):
    result = {'dataset': row['dataset'], 'manifest_position': row['manifest_position']}
    for role in ('robot', 'human'):
        with av.open(row[f'{role}_source']) as container:
            stream = container.streams.video[0]
            fps = float(stream.average_rate) if stream.average_rate else None
            result[role] = {'fps': fps, 'width': stream.width, 'height': stream.height,
                            'codec': stream.codec_context.name,
                            'duration_from_frames_seconds': row[f'{role}_frames'] / fps if fps else None}
    return result


def make_figures(rows, dataset_names, summary, asset_dir):
    asset_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    colors = ['#4477AA', '#66AABB', '#228833', '#CCBB44', '#EE6677', '#AA3377']
    fig, ax = plt.subplots(figsize=(9, 4.8), layout='constrained')
    positions = np.arange(len(dataset_names))
    ax.barh(positions, [summary[name]['scanned'] for name in dataset_names], color='#d9dde2', label='Input pairs')
    ax.barh(positions, [summary[name]['accepted'] for name in dataset_names], color=colors, label='Retained pairs')
    for i, name in enumerate(dataset_names):
        stats = summary[name]
        ax.text(stats['scanned'] + 70, i, f'{stats["accepted"]:,} / {stats["scanned"]:,}', va='center', fontsize=9)
    ax.set(yticks=positions, yticklabels=dataset_names, xlabel='Pair records', title='HumanGen pair composition and filtering')
    ax.set_xlim(0, max(stats['scanned'] for stats in summary.values()) * 1.2)
    ax.invert_yaxis()
    ax.legend(loc='lower right')
    fig.savefig(asset_dir / 'composition.png', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), layout='constrained')
    for name, color in zip(dataset_names, colors):
        group = [row for row in rows if row['dataset'] == name]
        for ax, role in zip(axes, ('robot', 'human')):
            values = np.sort([row[f'{role}_frames'] for row in group])
            ax.step(values, np.arange(1, len(values) + 1) / len(values), where='post', label=name, color=color)
            ax.set(xscale='log', xlabel='Video frames (log scale)', ylabel='Cumulative fraction', title=f'{role.title()} frame-count distribution')
            ax.grid(alpha=0.2)
    axes[1].legend(fontsize=8, loc='lower right')
    fig.savefig(asset_dir / 'frame_counts.png', dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8), layout='constrained')
    for offset, key, label, color in (
        (-0.2, 'human_source', 'Unique human file paths / pairs', '#4477AA'),
        (0.2, '_trajectory', 'Original robot trajectories / pairs', '#EE6677'),
    ):
        values = []
        for name in dataset_names:
            group = [row for row in rows if row['dataset'] == name]
            values.append(100 * len({row[key] for row in group}) / len(group))
        bars = ax.bar(positions + offset, values, width=0.38, label=label, color=color)
        ax.bar_label(bars, fmt='%.1f%%', padding=3, fontsize=8)
    ax.set(xticks=positions, xticklabels=dataset_names, ylim=(0, 130), ylabel='Unique sources / pair records (%)',
           title='Pair records versus unique source identities')
    ax.legend(fontsize=9, loc='upper left')
    fig.savefig(asset_dir / 'source_reuse.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--media-samples-per-dataset', type=int, default=30)
    parser.add_argument('--seed', type=int, default=20261016)
    args = parser.parse_args()
    if args.media_samples_per_dataset < 1:
        parser.error('--media-samples-per-dataset must be positive')
    settings = load_settings(args.config)
    output = settings['output_root']
    with (output / 'pairs.jsonl').open(encoding='utf-8') as handle:
        rows = [json.loads(line) for line in handle]
    summary = json.loads((output / 'summary.json').read_text())
    paths = json.loads(video_paths_file(settings).read_text())
    assert paths == [row['robot_episode'] for row in rows]
    hashes = {}
    for dataset in settings['datasets']:
        path = settings['source_root'] / 'icl_configs' / f'ICL_config_{dataset}.json'
        raw = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        samples = json.loads(raw)['samples']
        for row in rows:
            if row['dataset'] != dataset:
                continue
            sample = samples[row['manifest_position']]
            assert (row['run'], row['sample']) == (sample['run'], sample['sample'])
            row['_clip'], row['_trajectory'], row['_task'] = source_identity(dataset, sample)
            row['_physics'], row['_semantic'] = sample.get('physics_score'), sample.get('semantic_score')
            row['_generator'] = Path(row['human_source']).stem.removeprefix('generated_video_')
    rng = np.random.default_rng(args.seed)
    test_indices, sampled_rows = set(), []
    for dataset in settings['datasets']:
        indices = np.array([i for i, row in enumerate(rows) if row['dataset'] == dataset])
        test_indices.update(int(i) for i in rng.permutation(indices)[:round(len(indices) * 0.2)])
        sampled_rows.extend(rows[int(i)] for i in rng.choice(indices, min(args.media_samples_per_dataset, len(indices)), replace=False))
    print('Computing full-manifest statistics and source-overlap simulation...', flush=True)
    stats = {'seed': args.seed, 'summary': summary, 'input_sha256': hashes,
             'all': analyze_group(rows), 'components': components(rows),
             'naive_80_20_split_simulation': source_overlap(rows, test_indices),
             'by_dataset': {}, 'media_metadata_sample': []}
    for dataset in settings['datasets']:
        indices = [i for i, row in enumerate(rows) if row['dataset'] == dataset]
        group = [rows[i] for i in indices]
        test = {j for j, i in enumerate(indices) if i in test_indices}
        stats['by_dataset'][dataset] = analyze_group(group)
        stats['by_dataset'][dataset]['naive_80_20_split_simulation'] = source_overlap(group, test)
    with ThreadPoolExecutor(max_workers=16) as executor:
        stats['media_metadata_sample'] = list(executor.map(probe_pair, sampled_rows))
    for dataset in settings['datasets']:
        sample = [row for row in stats['media_metadata_sample'] if row['dataset'] == dataset]
        result = {'pairs': len(sample)}
        for role in ('robot', 'human'):
            result[role] = {
                'fps_counts': dict(Counter(str(row[role]['fps']) for row in sample)),
                'resolution_counts': dict(Counter(f'{row[role]["width"]}x{row[role]["height"]}' for row in sample)),
                'codec_counts': dict(Counter(row[role]['codec'] for row in sample)),
                'duration_seconds': describe([row[role]['duration_from_frames_seconds'] for row in sample if row[role]['fps']]),
            }
        result['robot_human_duration_ratio'] = describe([
            row['robot']['duration_from_frames_seconds'] / row['human']['duration_from_frames_seconds']
            for row in sample if row['robot']['fps'] and row['human']['fps']
        ])
        stats['by_dataset'][dataset]['sampled_media_metadata'] = result
    for name in ('pairs.jsonl', 'skipped.jsonl', 'summary.json', 'validation_report.json', 'HumanGen_video_paths.json'):
        hashes[str(output / name)] = hashlib.sha256((output / name).read_bytes()).hexdigest()
    (output / 'analysis_stats.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    assets = Path(__file__).resolve().parent / 'analysis_assets'
    make_figures(rows, settings['datasets'], summary['datasets'], assets)
    print(f'Statistics: {output / "analysis_stats.json"}', flush=True)
    print(f'Figures: {assets}', flush=True)


if __name__ == '__main__':
    main()
