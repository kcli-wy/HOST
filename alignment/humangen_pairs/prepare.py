"""Export explicit HumanGen robot-to-human pairs for AlignmentDataset."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import time
from threading import Lock

import av
import decord

from common import DEFAULT_CONFIG, DATASET_NAME, load_settings, video_paths_file

TRANSCODE_LOCKS = defaultdict(Lock)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def read_task(task_dir):
    info = read_json(task_dir / 'meta/info.json')
    with (task_dir / 'meta/episodes.jsonl').open(encoding='utf-8') as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return task_dir, info, rows


def find_tasks(dataset_root):
    """Released HumanGen tasks live one or two levels below each robot root."""
    tasks = []
    for child in sorted(dataset_root.iterdir()):
        if not child.is_dir() or child.name.startswith('.'):
            continue
        if (child / 'meta/episodes.jsonl').is_file():
            tasks.append(child)
        else:
            tasks.extend(
                sub for sub in sorted(child.iterdir())
                if sub.is_dir() and (sub / 'meta/episodes.jsonl').is_file()
            )
    return tasks


def build_export_index(source_root, dataset, workers):
    dataset_root = source_root / f'{dataset}_data'
    tasks = find_tasks(dataset_root)
    by_position = {}
    by_source = {}
    robotwin_tasks = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for task_dir, info, rows in executor.map(read_task, tasks):
            if dataset == 'robotwin':
                robotwin_tasks[str(task_dir.relative_to(dataset_root))] = (
                    task_dir, info, {row['episode_index']: row for row in rows}
                )
                continue
            for row in rows:
                source_rel = row.get('source', {}).get('video_rel_path')
                if source_rel:
                    by_source.setdefault(source_rel, []).append((task_dir, info, row))
                position = row.get('icl', {}).get('manifest_position')
                if position is None:
                    # Some exports also contain robot-only episodes, outside the ICL pairs.
                    continue
                by_position.setdefault(position, []).append((task_dir, info, row))
    print(f'{dataset}: indexed {len(tasks)} tasks, {len(by_position)} exported pair identities', flush=True)
    return by_position, by_source, robotwin_tasks


@lru_cache(maxsize=8192)
def probe_video(path):
    if not path.is_file():
        raise FileNotFoundError(str(path))
    reader = decord.VideoReader(str(path), num_threads=1)
    frames = len(reader)
    if frames < 1:
        raise ValueError(f'Empty video: {path}')
    reader[0].asnumpy()
    return frames


def compatible_video(path, output, dataset):
    """Keep readable sources; encode AV1 locally because Decord's FFmpeg lacks AV1."""
    try:
        return path, probe_video(path)
    except FileNotFoundError:
        raise
    except Exception:
        with av.open(str(path)) as source:
            stream = source.streams.video[0]
            if stream.codec_context.name not in ('av1', 'libdav1d', 'libaom-av1'):
                raise
    digest = hashlib.sha256(str(path).encode()).hexdigest()[:24]
    target = output / 'compatible_videos' / dataset / f'{digest}.mp4'
    with TRANSCODE_LOCKS[str(target)]:
        if not target.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix('.mp4.tmp')
            try:
                with av.open(str(path)) as source, av.open(str(temporary), 'w', format='mp4') as destination:
                    input_stream = source.streams.video[0]
                    input_stream.codec_context.thread_count = 1
                    rate = input_stream.average_rate
                    if rate is None or rate <= 0:
                        raise ValueError(f'AV1 source has no usable frame rate: {path}')
                    encoder = destination.add_stream('libx264', rate=rate)
                    encoder.width = input_stream.codec_context.width
                    encoder.height = input_stream.codec_context.height
                    encoder.pix_fmt = 'yuv420p'
                    encoder.options = {'crf': '18', 'preset': 'veryfast', 'threads': '1'}
                    for number, frame in enumerate(source.decode(input_stream)):
                        frame.pts = number
                        frame.time_base = 1 / rate
                        for packet in encoder.encode(frame):
                            destination.mux(packet)
                    for packet in encoder.encode():
                        destination.mux(packet)
                probe_video(temporary)
                temporary.replace(target)
            except Exception:
                temporary.unlink(missing_ok=True)
                raise
        return target, probe_video(target)


def resolve_record(source_root, dataset, position, sample, by_position, by_source, robotwin_tasks):
    """Use exported identities, never old episode numbers for renumbered datasets."""
    status = sample['status_info']
    source_rel = status['video_rel_path']
    view = PurePosixPath(source_rel).parent.name
    if dataset == 'robotwin':
        # RoboTwin retains original episode IDs and has no `icl` in episodes.jsonl.
        task_name, tail = source_rel.split('/videos/', 1)
        task_dir, info, rows = robotwin_tasks[task_name]
        episode_index = int(PurePosixPath(tail).stem.removeprefix('episode_'))
        row = rows[episode_index]
        robot_video = task_dir / 'videos' / tail
    else:
        if position in by_position:
            candidates = [
                entry for entry in by_position[position]
                if all(entry[2]['icl'].get(key) == sample[key]
                       for key in ('run', 'sample', 'human_video_path'))
                and entry[2]['source']['video_rel_path'] == source_rel
            ]
        else:
            candidates = by_source.get(source_rel, [])
        if len(candidates) > 1:
            source_range = status.get('interval')
            match = re.fullmatch(r'episode_\d+_(\d+)_(\d+)\.mp4', PurePosixPath(source_rel).name)
            if source_range is None and match:
                source_range = [int(value) for value in match.groups()]
            if source_range is not None:
                candidates = [entry for entry in candidates
                              if entry[2]['source'].get('frame_range') == source_range]
        if len(candidates) != 1:
            raise ValueError(f'Expected one exact exported pair, found {len(candidates)} at {dataset}/{position}: {source_rel}')
        task_dir, info, row = candidates[0]
        if row['source']['video_rel_path'] != source_rel:
            raise ValueError(f'Robot source mismatch at {dataset}/{position}')
        episode_index = row['episode_index']
        robot_video = task_dir / info['video_path'].format(
            episode_chunk=episode_index // info['chunks_size'],
            episode_index=episode_index, video_key=view,
        )
    human_rel = PurePosixPath(sample['human_video_path'])
    if human_rel.parts[0] == 'icl_raw_data':
        human_rel = PurePosixPath(*human_rel.parts[1:])
    if human_rel.is_absolute() or '..' in human_rel.parts or human_rel.parts[0] != 'human_data':
        raise ValueError(f'Unexpected human video path: {human_rel}')
    return {
        'dataset': dataset,
        'manifest_position': position,
        'run': sample['run'],
        'sample': sample['sample'],
        'human_source': str(source_root / human_rel),
        'robot_source': str(robot_video),
        'robot_metadata': str(task_dir / 'meta/episodes.jsonl'),
        'robot_episode_index': episode_index,
        'robot_source_view': view,
        'robot_source_frame_range': row.get('source', {}).get('frame_range'),
        'robot_expected_frames': row['length'],
        'robot_instruction': sample.get('robot_task_name') or 'Perform the task.',
        'human_instruction': sample.get('human_task_name') or 'Perform the task.',
    }


def check_record(record, min_frames, output):
    try:
        record = dict(record)
        for role in ('robot', 'human'):
            source = Path(record[f'{role}_source'])
            training_video, frames = compatible_video(source, output, record['dataset'])
            record[f'{role}_frames'] = frames
            if training_video != source:
                record[f'{role}_original_source'] = str(source)
                record[f'{role}_source'] = str(training_video)
        robot_frames, human_frames = record['robot_frames'], record['human_frames']
        if robot_frames != record['robot_expected_frames']:
            return None, 'frame_count_mismatch', f'robot metadata={record["robot_expected_frames"]}, decoded={robot_frames}'
        if min(robot_frames, human_frames) < min_frames:
            return None, 'too_short', f'robot={robot_frames}, human={human_frames}, minimum={min_frames}'
        return record, None, None
    except FileNotFoundError as exc:
        return None, 'missing_video', str(exc)
    except Exception as exc:
        return None, 'unreadable_video', f'{role}: {exc}'


def write_episode(path, source, instruction, peer):
    path.mkdir(parents=True, exist_ok=True)
    video = path / 'images.mp4'
    try:
        video.symlink_to(source)
    except FileExistsError:
        if not video.is_symlink() or video.readlink() != Path(source):
            raise ValueError(f'Refusing to replace an existing video or different symlink: {video}')
    (path / 'instruction.txt').write_text(instruction + '\n', encoding='utf-8')
    peers_json = json.dumps({'same': [str(peer)]}, ensure_ascii=False) + '\n'
    for name in ('task_paths.json', 'task_paths_eval.json'):
        (path / name).write_text(peers_json, encoding='utf-8')


def export_record(record, min_frames, output):
    record, reason, detail = check_record(record, min_frames, output)
    if record is not None:
        pair_dir = output / 'episodes' / record['dataset'] / f'{record["manifest_position"]:06d}'
        robot, human = pair_dir / 'robot', pair_dir / 'human'
        write_episode(robot, record['robot_source'], record['robot_instruction'], human)
        write_episode(human, record['human_source'], record['human_instruction'], robot)
        record.update(robot_episode=str(robot), human_episode=str(human))
    return record, reason, detail


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--output-root', type=Path, help='Override output directory (relative to current directory)')
    parser.add_argument('--limit-per-dataset', type=int, help='Inspect only the first N manifest records per dataset for a smoke run')
    args = parser.parse_args()
    settings = load_settings(args.config)
    if args.output_root:
        settings['output_root'] = args.output_root.resolve()
    if args.limit_per_dataset is not None and args.limit_per_dataset < 1:
        parser.error('--limit-per-dataset must be positive')
    output = settings['output_root']
    output.mkdir(parents=True, exist_ok=True)
    min_frames = settings['training']['num_frames']
    video_paths, cameras, summary = [], {}, {}
    start = time.monotonic()
    pairs_tmp = output / 'pairs.jsonl.tmp'
    skipped_tmp = output / 'skipped.jsonl.tmp'
    with pairs_tmp.open('w', encoding='utf-8') as pairs, skipped_tmp.open('w', encoding='utf-8') as skipped:
        for dataset in settings['datasets']:
            manifest_path = settings['source_root'] / 'icl_configs' / f'ICL_config_{dataset}.json'
            samples = read_json(manifest_path)['samples']
            by_position, by_source, robotwin_tasks = build_export_index(settings['source_root'], dataset, settings['workers'])
            counts = Counter()
            records = []
            selected = samples[:args.limit_per_dataset] if args.limit_per_dataset else samples
            for position, sample in enumerate(selected):
                counts['scanned'] += 1
                try:
                    records.append(resolve_record(settings['source_root'], dataset, position, sample, by_position, by_source, robotwin_tasks))
                except (KeyError, ValueError) as exc:
                    counts['unresolved_pair'] += 1
                    skipped.write(json.dumps({'dataset': dataset, 'manifest_position': position,
                                              'reason': 'unresolved_pair', 'detail': str(exc)}, ensure_ascii=False) + '\n')
            with ThreadPoolExecutor(max_workers=settings['workers']) as executor:
                checks = executor.map(lambda record: export_record(record, min_frames, output), records)
                for number, (original, (record, reason, detail)) in enumerate(zip(records, checks), 1):
                    if record is None:
                        counts[reason] += 1
                        skipped.write(json.dumps({**original, 'reason': reason, 'detail': detail}, ensure_ascii=False) + '\n')
                    else:
                        pairs.write(json.dumps(record, ensure_ascii=False) + '\n')
                        video_paths.append(record['robot_episode'])
                        cameras[str(Path(record['robot_episode']).parent)] = ['images']
                        counts['accepted'] += 1
                        for role in ('robot', 'human'):
                            if f'{role}_original_source' in record:
                                counts[f'transcoded_{role}'] += 1
                    if number % 500 == 0:
                        print(f'{dataset}: checked {number}/{len(records)}, accepted={counts["accepted"]}', flush=True)
            summary[dataset] = dict(counts)
            print(f'{dataset}: {dict(counts)}', flush=True)
    if not video_paths:
        raise RuntimeError(f'No usable pairs. Inspect {skipped_tmp}; no training index was written.')
    pairs_tmp.replace(output / 'pairs.jsonl')
    skipped_tmp.replace(output / 'skipped.jsonl')
    write_json(video_paths_file(settings), video_paths)
    write_json(output / 'cam_mapping' / f'{DATASET_NAME}_cam_mapping.json', cameras)
    write_json(output / 'summary.json', {
        'source_root': str(settings['source_root']),
        'output_root': str(output), 'min_frames': min_frames,
        'limit_per_dataset': args.limit_per_dataset,
        'pairs': len(video_paths), 'datasets': summary,
        'elapsed_seconds': round(time.monotonic() - start, 2),
    })
    print(f'Exported {len(video_paths)} robot-to-human pairs to {output}', flush=True)
    print(f'Training index: {video_paths_file(settings)}', flush=True)


if __name__ == '__main__':
    main()
