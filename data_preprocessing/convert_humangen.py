"""Convert approved HumanGen robot/human pairs to the HOST alignment layout."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
import threading
import time


SOURCES = ('agibot', 'interna1', 'oxe', 'robocoin', 'robomind', 'robotwin')


def read_json(path):
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write('\n')


def contained(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f'Path outside source root: {relative}')
    return path


def probe_video(path):
    """Decode every frame; container frame counts alone are insufficient."""
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = stream.average_rate
        if fps is None or fps <= 0:
            raise ValueError(f'Invalid FPS: {path}')
        count = 0
        size = None
        for frame in container.decode(stream):
            current = (frame.width, frame.height)
            if size is not None and current != size:
                raise ValueError(f'Changing frame size: {path}')
            size = current
            count += 1
        if count < 24:
            raise ValueError(f'Video too short for alignment: {count} frames: {path}')
        return {'frames': count, 'fps': str(fps), 'width': size[0],
                'height': size[1], 'codec': stream.codec_context.name}


def check_decord(path, expected):
    import decord

    reader = decord.VideoReader(str(path), num_threads=1)
    if len(reader) != expected['frames']:
        raise ValueError(f'Decord frame count mismatch: {path}')
    # Check every frame, including codecs whose container opens but cannot decode.
    for start in range(0, len(reader), 32):
        frames = reader.get_batch(list(range(start, min(start + 32, len(reader))))).asnumpy()
        if frames.shape[1:3] != (expected['height'], expected['width']):
            raise ValueError(f'Decord frame size mismatch: {path}')


def convert_video(source, target, expected):
    """Copy readable MP4s; transcode unsupported codecs without temporal resampling."""
    import av
    import decord

    try:
        check_decord(source, expected)
        shutil.copyfile(source, target)
        method = 'copy'
    except (RuntimeError, ValueError, decord.DECORDError):
        with av.open(str(source)) as incoming, av.open(str(target), 'w') as outgoing:
            encoder = outgoing.add_stream('libx264', rate=Fraction(expected['fps']))
            encoder.width = expected['width']
            encoder.height = expected['height']
            encoder.pix_fmt = 'yuv420p'
            encoder.options = {'crf': '18', 'preset': 'fast'}
            for index, frame in enumerate(incoming.decode(video=0)):
                frame.pts = index
                frame.time_base = 1 / Fraction(expected['fps'])
                for packet in encoder.encode(frame):
                    outgoing.mux(packet)
            for packet in encoder.encode():
                outgoing.mux(packet)
        method = 'transcode_h264'
    actual = probe_video(target)
    for key in ('frames', 'fps', 'width', 'height'):
        if actual[key] != expected[key]:
            raise ValueError(f'{key} changed during conversion: {source}')
    check_decord(target, expected)
    return {**actual, 'method': method}


class HumanGenConverter:
    def __init__(self, source_root, output_root, resume=False):
        self.source_root = source_root.resolve()
        self.output_root = output_root.resolve()
        self.repo_cache = {}
        self.robot_index = {}
        self.probes = {}
        self.episodes = {}
        self.resume = resume
        self.locks = defaultdict(threading.Lock)

    def robot_episode(self, family, sample):
        relative = sample['status_info']['video_rel_path']
        family_root = self.source_root / f'{family}_data'
        if family == 'robotwin':
            repo = contained(family_root, relative.split('/videos/', 1)[0])
            match = re.search(r'episode_(\d+)\.mp4$', relative)
            if match is None:
                raise ValueError(f'Unexpected RobotWin video path: {relative}')
            index = int(match.group(1))
        else:
            # Unpacked repository names differ from paths in the original manifests.
            repo_relative = relative.split('/videos/', 1)[0]
            repo_key = (family, repo_relative)
            if repo_key not in self.robot_index:
                by_identity = defaultdict(list)
                slug = repo_relative.replace('/', '__')
                direct = family_root / repo_relative / 'meta/episodes.jsonl'
                metadata_files = [direct] if direct.is_file() else []
                metadata_files.extend(family_root.glob(f'*/{slug}/meta/episodes.jsonl'))
                if not metadata_files:
                    raise ValueError(f'Cannot locate unpacked repository for {repo_relative}')
                for metadata in sorted(metadata_files):
                    with open(metadata, encoding='utf-8') as handle:
                        for line in handle:
                            episode = json.loads(line)
                            icl = episode.get('icl', {})
                            if icl:
                                by_identity[(icl.get('run'), icl.get('sample'))].append(
                                    (metadata.parent.parent, episode))
                self.robot_index[repo_key] = by_identity
            matches = self.robot_index[repo_key].get((sample['run'], sample['sample']), [])
            matches = [(repo, episode) for repo, episode in matches
                       if episode.get('source', {}).get('video_rel_path') == relative]
            if len(matches) != 1:
                raise ValueError(f'Expected one current robot episode, got {len(matches)}: {relative}')
            repo, episode = matches[0]
            index = episode['episode_index']
        if repo not in self.repo_cache:
            self.repo_cache[repo] = read_json(repo / 'meta/info.json')
        info = self.repo_cache[repo]
        views = {}
        for key, feature in info['features'].items():
            if feature.get('dtype') != 'video':
                continue
            view = key.removeprefix('observation.images.')
            if view in views or '/' in view or view in ('.', '..'):
                raise ValueError(f'Invalid or duplicate camera name: {key}')
            views[view] = contained(repo, info['video_path'].format(
                episode_chunk=index // info['chunks_size'], episode_index=index, video_key=key))
        if not views:
            raise ValueError(f'No video features: {repo}')
        identity = f'{family}/robot/{relative}'
        # Preserve the actual unpacked source hierarchy above the paired episode.
        destination = self.output_root / repo.relative_to(self.source_root) / f'episode_{index:06d}'
        return identity, destination, views

    def human_episode(self, family, sample):
        relative = sample['human_video_path'].removeprefix('icl_raw_data/')
        source = contained(self.source_root, relative)
        identity = f'{family}/human/{relative}'
        digest = hashlib.sha256(identity.encode()).hexdigest()[:12]
        # Parent follows the actual video path, so reuse across manifest records deduplicates.
        run = source.parent.parent.parent.name
        destination = self.output_root / family / f'human_{run}' / f'human_{digest}'
        return identity, destination, {'images': source}

    def inspect(self, views):
        metadata = {}
        for view, path in views.items():
            with self.locks[f'probe:{path}']:
                if path not in self.probes:
                    self.probes[path] = probe_video(path)
            metadata[view] = self.probes[path]
        if len({(v['frames'], v['fps']) for v in metadata.values()}) != 1:
            raise ValueError(f'Unsynchronized camera frame counts or FPS: {metadata}')
        return metadata

    def materialize(self, identity, destination, views, metadata, instruction):
        with self.locks[identity]:
            return self._materialize(identity, destination, views, metadata, instruction)

    def _materialize(self, identity, destination, views, metadata, instruction):
        if identity in self.episodes:
            return self.episodes[identity]
        source_views = {view: str(path) for view, path in views.items()}
        if destination.exists():
            if not self.resume:
                raise FileExistsError(f'Existing episode requires --resume: {destination}')
            record = read_json(destination / 'episode_metadata.json')
            if record['identity'] != identity or record['source_views'] != source_views:
                raise ValueError(f'Resume source mismatch: {destination}')
            for view, video in record['videos'].items():
                if (destination / f'{view}.mp4').stat().st_size != video['file_size']:
                    raise ValueError(f'Incomplete resumed video: {destination}/{view}')
                for field in ('frames', 'fps', 'width', 'height'):
                    if video[field] != metadata[view][field]:
                        raise ValueError(f'Resume metadata mismatch: {destination}/{view}/{field}')
            self.episodes[identity] = record
            return record
        destination.parent.mkdir(parents=True, exist_ok=True)
        # An episode is published only after all its cameras pass verification.
        temporary = Path(tempfile.mkdtemp(prefix='.converting-', dir=destination.parent))
        try:
            converted = {view: convert_video(path, temporary / f'{view}.mp4', metadata[view])
                         for view, path in views.items()}
            for view, video in converted.items():
                video['file_size'] = (temporary / f'{view}.mp4').stat().st_size
            (temporary / 'instruction.txt').write_text(instruction + '\n', encoding='utf-8')
            result = {'path': str(destination), 'identity': identity,
                      'source_views': source_views, 'videos': converted, 'instruction': instruction}
            write_json(temporary / 'episode_metadata.json', result)
            temporary.rename(destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        self.episodes[identity] = result
        return result

    def run(self, sources, pairs_per_source, dry_run=False, workers=1, task_paths=None):
        import decord

        report = {'source_root': str(self.source_root), 'output_root': str(self.output_root),
                  'layout': 'paired_v1', 'task_paths': task_paths or [],
                  'dry_run': dry_run, 'workers': workers, 'sources': {}, 'pairs': [], 'failures': []}
        peers = defaultdict(set)
        by_family = defaultdict(set)
        started = time.time()
        def save_progress(status, family=None):
            if not dry_run:
                progress = {'status': status, 'current_source': family, 'workers': workers,
                            'selected_pairs': len(report['pairs']), 'failed_pairs': len(report['failures']),
                            'sources': report['sources'], 'elapsed_seconds': round(time.time() - started),
                            'updated_at': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
                temporary = self.output_root / '.progress.json.tmp'
                write_json(temporary, progress)
                temporary.replace(self.output_root / 'progress.json')

        for family in sources:
            manifest = self.source_root / 'icl_configs' / f'ICL_config_{family}.json'
            samples = read_json(manifest)['samples']
            selected = [(position, sample) for position, sample in enumerate(samples)
                        if not task_paths or sample['status_info']['video_rel_path'].split('/videos/', 1)[0] in task_paths]
            converted = 0
            report['sources'][family] = {'manifest_pairs': len(samples), 'eligible_pairs': len(selected),
                                         'selected_pairs': 0, 'episodes': 0}
            save_progress('running', family)

            def convert_pair(item):
                position, sample = item
                try:
                    robot = self.robot_episode(family, sample)
                    human = self.human_episode(family, sample)
                    pair_id = hashlib.sha256(f'{family}:{position}'.encode()).hexdigest()[:12]
                    pair_dir = robot[1].with_name(f'{robot[1].name}_{pair_id}')
                    robot_source_identity, human_source_identity = robot[0], human[0]
                    robot = (f'{family}/robot/{pair_id}', pair_dir / 'robot', robot[2])
                    human = (f'{family}/human/{pair_id}', pair_dir / 'human', human[2])
                    robot_meta = self.inspect(robot[2])
                    human_meta = self.inspect(human[2])
                    edge = {'source': family, 'manifest': str(manifest), 'manifest_position': position,
                            'pair_id': pair_id, 'pair_dir': str(pair_dir),
                            'robot_source_identity': robot_source_identity, 'human_source_identity': human_source_identity,
                            'run': sample['run'], 'sample': sample['sample'],
                            'sample_id': sample['sample_id'], 'robot': str(robot[1]), 'human': str(human[1]),
                            'robot_task_name': sample['robot_task_name'],
                            'human_task_name': sample['human_task_name'],
                            'instructions_differ': sample['robot_task_name'] != sample['human_task_name'],
                            'robot_source_record': sample['status_info'],
                            'human_source_path': sample['human_video_path'],
                            'robot_views': robot_meta, 'human_views': human_meta}
                    if not dry_run:
                        self.materialize(*robot, robot_meta, sample['robot_task_name'])
                        self.materialize(*human, human_meta, sample['human_task_name'])
                        write_json(pair_dir / 'pair_info.json', edge)
                    return edge, None
                except (OSError, ValueError, RuntimeError, decord.DECORDError) as error:
                    return None, {'source': family, 'manifest_position': position, 'error': str(error)}

            # Bounded pilots stay lazy; full runs use parallel decoding and copying.
            with ThreadPoolExecutor(max_workers=workers) as executor:
                results = (executor.map(convert_pair, selected) if not pairs_per_source
                           else map(convert_pair, selected))
                for edge, failure in results:
                    if failure:
                        report['failures'].append(failure)
                        if len(report['failures']) <= 10:
                            print(f'{family}: skipping manifest record {failure["manifest_position"]}: {failure["error"]}', flush=True)
                    else:
                        by_family[family].update((edge['robot'], edge['human']))
                        if not dry_run:
                            peers[edge['robot']].add(edge['human'])
                            peers[edge['human']].add(edge['robot'])
                        report['pairs'].append(edge)
                        converted += 1
                        print(f'{family}: pair {converted}: {Path(edge["pair_dir"]).name}/robot <-> human', flush=True)
                    report['sources'][family].update(selected_pairs=converted, episodes=len(by_family[family]))
                    if (len(report['pairs']) + len(report['failures'])) % 10 == 0:
                        save_progress('running', family)
                    if pairs_per_source and converted >= pairs_per_source:
                        break
            report['sources'][family] = {'manifest_pairs': len(samples), 'eligible_pairs': len(selected), 'selected_pairs': converted,
                                         'episodes': len(by_family[family])}
        if not dry_run:
            for episode, references in peers.items():
                for filename in ('task_paths.json', 'task_paths_eval.json'):
                    write_json(Path(episode) / filename, {'same': sorted(references)})
            episode_records = {record['path']: record for record in self.episodes.values()
                               if record['path'] in peers}
            for family, paths in by_family.items():
                write_json(self.output_root / f'{family}_video_paths.json', sorted(paths))
                mapping = {}
                for episode in paths:
                    cameras = list(episode_records[episode]['videos'])
                    mapping[episode] = cameras
                write_json(self.output_root / 'cam_mapping' / f'{family}_cam_mapping.json', mapping)
            write_json(self.output_root / 'source_index.json', {'pairs': report['pairs'],
                       'episodes': list(episode_records.values())})
            write_json(self.output_root / 'conversion_report.json', report)
        missing = [family for family, stats in report['sources'].items()
                   if not stats['selected_pairs'] or (pairs_per_source and stats['selected_pairs'] < pairs_per_source)]
        if missing:
            save_progress('incomplete')
            raise RuntimeError(f'Could not satisfy pair count for: {missing}; see conversion_report.json')
        save_progress('completed_with_failures' if report['failures'] else 'completed')
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=Path('/AOSS/HumanGen'))
    parser.add_argument('--output-root', type=Path, default=Path('/tmp/HumanGen_alignment_paired_pilot'))
    parser.add_argument('--sources', nargs='+', choices=SOURCES, default=list(SOURCES))
    parser.add_argument('--pairs-per-source', type=int, default=1, help='0 selects all manifest pairs')
    parser.add_argument('--task-path', action='append', help='Exact task path before /videos/ in source manifests; repeatable')
    parser.add_argument('--dry-run', action='store_true', help='Decode source videos and preview; write nothing')
    parser.add_argument('--workers', type=int, default=1, help='Concurrent pair conversions in full runs')
    parser.add_argument('--resume', action='store_true', help='Reuse verified episode checkpoints from this run')
    args = parser.parse_args()
    if args.pairs_per_source < 0:
        parser.error('--pairs-per-source must be nonnegative')
    if args.workers < 1:
        parser.error('--workers must be positive')
    source, output = args.source_root.resolve(), args.output_root.resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        parser.error('Source and output directories must not overlap')
    if not args.dry_run:
        lock_key = hashlib.sha256(str(output).encode()).hexdigest()[:16]
        run_lock = open(Path(tempfile.gettempdir()) / f'host-humangen-{lock_key}.lock', 'a')
        try:
            fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error(f'Another conversion is already running for {output}')
        # Refuse reuse rather than overwrite earlier outputs or partial runs.
        config = {'source_root': str(source), 'sources': args.sources, 'pairs_per_source': args.pairs_per_source,
                  'layout': 'paired_v1', 'task_paths': args.task_path or []}
        if args.resume:
            if read_json(output / 'conversion_config.json') != config:
                parser.error('--resume configuration does not match the existing run')
        else:
            output.mkdir(parents=True, exist_ok=False)
            write_json(output / 'conversion_config.json', config)
    converter = HumanGenConverter(source, output, resume=args.resume)
    report = converter.run(args.sources, args.pairs_per_source, args.dry_run, args.workers, args.task_path)
    print(json.dumps(report['sources'], indent=2))


if __name__ == '__main__':
    main()
