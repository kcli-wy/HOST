"""Standalone regression tests for HumanGen data-pair generation."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from convert_humangen import HumanGenConverter, convert_video, probe_video, read_json, write_json


class PairingTests(unittest.TestCase):
    def test_decord_specific_error_triggers_verified_transcode(self):
        import av
        import decord
        import numpy as np

        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / 'source.mp4'
            target = Path(temporary) / 'target.mp4'
            with av.open(str(source), 'w') as container:
                stream = container.add_stream('libx264', rate=30)
                stream.width = stream.height = 32
                stream.pix_fmt = 'yuv420p'
                for i in range(24):
                    frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), i, dtype=np.uint8), format='rgb24')
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            metadata = probe_video(source)
            with patch('convert_humangen.check_decord', side_effect=[decord.DECORDError('unsupported codec'), None]):
                result = convert_video(source, target, metadata)
            self.assertEqual(result['method'], 'transcode_h264')
            self.assertEqual(result['frames'], 24)
            self.assertEqual(result['fps'], '30')

    def test_reused_human_has_a_role_directory_in_each_pair(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root / 'source', root / 'output'
            samples = [{'run': 'run1', 'sample': f'sample{i}', 'sample_id': i,
                        'status_info': {'video_rel_path': f'clip{i}'},
                        'human_video_path': 'same_human.mp4',
                        'robot_task_name': f'Robot wording {i}', 'human_task_name': 'Human wording'}
                       for i in range(2)]
            write_json(source / 'icl_configs/ICL_config_agibot.json', {'samples': samples})
            converter = HumanGenConverter(source, output)
            metadata = {'frames': 24, 'fps': '30', 'width': 32, 'height': 32, 'codec': 'h264'}

            def robot(family, sample):
                i = sample['sample_id']
                return f'robot{i}', output / 'agibot/robot_task' / f'robot{i}', {'head': root / f'robot{i}.mp4'}

            def human(family, sample):
                return 'human', output / 'agibot/human_task/human', {'images': root / 'human.mp4'}

            def fake_convert(source, target, expected):
                target.touch()
                return {**expected, 'method': 'copy'}

            with patch.object(converter, 'robot_episode', side_effect=robot), \
                    patch.object(converter, 'human_episode', side_effect=human), \
                    patch('convert_humangen.probe_video', return_value=metadata), \
                    patch('convert_humangen.convert_video', side_effect=fake_convert):
                report = converter.run(['agibot'], 0, workers=4)
            index = read_json(output / 'source_index.json')
            self.assertEqual(len(index['episodes']), 4)
            self.assertEqual(len(report['pairs']), 2)
            for pair in index['pairs']:
                robot_path, human_path = Path(pair['robot']), Path(pair['human'])
                self.assertEqual(robot_path.parent, human_path.parent)
                self.assertEqual(robot_path.name, 'robot')
                self.assertEqual(human_path.name, 'human')
                self.assertEqual(read_json(robot_path / 'task_paths.json'), {'same': [str(human_path)]})
                self.assertEqual(read_json(human_path / 'task_paths.json'), {'same': [str(robot_path)]})
                self.assertEqual(read_json(robot_path.parent / 'pair_info.json'), pair)
            mapping = read_json(output / 'cam_mapping/agibot_cam_mapping.json')
            self.assertEqual(len(mapping), 4)
            self.assertEqual(set(tuple(v) for v in mapping.values()), {('head',), ('images',)})
            resumed = HumanGenConverter(source, output, resume=True)
            with patch.object(resumed, 'robot_episode', side_effect=robot), \
                    patch.object(resumed, 'human_episode', side_effect=human), \
                    patch('convert_humangen.probe_video', return_value=metadata), \
                    patch('convert_humangen.convert_video', side_effect=AssertionError('Unexpected re-conversion')):
                repeated = resumed.run(['agibot'], 0, workers=4)
            self.assertEqual(len(repeated['pairs']), 2)
            self.assertEqual(repeated['sources']['agibot']['episodes'], 4)
            preview = HumanGenConverter(source, root / 'preview')
            with patch.object(preview, 'robot_episode', side_effect=robot), \
                    patch.object(preview, 'human_episode', side_effect=human), \
                    patch('convert_humangen.probe_video', return_value=metadata), \
                    patch.object(preview, 'materialize', side_effect=AssertionError('Dry run wrote an episode')):
                filtered = preview.run(['agibot'], 0, dry_run=True, task_paths=['clip1'])
            self.assertEqual(filtered['sources']['agibot']['eligible_pairs'], 1)
            self.assertEqual(filtered['pairs'][0]['manifest_position'], 1)
            self.assertFalse((root / 'preview').exists())

    def test_failed_pair_is_excluded_from_all_manifests(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root / 'source', root / 'output'
            samples = [{'run': 'run', 'sample': str(i), 'sample_id': i,
                        'status_info': {'video_rel_path': str(i)}, 'human_video_path': str(i),
                        'robot_task_name': 'robot', 'human_task_name': 'human'} for i in range(2)]
            write_json(source / 'icl_configs/ICL_config_agibot.json', {'samples': samples})
            converter = HumanGenConverter(source, output)
            metadata = {'frames': 24, 'fps': '30', 'width': 32, 'height': 32, 'codec': 'h264'}

            def episode(domain, family, sample):
                identity = f'{domain}{sample["sample_id"]}'
                return identity, output / family / domain / identity, {'images': root / identity}

            def fake_convert(source, target, expected):
                if source.name == 'human0':
                    raise ValueError('broken human video')
                target.touch()
                return {**expected, 'method': 'copy'}

            with patch.object(converter, 'robot_episode', side_effect=lambda *args: episode('robot', *args)), \
                    patch.object(converter, 'human_episode', side_effect=lambda *args: episode('human', *args)), \
                    patch('convert_humangen.probe_video', return_value=metadata), \
                    patch('convert_humangen.convert_video', side_effect=fake_convert):
                report = converter.run(['agibot'], 0, workers=2)
            self.assertEqual(len(report['failures']), 1)
            paths = read_json(output / 'agibot_video_paths.json')
            self.assertEqual(len(paths), 2)
            self.assertEqual({Path(path).name for path in paths}, {'robot', 'human'})
            self.assertEqual(read_json(output / 'source_index.json')['pairs'][0]['sample_id'], 1)
            self.assertEqual(len(read_json(output / 'source_index.json')['episodes']), 2)

    def test_reindexed_robot_uses_manifest_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / 'interna1_data/interna1_franka/basic__franka__pick'
            write_json(repo / 'meta/info.json', {'chunks_size': 1000,
                       'video_path': 'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4',
                       'features': {'images.rgb.head': {'dtype': 'video'}, 'images.rgb.hand': {'dtype': 'video'}}})
            relative = 'basic/franka/pick/videos/chunk-001/images.rgb.head/episode_001234_3_70.mp4'
            sample = {'run': 'run2', 'sample': 'sample1', 'status_info': {'video_rel_path': relative}}
            metadata = repo / 'meta/episodes.jsonl'
            metadata.write_text(json.dumps({'episode_index': 7,
                'source': {'video_rel_path': relative}, 'icl': {'run': 'run2', 'sample': 'sample1'}}) + '\n')
            converter = HumanGenConverter(root, root.parent / 'output')
            _, destination, cameras = converter.robot_episode('interna1', sample)
            self.assertEqual(destination.parent, (root.parent / 'output') / repo.relative_to(root))
            self.assertEqual(len(cameras), 2)
            self.assertTrue(all(p.name == 'episode_000007.mp4' for p in cameras.values()))
            self.assertTrue(all('chunk-000' in str(p) for p in cameras.values()))



if __name__ == '__main__':
    unittest.main()
