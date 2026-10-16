"""Regression checks for HumanGen source identity and exported episode renumbering."""

import copy
from pathlib import Path
import unittest

from prepare import resolve_record


class PairIdentityTests(unittest.TestCase):
    def setUp(self):
        self.root = Path('/dataset/HumanGen')
        self.task = self.root / 'agibot_data/task_360'
        self.source_rel = 'task_360/videos/chunk-000/observation.images.head/episode_000230_669_851.mp4'
        self.sample = {
            'run': 'run_a', 'sample': 'sample_a',
            'human_video_path': 'icl_raw_data/human_data/agibot/run_a/sample_a/generated.mp4',
            'status_info': {'video_rel_path': self.source_rel, 'interval': [669, 851]},
        }
        self.info = {
            'chunks_size': 1000,
            'video_path': 'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4',
        }
        self.row = {
            'episode_index': 0, 'length': 182,
            'source': {'video_rel_path': self.source_rel, 'frame_range': [669, 851]},
            'icl': {key: self.sample[key] for key in ('run', 'sample', 'human_video_path')},
        }

    def resolve(self, rows, sample=None, sources=None):
        entries = [(self.task, self.info, row) for row in rows]
        return resolve_record(self.root, 'agibot', 0, sample or self.sample,
                              {0: entries} if entries else {}, sources or {}, {})

    def test_renumbered_export_uses_new_episode_and_preserves_crop(self):
        record = self.resolve([self.row])
        self.assertTrue(record['robot_source'].endswith('observation.images.head/episode_000000.mp4'))
        self.assertEqual(record['robot_source_frame_range'], [669, 851])
        self.assertEqual(record['robot_expected_frames'], 182)
        self.assertEqual(record['human_source'], '/dataset/HumanGen/human_data/agibot/run_a/sample_a/generated.mp4')

    def test_duplicate_identity_is_disambiguated_by_exact_frame_range(self):
        inclusive_variant = copy.deepcopy(self.row)
        inclusive_variant.update(episode_index=1, length=183)
        inclusive_variant['source']['frame_range'] = [669, 852]
        record = self.resolve([inclusive_variant, self.row])
        self.assertEqual(record['robot_episode_index'], 0)

    def test_wrong_human_identity_is_rejected(self):
        wrong = copy.deepcopy(self.row)
        wrong['icl']['human_video_path'] = 'another_human.mp4'
        with self.assertRaises(ValueError):
            self.resolve([wrong])

    def test_ambiguous_exports_are_rejected(self):
        duplicate = copy.deepcopy(self.row)
        duplicate['episode_index'] = 2
        with self.assertRaises(ValueError):
            self.resolve([self.row, duplicate])

    def test_missing_icl_allows_only_unique_exact_source_match(self):
        row = copy.deepcopy(self.row)
        del row['icl']
        record = self.resolve([], sources={self.source_rel: [(self.task, self.info, row)]})
        self.assertEqual(record['robot_episode_index'], 0)
        with self.assertRaises(ValueError):
            self.resolve([], sources={'unrelated_video.mp4': [(self.task, self.info, row)]})

    def test_robotwin_retains_original_episode_number(self):
        sample = copy.deepcopy(self.sample)
        sample['status_info']['video_rel_path'] = 'task_a/videos/chunk-000/observation.images.cam_high/episode_000973.mp4'
        record = resolve_record(self.root, 'robotwin', 0, sample, {}, {}, {
            'task_a': (self.root / 'robotwin_data/task_a', self.info,
                       {973: {'episode_index': 973, 'length': 139}}),
        })
        self.assertTrue(record['robot_source'].endswith('observation.images.cam_high/episode_000973.mp4'))
        self.assertEqual(record['robot_expected_frames'], 139)


if __name__ == '__main__':
    unittest.main()
