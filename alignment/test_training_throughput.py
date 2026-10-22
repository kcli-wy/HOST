"""CPU regression checks for the alignment training fast path.

Run from alignment/: USE_TF=0 python -m unittest test_training_throughput -v
"""

import os
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, default_collate
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

import datasets
from algos.algorithm import Algorithm
from config import CONFIG
from models import BaseModel
from monkey_patch_forward import replace_qwen3_with_mixed_modality_forward


class WorkerDataset(Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, index):
        return {'index': index, 'pid': os.getpid()}


class FixedFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Parameter(torch.randn(4, 3))

    def forward(self, data):
        return self.features


class StepEmbedding(nn.Module):
    def forward(self, features, num_steps):
        return features


class TrainingThroughputTest(unittest.TestCase):
    def test_context_and_alignment_pad_primary_view_on_both_sides(self):
        def frame(role, view, index):
            return f'video:///{role}/{view}.mp4::{int(index)}::224x224'

        camera_orders = {
            1: ['front', 'front', 'front'],
            2: ['front', 'side', 'front'],
            3: ['front', 'side', 'wrist'],
        }
        for mode in ('train', 'eval'):
            for main_count in (1, 2, 3):
                for ref_count in (1, 2, 3):
                    with self.subTest(mode=mode, main=main_count, ref=ref_count):
                        main_views = ['front', 'side', 'wrist'][:main_count]
                        ref_views = ['front', 'side', 'wrist'][:ref_count]
                        loaded = dict(
                            video_path='/episodes/main', ref_video_path='/episodes/ref',
                            dataset_name='fixture', view='front', ref_view='front',
                            cam_list=main_views, ref_available_views=ref_views,
                            main_files=[frame('main', 'front', i) for i in range(120)],
                            ref_files=[frame('ref', 'front', i) for i in range(120)],
                            instruction='main task', ref_instruction='ref task',
                            initial_frame_path=frame('main', 'front', 0),
                            ref_initial_frame_path=frame('ref', 'front', 0))
                        dataset = datasets.AlignmentDataset.__new__(datasets.AlignmentDataset)
                        dataset.mode = mode
                        dataset.views = ['images']
                        with mock.patch.object(dataset, '_load_video_data_from_json', return_value=loaded), \
                                mock.patch.dict(CONFIG.DATA, NUM_STEPS=3), \
                                mock.patch.dict(CONFIG.TRAIN, NUM_FRAMES=24, NUM_ALIGN_FRAMES=24, MAX_BATCH_FRAMES=96), \
                                mock.patch.dict(CONFIG.JOINTS, USE_JOINTS=False):
                            sample = dataset._get_item_impl(0)
                        for role, count, step_key, path_key, align_start in (
                                ('main', main_count, 'chosen_steps', 'frame_paths', 0),
                                ('ref', ref_count, 'ref_chosen_steps', 'ref_frame_paths', 24)):
                            expected_views = camera_orders[count]
                            expected = [frame(role, view, index)
                                        for index in sample[step_key] for view in expected_views]
                            self.assertEqual(sample[path_key], expected)
                            align = sample['align_frame_paths'][align_start:align_start + 24]
                            self.assertEqual(len(align), 24)
                            for i in range(0, 24, 3):
                                index = int(align[i].split('::')[1])
                                self.assertEqual(align[i:i + 3],
                                                 [frame(role, view, index) for view in expected_views])
                        self.assertEqual(sample['num_align_frames'], 48)
                        self.assertEqual(loaded['cam_list'], ['front', 'side', 'wrist'][:main_count])
                        self.assertEqual(loaded['ref_available_views'], ['front', 'side', 'wrist'][:ref_count])

    def test_missing_native_view_is_rejected_before_padding(self):
        dataset = datasets.AlignmentDataset.__new__(datasets.AlignmentDataset)
        mapping = {'raw': {'/episode': ['front', 'side']}, 'index': {'/episode': '/episode'}}
        with mock.patch.object(dataset, '_view_exists_on_disk',
                               side_effect=lambda info, view: view == 'front'), \
                mock.patch.object(dataset, '_get_view_frame_count') as frame_count:
            with self.assertRaises(datasets.IgnoreSample):
                dataset._resolve_cam_list_main('/episode', 'fixture', mapping)
            frame_count.assert_not_called()

    def test_video_reads_preserve_frame_order_and_views(self):
        requests = {}

        def reader(path, num_threads):
            def get_batch(indices):
                requests[path] = indices
                offset = 100 if path == 'b.mp4' else 0
                frames = np.stack([np.full((3, 4, 3), offset + index, dtype=np.uint8)
                                   for index in indices])
                return SimpleNamespace(asnumpy=lambda: frames)
            return SimpleNamespace(get_batch=get_batch)

        paths = ['video://a.mp4::5::2x2', 'video://a.mp4::1::2x2',
                 'video://a.mp4::5::2x2', 'video://b.mp4::3::3x2',
                 'video://a.mp4::0::2x2']
        with mock.patch.object(datasets.decord, 'VideoReader', side_effect=reader):
            images = datasets.AlignmentCollator().load_imgs(paths)
        for image, value, size in zip(images, [5, 1, 5, 103, 0],
                                      [(2, 2), (2, 2), (2, 2), (3, 2), (2, 2)]):
            self.assertEqual(image.size, size)
            self.assertTrue((np.asarray(image) == value).all())
        self.assertEqual(requests, {'a.mp4': [0, 1, 5], 'b.mp4': [3]})

    def test_features_and_gradients_match_full_qwen_forward(self):
        replace_qwen3_with_mixed_modality_forward()
        config = Qwen3VLConfig(
            text_config=dict(vocab_size=64, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=2, num_attention_heads=4,
                             num_key_value_heads=2, head_dim=8,
                             rope_scaling={'rope_type': 'default',
                                           'mrope_section': [2, 1, 1],
                                           'mrope_interleaved': True}),
            vision_config=dict(depth=2, hidden_size=32, intermediate_size=64,
                               num_heads=4, out_hidden_size=32,
                               num_position_embeddings=64,
                               deepstack_visual_indexes=[0, 1]),
            image_token_id=12, video_token_id=13,
            vision_start_token_id=10, vision_end_token_id=11)
        config._attn_implementation = 'sdpa'
        torch.manual_seed(7)
        backbone = Qwen3VLForConditionalGeneration(config).eval()
        model = BaseModel.__new__(BaseModel)
        nn.Module.__init__(model)
        model.base_model = backbone
        inputs = {
            'input_ids': torch.tensor([[1, 10, 12, 11, 3, 10, 12, 11, 4]] * 2),
            'attention_mask': torch.ones(2, 9, dtype=torch.long),
            'pixel_values': torch.randn(16, 1536),
            'image_grid_thw': torch.tensor([[1, 2, 2]] * 4),
            'num_mains': torch.tensor([1, 0]),
            'num_refs': torch.tensor([0, 1]),
            'cls_token_id': torch.tensor(4),
        }
        with mock.patch.dict(CONFIG.SPECIAL_TOKENS,
                             ALIGN_END_TOKEN_ID=3, CLS_TOKEN_ID=4):
            expected = backbone(**inputs, output_hidden_states=True,
                                return_dict=True, use_cache=False).hidden_states[-1][:, -1]
            target = torch.randn_like(expected)
            (expected * target).sum().backward()
            gradients = {name: param.grad.clone()
                         for name, param in backbone.named_parameters()
                         if param.grad is not None}
            backbone.zero_grad(set_to_none=True)
            with mock.patch.object(backbone.lm_head, 'forward',
                                   side_effect=AssertionError('Unused vocabulary projection')):
                actual = model({'qwen_input': inputs})
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            (actual * target).sum().backward()
        self.assertTrue(any(name.startswith('model.visual') for name in gradients))
        for name, param in backbone.named_parameters():
            if name in gradients:
                torch.testing.assert_close(param.grad, gradients[name], rtol=1e-5, atol=1e-6)
            else:
                self.assertIsNone(param.grad)

    def test_training_does_not_cache_graphs_but_eval_caches_references(self):
        cnn = FixedFeatures()
        algo = Algorithm(model={'cnn': cnn, 'emb': StepEmbedding()})
        data = {'qwen_input': {}, 'ref_frame_paths': [('main',), ('ref',)]}
        steps = torch.tensor([[0, 1], [0, 1]])
        lengths = torch.tensor([2, 2])
        with mock.patch.dict(CONFIG.TRAIN, NUM_FRAMES=2):
            result = algo(data, steps, lengths, training=True)
            result.sum().backward()
        torch.testing.assert_close(cnn.features.grad, torch.ones_like(cnn.features))
        self.assertEqual(len(algo._ref_cache._store), 0)
        with mock.patch.dict(CONFIG.EVAL, NUM_FRAMES=2, REF_CACHE_MAIN_ONLY=False):
            with torch.no_grad():
                algo(data, steps, lengths, training=False)
        self.assertEqual(len(algo._ref_cache._store), 1)

    def test_workers_are_reused_across_complete_epochs(self):
        with mock.patch.object(datasets, 'AlignmentDataset', return_value=WorkerDataset()), \
                mock.patch.object(datasets, 'AlignmentCollator', return_value=default_collate), \
                mock.patch.dict(CONFIG.DATA, NUM_WORKERS=1):
            loader = datasets.create_dataset('train', 'train', batch_size=2)
            try:
                epochs = [[next(loader) for _ in range(4)] for _ in range(2)]
                pids = {pid.item() for epoch in epochs for batch in epoch for pid in batch['pid']}
                self.assertEqual(len(pids), 1)
                self.assertNotIn(os.getpid(), pids)
                for epoch in epochs:
                    self.assertEqual(sorted(torch.cat([b['index'] for b in epoch]).tolist()),
                                     list(range(8)))
            finally:
                loader.iterator._shutdown_workers()

    def test_loader_supports_zero_workers(self):
        with mock.patch.object(datasets, 'AlignmentDataset', return_value=WorkerDataset()), \
                mock.patch.object(datasets, 'AlignmentCollator', return_value=default_collate), \
                mock.patch.dict(CONFIG.DATA, NUM_WORKERS=0):
            loader = datasets.create_dataset('train', 'train', batch_size=2)
            self.assertTrue((next(loader)['pid'] == os.getpid()).all())


if __name__ == '__main__':
    unittest.main()
