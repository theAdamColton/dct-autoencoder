import random
import torch
import unittest

from ..dct_processor import DCTProcessor, DCTPatches

class TestPatching(unittest.TestCase):
    def test_preprocess(self):
        channels = 1
        patch_size = 4
        compression_factor = 1.0
        max_n_patches = 32
        max_seq_len = max_n_patches * 2
        batch_size = 100
        max_batch_size = batch_size

        proc = DCTProcessor(channels, patch_size, compression_factor, max_n_patches, max_seq_len, max_batch_size)

        random.seed(42)
        x = []
        for _ in range(batch_size):
            h = random.randint(patch_size, patch_size * max_n_patches)
            w = random.randint(patch_size, patch_size * max_n_patches)
            x.append(torch.randn(channels, h, w))

        patched, positions, original_sizes = proc.preprocess(x)

        batched_x = proc.batch(patched, positions, original_sizes)

        rec_x = proc.postprocess(batched_x)

        for og, rec in zip(x, rec_x):
            self.assertTrue(torch.allclose(og, rec))
