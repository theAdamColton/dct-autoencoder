import random
import torch
import unittest

from ..dct_processor import DCTProcessor, DCTPatches


class TestPatching(unittest.TestCase):
    def testpreprocess_compressed(self):
        self.__test_preprocess(compression_factor=0.5)

    def test_preprocess_many(self):
        for channels in [1, 3, 4]:
            for patch_size in [2, 8, 16]:
                for compression_factor in [1.0, 0.5, 0.1]:
                    for _ in range(5):
                        self.__test_preprocess(
                            channels,
                            patch_size,
                            compression_factor,
                        )

    def __test_preprocess(
        self,
        channels=1,
        patch_size=2,
        compression_factor=1.0,
        max_n_patches=4,
        batch_size=100,
    ):
        max_seq_len = max_n_patches * 2
        max_batch_size = batch_size

        proc = DCTProcessor(
            channels,
            patch_size,
            compression_factor,
            max_n_patches,
            max_seq_len,
            max_batch_size,
        )
        proc._transform_image_in = lambda x: x
        proc._transform_image_out = lambda x: x

        random.seed(42)
        x = []
        for _ in range(batch_size):
            h = random.randint(patch_size, patch_size * max_n_patches)
            w = random.randint(patch_size, patch_size * max_n_patches)
            x.append(torch.arange(channels * h * w).reshape(channels, h, w) * 1.0)

        patched, positions, original_sizes, patch_sizes = proc.preprocess(x)

        batched_x = proc.batch(patched, positions, original_sizes, patch_sizes)

        rec_x = proc.postprocess(batched_x)

        if compression_factor == 1.0:
            for og, rec in zip(
                x,
                rec_x,
            ):
                _, h, w = og.shape
                ch, cw = proc._get_crop_dims(h, w)
                og = og[:, :ch, :cw]
                rec = rec[:, :ch, :cw]
                if not torch.allclose(og, rec):
                    print("not all close")
                    print("x", og)
                    print("rec_x", rec)
                self.assertTrue(torch.allclose(og, rec))
