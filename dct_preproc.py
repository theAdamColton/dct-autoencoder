from typing import List
import torch
import torchvision.transforms.v2 as transforms

from util import (
    dct2,
)


class DCTPreprocessor:
    def __init__(self, image_channels:int, patch_size:int, dct_compression_factor: float, max_n_patches: int, ):
        self.image_channels = image_channels
        self.patch_size = patch_size
        self.dct_compression_factor = dct_compression_factor
        self.max_n_patches = max_n_patches
        self.max_res = patch_size * max_n_patches


    @torch.no_grad()
    def __call__(self, batch: List[torch.Tensor]):
        """
        batch: list of tensors containing unnormalized image pixels

        returns a list of dct features tensors which are cropped to
            the correct dimensions
        """
        dct_features = []
        original_sizes = []

        for x in batch:
            # TODO take features that are closer to 0,0
            # rather than a rectangle

            c, h, w = x.shape

            assert c == self.image_channels

            if h < self.patch_size:
                rz = transforms.Resize((self.patch_size, int((self.patch_size / h) * w)))
                x = rz(x)
            if w < self.patch_size:
                rz = transforms.Resize((int((self.patch_size / w) * h), self.patch_size))
                x = rz(x)
            c, h, w = x.shape

            assert h >= self.patch_size
            assert w >= self.patch_size

            original_sizes.append((h,w))

            x_dct = dct2(x, "ortho")
        
            h_c, w_c = (1-self.dct_compression_factor) * h, (1-self.dct_compression_factor) * w

            p_h = round(h_c / self.patch_size)
            p_w = round(w_c / self.patch_size)
            p_h = max(p_h, 1)
            p_w = max(p_w, 1)

            # we need that
            # ar = p_h_c / p_w_c = p_h / p_w
            # p_h_c * p_w_c <= self.max_n_patches
            # ar * p_w_c <= self.max_n_patches / p_w_c
            # ar * p_w_c ** 2 <= self.max_n_patches
            # p_w_c = sqrt(self.max_n_patches / ar)
            # p_h_c = ar * p_w_c

            ar = p_h/p_w
            p_w_c = int((self.max_n_patches / ar)**0.5)
            p_h_c = int(ar * p_w_c)

            assert p_h_c * p_w_c <= self.max_n_patches

            p_h = min(p_h, p_h_c)
            p_w = min(p_w, p_w_c)
            p_h = max(p_h, 1)
            p_w = max(p_w, 1)

            dct_h = p_h * self.patch_size
            dct_w = p_w * self.patch_size

            assert dct_h % self.patch_size == 0
            assert dct_w % self.patch_size == 0
            assert dct_h <= self.max_res
            assert dct_w <= self.max_res
            assert dct_h >= self.patch_size
            assert dct_w >= self.patch_size


            x_dct = x_dct[..., :dct_h, :dct_w]

            dct_features.append(x_dct)


        return dct_features, original_sizes

