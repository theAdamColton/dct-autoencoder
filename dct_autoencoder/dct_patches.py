from typing import Tuple, List
import torch
from dataclasses import dataclass


@dataclass
class DCTPatches:
    patches: torch.Tensor
    key_pad_mask: torch.BoolTensor
    attn_mask: torch.BoolTensor
    batched_image_ids: torch.LongTensor
    # b,s
    patch_channels: torch.LongTensor
    # b,s,2
    patch_positions: torch.LongTensor
    # ph, pw of the patches
    patch_sizes: List[Tuple]
    # h,w of the original image pixels
    original_sizes: List[Tuple]

    @property
    def h_indices(self):
        return self.patch_positions[..., 0]

    @property
    def w_indices(self):
        return self.patch_positions[..., 1]

    def shallow_copy(self):
        return DCTPatches(
            patches=self.patches,
            key_pad_mask=self.key_pad_mask,
            attn_mask=self.attn_mask,
            batched_image_ids=self.batched_image_ids,
            patch_channels=self.patch_channels,
            patch_positions=self.patch_positions,
            patch_sizes=self.patch_sizes,
            original_sizes=self.original_sizes,
        )

    def to(self, what):
        self.patches = self.patches.to(what)
        self.key_pad_mask = self.key_pad_mask.to(what)
        self.attn_mask = self.attn_mask.to(what)
        self.batched_image_ids = self.batched_image_ids.to(what)
        self.patch_channels = self.patch_channels.to(what)
        self.patch_positions = self.patch_positions.to(what)
        return self


def slice_dctpatches(p: DCTPatches, i: int) -> Tuple[DCTPatches, DCTPatches]:
    """
    slices along batch
    """
    if i >= p.patches.shape[0]:
        return (p, None)

    n_images_per_batch_element = p.batched_image_ids.max(dim=-1).values + 1
    n_images_p1 = n_images_per_batch_element[:i].sum().item()
    return (
        DCTPatches(
            p.patches[:i],
            p.key_pad_mask[:i],
            p.attn_mask[:i],
            p.batched_image_ids[:i],
            p.patch_channels[:i],
            p.patch_positions[:i],
            p.patch_sizes[:n_images_p1],
            p.original_sizes[:n_images_p1],
        ),
        DCTPatches(
            p.patches[i:],
            p.key_pad_mask[i:],
            p.attn_mask[i:],
            p.batched_image_ids[i:],
            p.patch_channels[i:],
            p.patch_positions[i:],
            p.patch_sizes[n_images_p1:],
            p.original_sizes[n_images_p1:],
        ),
    )
