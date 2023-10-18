from typing import Tuple, List
import torch
from dataclasses import dataclass

@dataclass
class DCTPatches:
    patches: torch.Tensor
    key_pad_mask: torch.BoolTensor
    attn_mask: torch.BoolTensor
    batched_image_ids: torch.LongTensor
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

    def to(self, what):
        self.patches = self.patches.to(what)
        self.key_pad_mask = self.key_pad_mask.to(what)
        self.attn_mask = self.attn_mask.to(what)
        self.batched_image_ids = self.batched_image_ids.to(what)
        self.patch_positions = self.patch_positions.to(what)
        return self

def concat_dctpatches(patches: List[DCTPatches]) -> DCTPatches:
    new_patches = torch.concat([p.patches for p in patches])
    key_pad_mask = torch.concat([p.key_pad_mask for p in patches])
    attn_mask = torch.concat([p.attn_mask for p in patches])
    batched_image_ids = torch.concat([p.batched_image_ids for p in patches])
    patch_positions = torch.concat([p.patch_positions for p in patches])

    patch_sizes = []
    for p in patches:
        patch_sizes = patch_sizes + p.patch_sizes

    original_sizes = []
    for p in patches:
        original_sizes = original_sizes + p.original_sizes

    return DCTPatches(
            patches=new_patches,
            key_pad_mask=key_pad_mask,
            attn_mask=attn_mask,
            batched_image_ids=batched_image_ids,
            patch_positions=patch_positions,
            patch_sizes=patch_sizes,
            original_sizes=original_sizes,
            )

def slice_dctpatches(p: DCTPatches, i:int) -> Tuple[DCTPatches, DCTPatches]:
    if i >= p.patches.shape[0]:
        return (p,None)

    n_images_per_batch_element = p.batched_image_ids.max(dim=-1).values + 1
    n_images_p1 = n_images_per_batch_element[:i].sum().item()
    return (
        DCTPatches(
                p.patches[:i],
                p.key_pad_mask[:i],
                p.attn_mask[:i],
                p.batched_image_ids[:i],
                p.patch_positions[:i],
                p.patch_sizes[:n_images_p1],
                p.original_sizes[:n_images_p1],
        ),
        DCTPatches(
            p.patches[i:],
                p.key_pad_mask[i:],
                p.attn_mask[i:],
                p.batched_image_ids[i:],
                p.patch_positions[i:],
                p.patch_sizes[n_images_p1:],
                p.original_sizes[n_images_p1:],
    ),)

