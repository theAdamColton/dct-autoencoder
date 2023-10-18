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


