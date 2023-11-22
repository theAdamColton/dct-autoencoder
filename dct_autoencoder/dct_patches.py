from typing import Any, Dict, Optional, Tuple, List
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

    _data: Optional[Dict[str,List[Any]]] = None

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
            _data = self._data
        )

    def to(self, what):
        self.patches = self.patches.to(what)
        self.key_pad_mask = self.key_pad_mask.to(what)
        self.attn_mask = self.attn_mask.to(what)
        self.batched_image_ids = self.batched_image_ids.to(what)
        self.patch_channels = self.patch_channels.to(what)
        self.patch_positions = self.patch_positions.to(what)
        return self

def to_dict(dct_patches: DCTPatches, codes: torch.LongTensor):
    b,s,h = codes.shape
    assert b == dct_patches.patches.shape[0]
    assert s == dct_patches.patches.shape[1]

    objs = []

    for batch_i in range(b):
        for image_i in range(dct_patches.batched_image_ids[batch_i].max()+1):
            is_image_i_mask = (dct_patches.batched_image_ids[batch_i] == image_i) & ~ dct_patches.key_pad_mask[batch_i]
            obj = {
                    "size": dct_patches.patch_sizes[len(objs)],
                    "original_size": dct_patches.original_sizes[len(objs)],
                    "codes": [
                        {
                            "c": channel.item(),
                            "h": height.item(),
                            "w": width.item(),
                            "data": patch_codes.tolist(),
                        }
                        for channel, height, width, patch_codes in zip(
                            dct_patches.patch_channels[batch_i, is_image_i_mask],
                            dct_patches.h_indices[batch_i, is_image_i_mask],
                            dct_patches.w_indices[batch_i, is_image_i_mask],
                            codes[batch_i, is_image_i_mask],
                        )
                    ]
                }
            objs.append(obj)
    return objs


def from_dict(obj: dict):
    patch_size = obj['size']
    original_size = obj['original_size']
    h_indices = []
    w_indices = []
    channels = []
    codes = []
    for d in obj['codes']:
        h_indices.append(d['h'])
        w_indices.append(d['w'])
        channels.append(d['c'])
        codes.append(d['data'])

    key_pad_mask = torch.zeros(1, len(h_indices), dtype = torch.bool)
    attn_mask = torch.ones(1, len(h_indices), len(h_indices), dtype=torch.bool)
    batched_image_ids = torch.zeros(1,len(h_indices), dtype = torch.long)
    patch_channels = torch.LongTensor(channels).unsqueeze(0)
    patch_positions = torch.stack((torch.LongTensor(h_indices),
                                  torch.LongTensor(w_indices),),
                                  dim=-1).unsqueeze(0)
    patch_sizes = [patch_size]
    original_sizes = [original_size]

    dct_patches = DCTPatches(
            patches=torch.zeros(1),
            key_pad_mask=key_pad_mask,
            attn_mask = attn_mask,
            batched_image_ids=batched_image_ids,
            patch_channels=patch_channels,
            patch_positions=patch_positions,
            patch_sizes=patch_sizes,
            original_sizes=original_sizes
            )

    codes = torch.LongTensor(codes)

    return dct_patches, codes

