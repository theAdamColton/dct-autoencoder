from typing import List
import torch
from torch import nn
from torchvision.ops import MLP

from .dct_patches import DCTPatches

class Discriminator(nn.Module):
    def __init__(self, max_patch_h: int, max_patch_w: int, channels: int, in_dim: int, hidden_channels: List[int]):
        super().__init__()
        assert hidden_channels[-1] == 2

        self.pos_embed_channel = nn.Parameter(torch.randn(channels, in_dim))
        self.pos_embed_height = nn.Parameter(torch.randn(max_patch_h, in_dim))
        self.pos_embed_width = nn.Parameter(torch.randn(max_patch_w, in_dim))

        self.model = MLP(
                in_dim,
                hidden_channels,
                norm_layer=nn.LayerNorm,
                activation_layer=nn.GELU,
                bias=False,
        )

    def add_pos_embedding_(self, dct_patches: DCTPatches):
        """
        in place
        """
        c_pos = self.pos_embed_channel[dct_patches.patch_channels]
        h_pos = self.pos_embed_height[dct_patches.h_indices]
        w_pos = self.pos_embed_width[dct_patches.w_indices]
        dct_patches.patches = dct_patches.patches + h_pos + w_pos + c_pos
        return dct_patches

    def forward(self, x):
        return self.model(x)
