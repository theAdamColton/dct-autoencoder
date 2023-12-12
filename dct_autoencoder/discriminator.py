import torch
from torch import nn
from transformers import CLIPVisionConfig
from transformers.models.clip.modeling_clip import CLIPEncoder

from .dct_patches import DCTPatches

class Discriminator(nn.Module):
    def __init__(self, max_patch_h: int, max_patch_w: int, channels: int, patch_dim: int, condition_dim: int):
        super().__init__()
        self.pos_embed_channel = nn.Parameter(torch.randn(channels, patch_dim))
        self.pos_embed_height = nn.Parameter(torch.randn(max_patch_h, patch_dim))
        self.pos_embed_width = nn.Parameter(torch.randn(max_patch_w, patch_dim))

        feature_dim = 256

        self.to_patch_embedding = nn.Sequential(
            nn.Linear(patch_dim + condition_dim, feature_dim, bias=False),
            nn.LayerNorm(feature_dim, eps=1e-4),
        )

        self.encoder_config = CLIPVisionConfig(
            hidden_size = feature_dim,
            intermediate_size=1024,
            num_hidden_layers=4,
            num_attention_heads=8,
        )

        self.encoder = CLIPEncoder(
            self.encoder_config
        )

        self.proj_out = nn.Sequential(
                nn.LayerNorm(feature_dim, eps=1e-4),
                nn.Linear(feature_dim, 1, bias=False))

    def add_pos_embedding_(self, dct_patches: DCTPatches):
        """
        in place
        """
        c_pos = self.pos_embed_channel[dct_patches.patch_channels]
        h_pos = self.pos_embed_height[dct_patches.h_indices]
        w_pos = self.pos_embed_width[dct_patches.w_indices]
        patches = dct_patches.patches + h_pos + w_pos + c_pos
        return patches

    def forward(self, x:DCTPatches, z: torch.Tensor):
        """
        x should be a batch of normalized DCTPatches
        z is the embedded tensor for conditioning
        """
        x_proj = self.add_pos_embedding_(x)
        z = torch.concat([x_proj,z], dim=-1)
        z = self.to_patch_embedding(z)
        preds = self.encoder(z, attention_mask = x.attn_mask).last_hidden_state
        return self.proj_out(preds)
