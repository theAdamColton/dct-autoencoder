from einops import rearrange
import torch.nn.functional as F
import torch
import torch.nn as nn

from x_transformers import Encoder
from vector_quantize_pytorch import LFQ
from transformers import PreTrainedModel

from dct_autoencoder.patchnorm import PatchNorm
from .configuration_dct_autoencoder import DCTAutoencoderConfig
from .util import (
    calculate_perplexity,
)
from .dct_patches import DCTPatches


class DCTAutoencoder(PreTrainedModel):
    config_class = DCTAutoencoderConfig
    base_model_prefix = "dct_autoencoder"

    def __init__(
        self,
        config: DCTAutoencoderConfig,
    ):
        """
        input_res: the square integer input resolution size.
        """
        super().__init__(config)

        self.config = config

        patch_dim = config.image_channels * (config.patch_size**2)

        self.pos_embed_height = nn.Parameter(torch.zeros(config.max_n_patches, patch_dim))
        self.pos_embed_width = nn.Parameter(torch.zeros(config.max_n_patches, patch_dim))

        self.patchnorm = PatchNorm(max_n_patches_h=config.max_n_patches, max_n_patches_w=config.max_n_patches, patch_res = config.patch_size, channels=config.image_channels)

        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, config.feature_dim, bias=False),
            nn.LayerNorm(config.feature_dim),
        )

        self.encoder = Encoder(
                dim=config.feature_dim,
                depth=config.n_layers,
                heads=config.n_attention_heads,
                attn_flash = True,
                ff_glu = True,
                ff_no_bias = True,
                sandwich_norm = True,
            )

        self.vq_model = LFQ(
                dim = config.feature_dim,
                num_codebooks=config.vq_num_codebooks,
                codebook_size=config.vq_codebook_size,
                )

        self.decoder = Encoder(
            dim=config.feature_dim,
            depth=config.n_layers,
            heads=config.n_attention_heads,
            attn_flash = True,
            ff_glu = True,
            ff_no_bias = True,
            sandwich_norm = True,
        )

        self.proj_out = nn.Sequential(
            nn.LayerNorm(config.feature_dim),
            nn.Linear(config.feature_dim, patch_dim, bias=False),
        )
        

    def _add_pos_embedding(self,
                           dct_patches: DCTPatches):
        """
        in place
        """
        h_pos = self.pos_embed_height[dct_patches.h_indices]
        w_pos = self.pos_embed_width[dct_patches.w_indices]

        dct_patches.patches = dct_patches.patches + h_pos + w_pos
        return dct_patches

    def _normalize(self,
                   x: DCTPatches,
                   ):
        with torch.no_grad():
            x.patches = self.patchnorm(x.patches, x.h_indices, x.w_indices, x.key_pad_mask, )
        return x

    def encode(
            self,
            dct_patches: DCTPatches,
            do_normalize: bool = False,
            ):
        # normalizes
        if do_normalize:
            dct_patches = self._normalize(dct_patches)

        dct_patches = self._add_pos_embedding(dct_patches)

        # Adds positional embedding info
        dct_patches.patches = self.to_patch_embedding(dct_patches.patches)

        # this mask is True where actual input patches are,
        # and is false where padding was added
        mask = ~dct_patches.key_pad_mask

        # X-Transformers uses ~attn_mask
        dct_patches.patches = self.encoder(dct_patches.patches, attn_mask=~dct_patches.attn_mask)

        dct_patches.patches, codes, vq_loss = self.vq_model(dct_patches.patches)

        return dct_patches, codes, vq_loss

    def decode(
            self,
            x: DCTPatches,
            ) -> torch.Tensor:
        """
        TODO
        Can provide either dct_patches embeddings x, or vq codes
        """
        return self.proj_out(self.decoder(x.patches, attn_mask=~x.attn_mask))


    def forward(
        self,
        # expects already normalized dct patches
        dct_patches: DCTPatches,
        return_loss: bool = True,
    ):
        dct_patches = self._normalize(dct_patches)

        if return_loss:
            # stores the input features for the loss calculation later
            input_patches = dct_patches.patches

        dct_patches, codes, vq_loss = self.encode(dct_patches, do_normalize=False)

        mask = ~dct_patches.key_pad_mask

        if return_loss:
            with torch.no_grad():
                perplexity = calculate_perplexity(codes[mask], self.config.vq_codebook_size)
        else:
            perplexity = 0.0

        dct_patches.patches = self.decode(dct_patches)

        if return_loss:
            rec_loss = F.mse_loss(dct_patches.patches[mask], input_patches[mask])
        else:
            rec_loss = 0.0

        if return_loss and self.config.channel_diversity_loss_coeff > 0.0:
            def pull_out_channels(x):
                c, p1, p2 = self.config.image_channels, self.config.patch_size, self.config.patch_size
                return rearrange(x, '... (c p1 p2) -> ... c (p1 p2)', c=c,p1=p1,p2=p2)

            # normalize the reconstructed dct patches by subtracting the mean
            # In essence, this removes the grayscale information from the image.
            # For channel_diversity_loss, we don't care about getting the right grayscale 
            # cross-channel intensities.
            # We instead care about getting the right variation across
            # channels, per pixel.
            patch_hats = pull_out_channels(dct_patches.patches)[mask]
            patch = pull_out_channels(input_patches)[mask]

            channel_diversity_loss = F.mse_loss(
                    patch - patch.mean(dim=-2, keepdim=True),
                    patch_hats - patch_hats.mean(dim=-2, keepdim=True),
            ) * self.config.channel_diversity_loss_coeff
        else:
            channel_diversity_loss = 0.0

        return dict(
            dct_patches=dct_patches,
            perplexity=perplexity,
            commit_loss=vq_loss,
            rec_loss=rec_loss,
            codes=codes,
            channel_diversity_loss=channel_diversity_loss,
        )
    
