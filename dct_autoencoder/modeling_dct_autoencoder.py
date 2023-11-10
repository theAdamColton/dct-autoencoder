import torch
import torch.nn as nn

from x_transformers import Encoder
from vector_quantize_pytorch import LFQ
from transformers import PreTrainedModel

from dct_autoencoder.patchnorm import PatchNorm
from .configuration_dct_autoencoder import DCTAutoencoderConfig
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

        self.patchnorm = PatchNorm(
            max_patch_h=config.max_patch_h,
            max_patch_w=config.max_patch_w,
            patch_size=config.patch_size,
            channels=config.image_channels,
        )

        patch_dim = config.patch_size**2

        self.encoder_pos_embed_channel = nn.Parameter(
            torch.randn(self.config.image_channels, self.config.feature_dim)
        )
        self.encoder_pos_embed_height = nn.Parameter(
            torch.randn(config.max_patch_h, self.config.feature_dim)
        )
        self.encoder_pos_embed_width = nn.Parameter(
            torch.randn(config.max_patch_w, self.config.feature_dim)
        )

        self.decoder_pos_embed_channel = nn.Parameter(
            torch.randn(self.config.image_channels, config.feature_dim)
        )
        self.decoder_pos_embed_height = nn.Parameter(
            torch.randn(config.max_patch_h, config.feature_dim)
        )
        self.decoder_pos_embed_width = nn.Parameter(
            torch.randn(config.max_patch_w, config.feature_dim)
        )

        self.to_patch_embedding = nn.Sequential(
            nn.Linear(patch_dim, config.feature_dim, bias=False),
            nn.LayerNorm(config.feature_dim),
        )

        self.encoder = Encoder(
            dim=config.feature_dim,
            depth=config.n_layers,
            heads=config.n_attention_heads,
            attn_flash=True,
            ff_glu=True,
            ff_no_bias=True,
            sandwich_norm=True,
            num_memory_tokens = 4,
        )

        self.vq_model = LFQ(
            dim=config.feature_dim,
            num_codebooks=config.vq_num_codebooks,
            codebook_size=config.vq_codebook_size,
        )

        self.decoder = Encoder(
            dim=config.feature_dim,
            depth=config.n_layers,
            heads=config.n_attention_heads,
            attn_flash=True,
            ff_glu=True,
            ff_no_bias=True,
            sandwich_norm=True,
            num_memory_tokens = 4,
        )

        self.proj_out = nn.Sequential(
            nn.LayerNorm(config.feature_dim),
            nn.Linear(config.feature_dim, patch_dim, bias=False),
        )

    def _add_pos_embedding_decoder(self, dct_patches: DCTPatches):
        """
        in place
        """
        c_pos = self.decoder_pos_embed_channel[dct_patches.patch_channels]
        h_pos = self.decoder_pos_embed_height[dct_patches.h_indices]
        w_pos = self.decoder_pos_embed_width[dct_patches.w_indices]

        dct_patches.patches = dct_patches.patches + h_pos + w_pos + c_pos
        return dct_patches

    def _add_pos_embedding_encoder(self, dct_patches: DCTPatches):
        """
        in place
        """
        c_pos = self.encoder_pos_embed_channel[dct_patches.patch_channels]
        h_pos = self.encoder_pos_embed_height[dct_patches.h_indices]
        w_pos = self.encoder_pos_embed_width[dct_patches.w_indices]

        dct_patches.patches = dct_patches.patches + h_pos + w_pos + c_pos
        return dct_patches

    def _normalize(
        self,
        x: DCTPatches,
    ):
        with torch.no_grad():
            x.patches = self.patchnorm(x)
        return x

    def _inv_normalize(
        self,
        x: DCTPatches,
    ):
        x.patches = self.patchnorm.inverse_norm(x)
        return x

    def encode(
        self,
        dct_patches: DCTPatches,
        do_normalize: bool = False,
    ):
        # normalizes
        if do_normalize:
            dct_patches = self._normalize(dct_patches)

        dct_patches.patches = self.to_patch_embedding(dct_patches.patches)

        # Adds positional embedding info
        dct_patches = self._add_pos_embedding_encoder(dct_patches)

        # X-Transformers uses ~attn_mask
        dct_patches.patches = self.encoder(
            dct_patches.patches, attn_mask=~dct_patches.attn_mask
        )

        dct_patches.patches, codes, vq_loss = self.vq_model(dct_patches.patches)

        return dct_patches, codes, vq_loss

    def decode_from_codes(
            self, codes: torch.LongTensor, do_inv_norm:bool=False, **dct_patches_kwargs
    ) -> DCTPatches:
        x = self.vq_model.indices_to_codes(codes)
        x = DCTPatches(patches=x, **dct_patches_kwargs)
        x = self.decode(x, do_inv_norm=do_inv_norm)
        return x

    def decode(
        self,
        x: DCTPatches,
        do_inv_norm:bool=False,
    ) -> DCTPatches:
        """
        in-place
        """
        x = self._add_pos_embedding_decoder(x)
        x.patches = self.proj_out(self.decoder(x.patches, attn_mask=~x.attn_mask))
        if do_inv_norm:
            x = self._inv_normalize(x)
        return x

    def forward(
        self,
        # expects normalized dct patches
        dct_patches: DCTPatches,
        do_normalize:bool=False,
    ):
        dct_patches, codes, vq_loss = self.encode(dct_patches, do_normalize=do_normalize)

        dct_patches = self.decode(dct_patches)

        return dict(
            dct_patches=dct_patches,
            commit_loss=vq_loss,
            codes=codes,
        )
