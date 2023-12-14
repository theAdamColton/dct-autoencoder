import torch
import torch.nn as nn

from transformers import PreTrainedModel
from transformers.models.clip.modeling_clip import CLIPEncoder

from dct_autoencoder.patchnorm import PatchNorm
from .configuration_dct_autoencoder import DCTAutoencoderConfig
from .dct_patches import DCTPatches
from .lfq import LFQ
from .util import compute_entropy_loss


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

        feature_dim = config.encoder_config.hidden_size

        self.patchnorm = PatchNorm(
            max_patch_h=config.max_patch_h,
            max_patch_w=config.max_patch_w,
            patch_size=config.patch_size,
            channels=config.image_channels,
        )

        patch_dim = config.patch_size**2

        self.encoder_pos_embed_channel = nn.Parameter(
            torch.randn(self.config.image_channels, feature_dim)
        )
        self.encoder_pos_embed_height = nn.Parameter(
            torch.randn(config.max_patch_h, feature_dim)
        )
        self.encoder_pos_embed_width = nn.Parameter(
            torch.randn(config.max_patch_w, feature_dim)
        )

        self.decoder_pos_embed_channel = nn.Parameter(
            torch.randn(self.config.image_channels, feature_dim)
        )
        self.decoder_pos_embed_height = nn.Parameter(
            torch.randn(config.max_patch_h, feature_dim)
        )
        self.decoder_pos_embed_width = nn.Parameter(
            torch.randn(config.max_patch_w, feature_dim)
        )

        self.to_patch_embedding = nn.Sequential(
            nn.Linear(patch_dim, feature_dim, bias=False),
            nn.LayerNorm(feature_dim, eps=1e-4),
        )

        self.encoder = CLIPEncoder(
            self.config.encoder_config
        )

        self.vq_model = LFQ(
            dim=feature_dim,
            num_codebooks=config.vq_num_codebooks,
            codebook_size=config.vq_codebook_size,
        )

        self.decoder = CLIPEncoder(
                self.config.decoder_config
        )

        self.proj_out = nn.Sequential(
            nn.LayerNorm(feature_dim, eps=1e-4),
            nn.Linear(feature_dim, patch_dim, bias=False),
        )

    def get_pos_embedding_decoder(self, dct_patches: DCTPatches):
        c_pos = self.decoder_pos_embed_channel[dct_patches.patch_channels]
        h_pos = self.decoder_pos_embed_height[dct_patches.h_indices]
        w_pos = self.decoder_pos_embed_width[dct_patches.w_indices]
        return h_pos + w_pos + c_pos

    def add_pos_embedding_decoder_(self, dct_patches: DCTPatches):
        """
        in place
        """
        dct_patches.patches = dct_patches.patches + self.get_pos_embedding_decoder(dct_patches)
        return dct_patches

    def add_pos_embedding_encoder_(self, dct_patches: DCTPatches):
        """
        in place
        """
        c_pos = self.encoder_pos_embed_channel[dct_patches.patch_channels]
        h_pos = self.encoder_pos_embed_height[dct_patches.h_indices]
        w_pos = self.encoder_pos_embed_width[dct_patches.w_indices]

        dct_patches.patches = dct_patches.patches + h_pos + w_pos + c_pos
        return dct_patches

    @torch.no_grad()
    def normalize_(
        self,
        x: DCTPatches,
    ):
        x.patches = self.patchnorm(x)
        return x

    def inv_normalize_(
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
            dct_patches = self.normalize_(dct_patches)

        dct_patches.patches = self.to_patch_embedding(dct_patches.patches)

        # Adds positional embedding info
        dct_patches = self.add_pos_embedding_encoder_(dct_patches)

        # X-Transformers uses ~attn_mask
        # TODO Should be ~ attn mask?
        dct_patches.patches = self.encoder(
            dct_patches.patches, attention_mask=dct_patches.attn_mask
        ).last_hidden_state

        dct_patches.patches, codes, commit_loss, distances = self.vq_model(dct_patches.patches, mask=~dct_patches.key_pad_mask)

        return dct_patches, codes, commit_loss, distances

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
        x = self.add_pos_embedding_decoder_(x)
        # TODO Should be ~ attn mask?
        x.patches = self.proj_out(self.decoder(x.patches, attention_mask=x.attn_mask).last_hidden_state)
        if do_inv_norm:
            x = self.inv_normalize_(x)
        return x

    def forward(
        self,
        # expects normalized dct patches
        dct_patches: DCTPatches,
        do_normalize:bool=False,
    ):
        dct_patches, codes, commit_loss, distances = self.encode(dct_patches, do_normalize=do_normalize)
        dct_patches = self.decode(dct_patches)

        return dict(
            dct_patches=dct_patches,
            commit_loss=commit_loss,
            codes=codes,
            distances=distances,
        )

    def entropy_loss(self, distances: torch.Tensor, mask:torch.BoolTensor):
        return compute_entropy_loss(distances, mask)

