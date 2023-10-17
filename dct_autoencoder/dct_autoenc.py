from einops import rearrange
import torch.nn.functional as F
import torch
import torch.nn as nn
from x_transformers import Encoder

from .dct_processor import DCTPatches, DCTProcessor
from .util import (
    calculate_perplexity,
)


class DCTAutoencoder(nn.Module):
    def __init__(
        self,
        vq_model,
        image_channels: int = 3,
        depth: int = 4,
        feature_channels: int = 1024,
        patch_size: int = 32,
        max_n_patches: int = 512,
        dct_processor: DCTProcessor = None,
        codebook_size: int = None,
        pos_embed_before_proj: bool = True,
        # this is a loss coeff that penalizes making the same prediction across all dct channels
        channel_diversity_loss_coeff:float = 1e-1,
    ):
        """
        input_res: the square integer input resolution size.
        """
        super().__init__()

        self.channel_diversity_loss_coeff = channel_diversity_loss_coeff
        self.image_channels = image_channels
        self.patch_size = patch_size
        self.feature_channels = feature_channels
        self.max_n_patches = max_n_patches

        patch_dim = image_channels * (patch_size**2)

        pos_dim = patch_dim if pos_embed_before_proj else feature_channels
        self.pos_embed_before_proj = pos_embed_before_proj
        self.pos_embed_height = nn.Parameter(torch.zeros(max_n_patches, pos_dim))
        self.pos_embed_width = nn.Parameter(torch.zeros(max_n_patches, pos_dim))


        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, feature_channels, bias=False),
            nn.LayerNorm(feature_channels),
        )

        self.encoder = Encoder(
                dim=feature_channels,
                depth=depth,
                heads=16,
                attn_flash = True,
                ff_glu = True,
                ff_no_bias = True,
                sandwich_norm = True,
            )

        self.vq_model = vq_model

        self.decoder = Encoder(
            dim=feature_channels,
            depth=depth,
            heads=16,
            attn_flash = True,
            ff_glu = True,
            ff_no_bias = True,
            sandwich_norm = True,
                )

        self.proj_out = nn.Sequential(
            nn.LayerNorm(feature_channels),
            nn.Linear(feature_channels, image_channels * patch_size**2, bias=False),
        )
        


        self.codebook_size = codebook_size

        max_res = patch_size * max_n_patches
        self.max_res = max_res

        self.dct_processor = dct_processor

    def forward(
        self,
        # normalized dct patches
        dct_patches: DCTPatches = None,
        decode: bool = False,
    ):
        dct_normalized_patches = dct_patches.patches.clone()

        #assert not dct_patches.patches.isnan().any().item()

        # possibly adds pos embedding info before projecting in
        if self.pos_embed_before_proj:
            h_pos = self.pos_embed_height[dct_patches.h_indices]
            w_pos = self.pos_embed_width[dct_patches.w_indices]

            dct_patches.patches = dct_patches.patches + h_pos + w_pos
            dct_patches.patches = self.to_patch_embedding(dct_patches.patches)
        else:
            dct_patches.patches = self.to_patch_embedding(dct_patches.patches)
            h_pos = self.pos_embed_height[dct_patches.h_indices]
            w_pos = self.pos_embed_width[dct_patches.w_indices]

            dct_patches.patches = dct_patches.patches + h_pos + w_pos

        #print("patch statistics:", dct_patches.patches.max().item(), dct_patches.patches.min().item(), dct_patches.patches.mean().item(), dct_patches.patches.std().item(), dct_patches.patches.dtype)
    

        # X-Transformers uses ~attn_mask
        dct_patches.patches = self.encoder(dct_patches.patches, attn_mask=~dct_patches.attn_mask)

        #assert not dct_patches.patches.isnan().any().item()

        mask = ~dct_patches.key_pad_mask

        # TODO figure out how to make the mask work
        # with the vq model, because it effects the codebook
        # update
        #dct_patches.patches = dct_patches.patches.to(torch.float32)
        dct_patches.patches, codes, commit_loss = self.vq_model(dct_patches.patches)#, mask=mask)
        commit_loss= commit_loss.mean()
        #dct_patches.patches = dct_patches.patches.to(dct_normalized_patches.dtype)

        with torch.no_grad():
            perplexity = calculate_perplexity(codes[mask], self.codebook_size)

        dct_patches.patches = self.decoder(dct_patches.patches)

        dct_patches.patches = self.proj_out(dct_patches.patches)

        rec_loss = F.mse_loss(dct_patches.patches[mask], dct_normalized_patches[mask])

        if self.channel_diversity_loss_coeff > 0.0:
            def pull_out_channels(x):
                return rearrange(x, '... (c p1 p2) -> ... c (p1 p2)', c = self.image_channels, p1=self.patch_size, p2=self.patch_size)

            # normalize the reconstructed dct patches by subtracting the mean
            # In essence, this removes the grayscale information from the image.
            # For channel_diversity_loss, we don't care about getting the right grayscale 
            # cross-channel intensities.
            # We instead care about getting the right variation across
            # channels, per pixel.
            patch_hats = pull_out_channels(dct_patches.patches)[mask]
            patch = pull_out_channels(dct_normalized_patches)[mask]

            channel_diversity_loss = F.mse_loss(
                    patch - patch.mean(dim=-2, keepdim=True),
                    patch_hats - patch_hats.mean(dim=-2, keepdim=True),
            ) * self.channel_diversity_loss_coeff
        else:
            channel_diversity_loss = 0.0

        if not decode:
            return dict(
                perplexity=perplexity,
                commit_loss=commit_loss,
                rec_loss=rec_loss,
                codes=codes,
                channel_diversity_loss=channel_diversity_loss,
            )
    

        # TODO test if denormalizing normalizing AFTER doing the postprocessing padding gets better results
        # un normalizes patches
        dct_patches.patches = self.dct_processor.patch_norm.inverse_norm(dct_patches.patches, dct_patches.h_indices, dct_patches.w_indices, )

        images_hat = self.dct_processor.postprocess(dct_patches)

        # un normalizes patches
        dct_normalized_patches = self.dct_processor.patch_norm.inverse_norm(dct_normalized_patches, dct_patches.h_indices, dct_patches.w_indices, )
        dct_patches.patches = dct_normalized_patches
        images = self.dct_processor.postprocess(dct_patches)

        pixel_loss = 0.0

        for im, im_hat in zip(images, images_hat):
            pixel_loss += F.mse_loss(im, im_hat)

        pixel_loss = pixel_loss / len(images_hat)

        return dict(
            x_hat=images_hat,
            perplexity=perplexity,
            commit_loss=commit_loss,
            rec_loss=rec_loss,
            pixel_loss=pixel_loss,
            channel_diversity_loss=channel_diversity_loss,
            codes=codes,
            x=images,
        )
