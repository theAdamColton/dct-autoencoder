from typing import List
import torch.nn.functional as F
import torch
import torch.nn as nn

from .dct_processor import DCTPatches, DCTProcessor
from .na_vit import NaViT, FeedForward
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
    ):
        """
        input_res: the square integer input resolution size.
        """
        super().__init__()

        self.image_channels = image_channels
        self.patch_size = patch_size
        self.feature_channels = feature_channels
        self.max_n_patches = max_n_patches
        mlp_dim = 2048

        self.vq_norm_in = nn.Sequential(
            nn.LayerNorm(feature_channels),
        )

        self.proj_out = nn.Sequential(
            FeedForward(feature_channels, mlp_dim),
            nn.LayerNorm(feature_channels),
            nn.Linear(feature_channels, image_channels * patch_size**2),
        )

        self.encoder = NaViT(
            image_size=max_n_patches * patch_size,
            patch_size=patch_size,
            dim=feature_channels,
            depth=depth,
            heads=8,
            channels=image_channels,
            mlp_dim=mlp_dim,
            dropout=0.1,
            emb_dropout=0.1,
        )

        self.vq_model = vq_model
        self.vq_model.accept_image_fmap = False
        self.vq_model.channel_last = True

        max_res = patch_size * max_n_patches
        self.max_res = max_res

        self.dct_processor = dct_processor

    def forward(
        self,
        dct_patches: DCTPatches = None,
        pixel_values: List[torch.Tensor] = None,
        decode: bool = False,
    ):
        if pixel_values is not None:
            dct_patches = self.dct_processor.preprocess(pixel_values)

        dct_normalized_patches = dct_patches.patches.clone()

        z = self.encoder(dct_patches).patches

        z = dct_patches.patches
        mask = ~dct_patches.key_pad_mask

        # z = self.vq_norm_in(z)

        # TODO figure out how to make the mask work
        # with the vq model
        # z, codes, commit_loss = self.vq_model(z, mask=mask)
        codes = torch.Tensor([0]).to(torch.long).to(z.device)
        commit_loss = torch.Tensor([0.0]).to(z.device).to(z.dtype)

        with torch.no_grad():
            # perplexity = calculate_perplexity(codes[mask], self.vq_model.codebook_size)
            perplexity = torch.Tensor([0.0]).to(z.device).to(z.dtype)

        z = self.proj_out(z)

        # loss between z and normalized patches
        rec_loss = F.mse_loss(z[mask], dct_normalized_patches[mask])

        if not decode:
            return dict(
                perplexity=perplexity,
                commit_loss=commit_loss,
                rec_loss=rec_loss,
                codes=codes,
            )

        dct_patches.patches = z
        images_hat = self.dct_processor.postprocess(dct_patches)

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
            codes=codes,
            x=images,
            pixel_loss=pixel_loss,
        )
