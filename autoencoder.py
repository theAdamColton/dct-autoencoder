from typing import List, Tuple
import torch.nn.functional as F
import torch
import torch.nn as nn
from einops import rearrange, reduce
import torchvision.transforms.v2 as transforms

from na_vit import NaViT

from util import (
    calculate_perplexity,
    dct2,
    idct2,
    ema_update_2d,
)

class DCTAutoencoderTransformer(nn.Module):
    def __init__(
        self,
        vq_model,
        image_channels: int=3,
        depth:int =4,
        feature_channels: int=768,
        patch_size: int=32,
        # only take 75 of the dct features and pass to the encoder
        dct_compression_factor: float = 0.75,
        max_n_patches: int = 512,
    ):
        """
        input_res: the square integer input resolution size.
        """
        super().__init__()


        self.image_channels = image_channels
        self.patch_size = patch_size
        self.feature_channels = feature_channels
        self.dct_compression_factor = dct_compression_factor
        self.max_n_patches = max_n_patches

        self.vq_norm_in = nn.Sequential(
            nn.LayerNorm(feature_channels),
        )

        self.proj_out = nn.Sequential(
            #nn.GELU(),
            #nn.LayerNorm(feature_channels),
            nn.Linear(feature_channels, image_channels * patch_size ** 2),
        )

        self.encoder = NaViT(
            image_size=max_n_patches * patch_size,
            patch_size=patch_size,
            dim=feature_channels,
            depth=depth,
            heads=4,
            channels=image_channels,
            mlp_dim=1024,
            dropout=0.1,
            emb_dropout=0.1,
            token_dropout_prob=None,  # token dropout of 10% (keep 90% of tokens)
        )

        self.vq_model = vq_model
        self.vq_model.accept_image_fmap = False
        self.vq_model.channel_last = True

        self.ema_alpha = 1e-4
        max_res = patch_size * max_n_patches
        self.max_res = max_res
        self.dct_mean = nn.Parameter(torch.zeros(max_res, max_res), requires_grad=False)
        self.dct_std = nn.Parameter(torch.ones(max_res, max_res), requires_grad=False)


    def dct_norm(self, x:torch.Tensor, eps=1e-7):
        *_, h, w = x.shape
        mean = reduce(x, '... h w -> h w', 'mean')
        std = reduce(x, '... h w -> h w', torch.var)

        if self.training:
            ema_update_2d(self.dct_mean, mean, self.ema_alpha)

        # normalizes
        mean = self.dct_mean[:h, :w]
        std = self.dct_std[:h, :w]

        return (x - mean) / torch.clamp(std, eps)

    def dct_unnorm(self, x:torch.Tensor):
        *_, h, w = x.shape
        mean = self.dct_mean[:h, :w]
        std = self.dct_std[:h, :w]
        return (x+mean) * std

    @torch.no_grad()
    def prepare_batch(self, batch: List[torch.Tensor]):
        """
        batch: list of tensors containing unnormalized image pixels

        returns a list of dct features tensors which are cropped to
            the correct dimensions
        """
        dct_features = []
        original_sizes = []

        for x in batch:
            # TODO take features that are closer to 0,0
            # rather than a rectangle

            c, h, w = x.shape
            if h < self.patch_size:
                rz = transforms.Resize((self.patch_size, int((self.patch_size / h) * w)))
                x = rz(x)
            elif w < self.patch_size:
                rz = transforms.Resize((int((self.patch_size / w) * h), self.patch_size))
                x = rz(x)
            c, h, w = x.shape

            original_sizes.append((h,w))

            x_dct = dct2(x, "ortho")
        
            h_c, w_c = (1-self.dct_compression_factor) * h, (1-self.dct_compression_factor) * w

            p_h = round(h_c / self.patch_size)
            p_w = round(w_c / self.patch_size)
            p_h = max(p_h, 1)
            p_w = max(p_w, 1)

            # we need that
            # ar = p_h_c / p_w_c = p_h / p_w
            # p_h_c * p_w_c <= self.max_n_patches
            # ar * p_w_c <= self.max_n_patches / p_w_c
            # ar * p_w_c ** 2 <= self.max_n_patches
            # p_w_c = sqrt(self.max_n_patches / ar)
            # p_h_c = ar * p_w_c

            ar = p_h/p_w
            p_w_c = int((self.max_n_patches / ar)**0.5)
            p_h_c = int(ar * p_w_c)

            assert p_h_c * p_w_c <= self.max_n_patches

            p_h = min(p_h, p_h_c)
            p_w = min(p_w, p_w_c)
            p_h = max(p_h, 1)
            p_w = max(p_w, 1)

            dct_h = p_h * self.patch_size
            dct_w = p_w * self.patch_size

            assert dct_h % self.patch_size == 0
            assert dct_w % self.patch_size == 0
            assert dct_h <= self.max_res
            assert dct_w <= self.max_res
            assert dct_h >= self.patch_size
            assert dct_w >= self.patch_size

            assert h >= self.patch_size
            assert w >= self.patch_size

            x_dct = x_dct[..., :dct_h, :dct_w]

            dct_features.append(x_dct)


        return dct_features, original_sizes


    def forward(self, pixel_values: List[torch.Tensor]=None, dct_features: List[torch.Tensor]=None, original_sizes:List[Tuple] = None, decode:bool=True):
        """
        x: list of tensors containing unnormalized image pixels
        """

        if dct_features is None:
            # list of dct tensors
            x, original_sizes = self.prepare_batch(pixel_values)
        else:
            x = dct_features

        x = [self.dct_norm(feat) for feat in x]


        encoder_out = self.encoder(x,
             group_images = True,
             group_max_seq_len = self.max_n_patches*2,
        )
        z = encoder_out['x']
        mask = encoder_out['key_pad_mask']
        revert_patching = encoder_out['revert_patching']
        patches = encoder_out['patches']
        mask = ~mask

        z = self.vq_norm_in(z)

        # TODO figure out how to make the mask work
        # with the vq model
        #z, codes, commit_loss = self.vq_model(z, mask=mask)
        codes = torch.Tensor([0]).to(torch.long).to(z.device)
        commit_loss = torch.Tensor([0.0]).to(z.device).to(z.dtype)

        with torch.no_grad():
            #perplexity = calculate_perplexity(codes[mask], self.vq_model.codebook_size)
            perplexity = torch.Tensor([0.0]).to(z.device).to(z.dtype)

        z = self.proj_out(z)

        rec_loss = F.mse_loss(z[mask], patches[mask])


        #from util import imshow
        #import matplotlib.pyplot as plt

        #patch_rec = revert_patching(patches)

        ## this needs to pass, they should be exactly the same
        #assert torch.equal(patch_rec[0].reshape(3,256,256), x[0])

        if not decode:
            return dict(
                    perplexity=perplexity,
                    commit_loss=commit_loss,
                    rec_loss=rec_loss,
                    codes=codes,
                    z=z,
                    x=x,
            )


        # reverts patching, z is now a list of tensors,
        # each tensor being an image of patches
        z = revert_patching(z)

        pixel_loss = 0.0

        x_hat = []
        x_images = []

        with torch.no_grad():
            for x_dct_image, z_dct_image, (h,w) in zip(x, z, original_sizes):

                #z_dct_image = rearrange(z_dct_image, '(h w) (c p1 p2) -> c (h p1) (w p2)', p1 = self.patch_size, p2=self.patch_size, w=w)

                #z_dct_image = rearrange(z_dct_image, 'h w (c p1 p2) -> c (h p1) (w p2)', p1=self.patch_size, p2=self.patch_size, c=self.image_channels)

                def pad(to_pad:torch.Tensor, h,w):
                    c, ih, iw = to_pad.shape
                    padded = torch.zeros(c,h,w, device=to_pad.device, dtype=to_pad.dtype)
                    padded[:, :ih, :iw] = to_pad
                    return padded
        

                # un normalize
                x_dct_image = self.dct_unnorm(x_dct_image)
                z_dct_image = self.dct_unnorm(z_dct_image)


                # pads back to image size
                x_dct_image = pad(x_dct_image, h,w)
                z_dct_image = pad(z_dct_image, h,w)

                # FFT doesn't support cuda for non powers of two
                x_image = idct2(x_dct_image.detach().float().cpu(), "ortho")
                z_image = idct2(z_dct_image.detach().float().cpu(), "ortho")
                pixel_loss = F.mse_loss(x_image, z_image)

                x_hat.append(z_image)
                x_images.append(x_image)


        return dict(
            x_hat=x_hat,
            perplexity=perplexity,
            commit_loss=commit_loss,
            rec_loss=rec_loss,
            codes=codes,
            z=z,
            x=x_images,
            pixel_loss=pixel_loss,
        )
