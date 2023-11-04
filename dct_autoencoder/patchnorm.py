from typing import Optional
import torch
from torch import nn
from einops import repeat

from dct_autoencoder.dct_patches import DCTPatches


def scatter_add_3d(
    x: torch.Tensor,
    pos_c: torch.LongTensor,
    pos_h: torch.LongTensor,
    pos_w: torch.LongTensor,
    y: Optional[torch.Tensor] = None,
):
    """
    x: 4d tensor

    adds y to x in the indices indicated by y along
    the first 3 dimensions

    in place
    """
    c, h, w, z = x.shape
    i_flat = (pos_c * h * w +  pos_h * w + pos_w).flatten()
    i_flat = repeat(i_flat, "n -> n z", z=z)
    if y is None:
        y = torch.ones_like(i_flat, dtype=x.dtype)
    x.view(h * w * c, z).scatter_add_(0, i_flat, y)


class PatchNorm(nn.Module):
    """
    Records statistics of patch pixels,
    uses https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    to record estimates of mean and std
    """

    def __init__(
        self,
        max_patch_h: int,
        max_patch_w: int,
        patch_size: int,
        channels: int,
        eps: float = 1e-6,
        max_val: float = 5.0,
        min_val: float = -5.0,
    ):
        super().__init__()
        self.eps = eps
        self.patch_size = patch_size
        self.channels = channels

        self.n = nn.Parameter(
            torch.zeros(
                channels,
                max_patch_h,
                max_patch_w,
            ),
            requires_grad=False,
        )
        self.mean = nn.Parameter(
            torch.zeros(channels, max_patch_h, max_patch_w, patch_size ** 2),
            requires_grad=False,
        )
        self.m2 = nn.Parameter(
            torch.zeros(channels, max_patch_h, max_patch_w, patch_size ** 2),
            requires_grad=False,
        )

        self.frozen = False

        self.max_val = max_val
        self.min_val = min_val

    @property
    def var(self):
        mask = self.n < 2
        var = self.m2 / self.n.unsqueeze(-1).clamp(1)
        var[mask] = 1.0
        return var

    @property
    def std(self):
        return self.var.sqrt()

    def forward(
        self,
        dct_patches: DCTPatches,
    ) -> torch.Tensor:
        """
        normalizes patches using patch wieghts and biases that are indexed by pos_h and pos_w and channels

        key_pad_mask is True where padding has been added

        patches should be (..., dim)
        """

        patches = dct_patches.patches
        pos_channels = dct_patches.patch_channels
        pos_h = dct_patches.h_indices
        pos_w = dct_patches.w_indices
        key_pad_mask = dct_patches.key_pad_mask

        # first masks based on the key_pad_mask
        # this is important because we don't want the patch statistics effected
        # by the padding patches, which are all zeros
        pos_channels= pos_channels[~key_pad_mask]
        pos_h = pos_h[~key_pad_mask]
        pos_w = pos_w[~key_pad_mask]
         
        patches_shape = patches.shape

        patches = patches[~key_pad_mask]

        if self.training and not self.frozen:
            with torch.no_grad():
                # updates n by incrementing all values at pos_channels, pos_h and pos_w
                scatter_add_3d(self.n.unsqueeze(-1), pos_channels, pos_h, pos_w)

                # updates the mean
                delta = patches - self.mean[pos_channels, pos_h, pos_w]

                scatter_add_3d(
                    self.mean, pos_channels, pos_h, pos_w, delta / self.n[pos_channels, pos_h, pos_w].unsqueeze(-1)
                )

                delta2 = patches - self.mean[pos_channels, pos_h, pos_w]

                scatter_add_3d(self.m2, pos_channels, pos_h, pos_w, delta * delta2)

        patches = (patches - self.mean[pos_channels, pos_h, pos_w]) / (
            self.std[pos_channels, pos_h, pos_w] + self.eps
        )
        
        patches.clamp_(self.min_val, self.max_val)

        out = torch.zeros(patches_shape, dtype=patches.dtype, device=patches.device)
        out[~key_pad_mask] = patches

        return out

    def inverse_norm(
            self, dct_patches: DCTPatches
    ) -> torch.Tensor:
        patches = dct_patches.patches
        pos_channels = dct_patches.patch_channels
        pos_h = dct_patches.h_indices
        pos_w = dct_patches.w_indices
        return  patches * (self.std[pos_channels, pos_h, pos_w] + self.eps) + self.mean[pos_channels, pos_h, pos_w]
