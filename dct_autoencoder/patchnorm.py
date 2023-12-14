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

    adds y to x in the indices indicated by pos_c, pos_h and pos_w along
    the first 3 dimensions

    in place
    """
    c, h, w, z = x.shape
    i_flat = (pos_c * h * w + pos_h * w + pos_w).flatten()
    i_flat = repeat(i_flat, "n -> n z", z=z)
    if y is None:
        y = torch.ones_like(i_flat, dtype=x.dtype)
    x.view(h * w * c, z).scatter_add_(0, i_flat, y)


class PatchNorm(nn.Module):
    """ """

    def __init__(
        self,
        max_patch_h: int,
        max_patch_w: int,
        patch_size: int,
        channels: int,
        eps: float = 1e-6,
        max_val: float = 6.0,
        min_val: float = -6.0,
    ):
        super().__init__()
        self.eps = eps
        self.patch_size = patch_size
        self.channels = channels
        self.max_patch_h = max_patch_h
        self.max_patch_w = max_patch_w

        self.n = nn.Parameter(
            torch.zeros(
                channels,
                max_patch_h,
                max_patch_w,
            ),
            requires_grad=False,
        )

        self.median = nn.Parameter(
            torch.zeros(channels, max_patch_h, max_patch_w, patch_size**2),
            requires_grad=False,
        )
        # mean absolute deviation from the median
        self.b = nn.Parameter(
            torch.ones(channels, max_patch_h, max_patch_w, patch_size**2),
            requires_grad=False,
        )

        self.frozen = False

        self.max_val = max_val
        self.min_val = min_val

    @property
    def std(self) -> torch.Tensor:
        return self.b * 2**0.5

    
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
        pos_channels = pos_channels[~key_pad_mask]
        pos_h = pos_h[~key_pad_mask]
        pos_w = pos_w[~key_pad_mask]

        patches_shape = patches.shape

        patches = patches[~key_pad_mask]

        if self.training and not self.frozen:
            with torch.no_grad():
                # updates n by incrementing all values at pos_channels, pos_h and pos_w
                batch_n = torch.zeros(
                    self.channels,
                    self.max_patch_h,
                    self.max_patch_w,
                    device=patches.device,
                    dtype=patches.dtype,
                )
                scatter_add_3d(batch_n.unsqueeze(-1), pos_channels, pos_h, pos_w)

                batch_median = torch.zeros_like(self.median.data)
                # updates the median
                for i in range(self.max_patch_h):
                    for j in range(self.max_patch_w):
                        for c in range(self.channels):
                            mask = (pos_h == i) & (pos_w == j) & (pos_channels == c)
                            if mask.sum() == 0:
                                continue
                            median = patches[mask].median(0).values
                            batch_median[c, i, j] = median

                # updates the running median by taking the weighted average of the
                # current batch median, and the stored median.
                # This is probably good enough as an approximation
                self.median.data = (
                    self.median * self.n.unsqueeze(-1)
                    + batch_median * batch_n.unsqueeze(-1)
                ) / (self.n + batch_n).clamp(1).unsqueeze(-1)

                distances = (patches - self.median[pos_channels, pos_h, pos_w]).abs()

                batch_b = torch.zeros_like(self.b)
                scatter_add_3d(batch_b, pos_channels, pos_h, pos_w, distances)
                batch_b = batch_b / batch_n.unsqueeze(-1).clamp(1)

                self.b.data = (
                    self.b * self.n.unsqueeze(-1) + batch_b * batch_n.unsqueeze(-1)
                ) / (self.n + batch_n).clamp(1).unsqueeze(-1)

                self.n.data = self.n + batch_n


        medians = self.median[pos_channels, pos_h, pos_w]
        std = self.std[pos_channels, pos_h, pos_w] + self.eps
        n = self.n[pos_channels, pos_h, pos_w]
        mask = n <= 2

        patches = (patches - medians) / std

        patches[mask] = 0.0

        patches.clamp_(self.min_val, self.max_val)

        out = torch.zeros(patches_shape, dtype=patches.dtype, device=patches.device)
        out[~key_pad_mask] = patches

        return out

    def inverse_norm(self, dct_patches: DCTPatches) -> torch.Tensor:
        patches = dct_patches.patches

        pos_channels = dct_patches.patch_channels
        pos_h = dct_patches.h_indices
        pos_w = dct_patches.w_indices

        medians = self.median[pos_channels, pos_h, pos_w]
        std = self.std[pos_channels, pos_h, pos_w] + self.eps

        return patches * std + medians
