from typing import Optional
import torch
from torch import nn
from einops import rearrange, repeat


def scatter_add_2d(
    x: torch.Tensor,
    pos_h: torch.LongTensor,
    pos_w: torch.LongTensor,
    y: Optional[torch.Tensor] = None,
):
    """
    x: 3d tensor

    adds y to x in the indices indicated by y along
    the last two dimensions

    in place
    """
    h, w, z = x.shape
    pos_flat = (pos_h * w + pos_w).flatten()
    pos_flat = repeat(pos_flat, "n -> n z", z=z)
    if y is None:
        y = torch.ones_like(pos_flat, dtype=x.dtype)
    x.view(h * w, z).scatter_add_(0, pos_flat, y)


class PatchNorm(nn.Module):
    """
    Records statistics of patch pixels,
    uses https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    to record estimates of mean and std
    """

    def __init__(
        self,
        max_n_patches_h: int,
        max_n_patches_w: int,
        patch_res: int,
        channels: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.eps = eps
        self.patch_res = patch_res
        self.channels = channels

        self.n = nn.Parameter(
            torch.zeros(
                max_n_patches_h,
                max_n_patches_w,
            ),
            requires_grad=False,
        )
        self.mean = nn.Parameter(
            torch.zeros(max_n_patches_h, max_n_patches_w, patch_res ** 2),
            requires_grad=False,
        )
        self.m2 = nn.Parameter(
            torch.zeros(max_n_patches_h, max_n_patches_w, patch_res ** 2),
            requires_grad=False,
        )

        self.frozen = False

    @property
    def var(self):
        mask = self.n < 2
        var = self.m2 / self.n.unsqueeze(-1).clamp(1.0)
        var[mask] = 1.0
        return var

    @property
    def std(self):
        return self.var.sqrt()

    def forward(
        self,
        patches: torch.Tensor,
        pos_h: torch.LongTensor,
        pos_w: torch.LongTensor,
        key_pad_mask: torch.BoolTensor,
    ):
        """
        normalizes patches using patch wieghts and biases that are indexed by pos_h and pos_w

        key_pad_mask is True where padding has been added

        patches should be (..., dim)
        """

        # first masks based on the key_pad_mask
        # this is important because we don't want the patch statistics effected
        # by the padding patches, which are all zeros
        pos_h = pos_h[~key_pad_mask]
        pos_w = pos_w[~key_pad_mask]
         
        # moves the channel dim up
        patches = rearrange(patches, 'b s (c p1 p2) -> b s c (p1 p2)', c = self.channels, p1=self.patch_res, p2=self.patch_res)
        patches_shape = patches.shape

        patches = patches[~key_pad_mask]


        if self.training and not self.frozen:
            with torch.no_grad():
                # updates n by incrementing all values at pos_h and pos_w
                scatter_add_2d(self.n.unsqueeze(-1), pos_h, pos_w)

                # updates the mean
                delta = patches - self.mean[pos_h, pos_w].unsqueeze(1)
                delta = delta.mean(dim=1)

                scatter_add_2d(
                    self.mean, pos_h, pos_w, delta / self.n[pos_h, pos_w].unsqueeze(-1)
                )

                delta2 = patches - self.mean[pos_h, pos_w].unsqueeze(1)
                delta2 = delta2.mean(dim=1)

                scatter_add_2d(self.m2, pos_h, pos_w, delta * delta2)

        patches = (patches - self.mean[pos_h, pos_w].unsqueeze(1)) / (
            self.std[pos_h, pos_w].unsqueeze(1) + self.eps
        )

        out = torch.zeros(patches_shape, dtype=patches.dtype, device=patches.device)
        out[~key_pad_mask] = patches

        # puts the channel dim back
        out = rearrange(out, 'b s c (p1 p2) -> b s (c p1 p2)', c=self.channels, p1=self.patch_res, p2=self.patch_res)
        return out

    def inverse_norm(
        self, patches: torch.Tensor, pos_h: torch.LongTensor, pos_w: torch.LongTensor
    ):
        patches = rearrange(patches, 'b s (c p1 p2) -> b s c (p1 p2)', c=self.channels, p1=self.patch_res, p2=self.patch_res)
        patches =  patches * (self.std[pos_h, pos_w].unsqueeze(-2) + self.eps) + self.mean[pos_h, pos_w].unsqueeze(-2)
        return rearrange(patches, 'b s c (p1 p2) -> b s (c p1 p2)', c=self.channels, p1=self.patch_res, p2=self.patch_res)
