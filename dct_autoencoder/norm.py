from typing import Optional
import torch
from torch import nn
from einops import repeat

def scatter_add_2d(x: torch.Tensor, pos_h: torch.LongTensor, pos_w: torch.LongTensor, y: Optional[torch.Tensor]=None):
    """
    x: 3d tensor

    adds y to x in the indices indicated by y along
    the last two dimensions

    in place
    """
    h, w, z = x.shape
    pos_flat = (pos_h * w + pos_w).flatten()
    pos_flat = repeat(pos_flat, 'n -> n z', z=z)
    if y is None:
        y = torch.ones_like(pos_flat, dtype=x.dtype)
    x.view(h*w, z).scatter_add_(0, pos_flat, y)

class PatchNorm(nn.Module):
    """
    Records statistics of patch pixels,
    uses https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    to record estimates of mean and std
    """
    def __init__(self, patch_height_dim: int, patch_width_dim: int, patch_dim:int, eps:float=1e-4,):
        super().__init__()
        self.eps=eps
        self.patch_dim = patch_dim

        self.n = nn.Parameter(torch.zeros(patch_height_dim,patch_width_dim,), requires_grad=False)
        self.mean = nn.Parameter(torch.zeros(patch_height_dim, patch_width_dim, patch_dim),requires_grad=False)
        # initializes m2 at eps
        self.m2 = nn.Parameter(torch.ones(patch_height_dim, patch_width_dim, patch_dim) * eps, requires_grad=False)

        self.frozen = False

    @property
    def var(self):
        mask = self.n == 0
        var = self.m2.clamp(0.0, 1e9) / self.n.unsqueeze(-1).clamp(1.0)
        var[mask] = 1.0
        return var

    @property
    def std(self):
        return self.var.sqrt()

    @property
    def dtype(self):
        return self.mean.dtype


    def forward(self, patches: torch.Tensor, pos_h: torch.LongTensor, pos_w: torch.LongTensor, key_pad_mask: torch.BoolTensor):
        """
        normalizes patches using patch wieghts and biases that are indexed by pos_h and pos_w

        key_pad_mask is True where padding has been added

        patches should be (..., dim)
        """
        patches_shape = patches.shape
        old_dtype = patches.dtype
        patches = patches.to(self.dtype)
         
        # first masks based on the key_pad_mask
        # this is important because we don't want the patch statistics effected
        # by the padding patches, which are all zeros
        pos_h = pos_h[~key_pad_mask]
        pos_w = pos_w[~key_pad_mask]
        patches = patches[~key_pad_mask]

        if self.training and not self.frozen:
            with torch.no_grad():
                # updates n by incrementing all values at pos_h and pos_w
                scatter_add_2d(self.n.unsqueeze(-1), pos_h, pos_w)

                # updates the mean
                delta = patches - self.mean[pos_h, pos_w]
                scatter_add_2d(self.mean, pos_h, pos_w, delta / self.n[pos_h, pos_w, None])

                delta2 = patches - self.mean[pos_h, pos_w]
                scatter_add_2d(self.m2, pos_h, pos_w, delta * delta2)

        patches = (patches - self.mean[pos_h, pos_w]) / (self.std[pos_h, pos_w] + self.eps)
        patches = patches.to(old_dtype)

        print("min ", patches.min().item(), "max", patches.max().item())

        out = torch.zeros(patches_shape, dtype=patches.dtype, device=patches.device)
        out[~key_pad_mask] = patches
        return out

    def inverse_norm(self, patches: torch.Tensor, pos_h: torch.LongTensor, pos_w: torch.LongTensor):
        return patches * (self.std[pos_h, pos_w] + self.eps) + self.mean[pos_h, pos_w]


