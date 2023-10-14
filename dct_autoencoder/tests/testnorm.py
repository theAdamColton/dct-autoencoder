from random import randint
from einops import reduce
import torch

from dct_autoencoder.norm import PatchNorm

# means and stds along the patch dim
patch_dim = 16
means = torch.arange(patch_dim)
stds = torch.arange(patch_dim)

patch_h = 10
patch_w = 10

patchnorm = PatchNorm(patch_h, patch_w, patch_dim)

def build_batch(n=100):
    x = []
    pos_h = []
    pos_w = [] 
    for _ in range(n):
        h = randint(0, patch_h-1)
        w = randint(0, patch_h-1)
        pos_h.append(h)
        pos_w.append(w)
        x.append(torch.randn(patch_dim) * stds + means)

    mask = torch.zeros(n, dtype=torch.bool)

    return torch.stack(x), torch.LongTensor(pos_h), torch.LongTensor(pos_w), mask
        

for _ in range(25):
    x, pos_h, pos_w, mask = build_batch()
    x_norm = patchnorm(x, pos_h, pos_w, mask)

    x_norm_std = reduce(patchnorm.std, 'h w z -> z', torch.std)
    x_norm_mean = reduce(patchnorm.mean, 'h w z -> z', torch.mean)

    print("predicted stds", x_norm_std)
    print("predicted means", x_norm_mean)
