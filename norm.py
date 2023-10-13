from einops import reduce, repeat
import torch
from torch import nn


class Norm3D(nn.Module):
    """
    computes mean and variance statistics from batches of data,
     over the pixels and channels of the last three dimensions,
     where the last two dimensions can be any size up to some 
     max size.

    uses https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    """
    def __init__(self, c: int, h: int, w: int, dtype=torch.float, device=torch.device('cpu')):
        super().__init__()
        self.h = h
        self.w = w
        self.c=c
        self.n = nn.Parameter(torch.zeros(h,w,dtype=torch.long,device=device), requires_grad=False)
        self.mean = nn.Parameter(torch.zeros(c,h,w,dtype=dtype,device=device), requires_grad=False)
        self.m2 = nn.Parameter(torch.zeros(c,h,w,dtype=dtype,device=device), requires_grad=False)

    def forward(self, x: torch.Tensor):
        if not self.training:
            return

        *_, c, h, w =x.shape
        assert c == self.c
        assert h <= self.h
        assert w <= self.w

        self.n[:h, :w] = self.n[:h, :w] + 1
        delta = reduce(x, '... c h w -> c h w', torch.mean) - self.mean[:,:h, :w]
        self.mean[:, :h, :w] = self.mean[:, :h, :w] + delta / self.n[None, :h, :w]
        delta_2 = reduce(x, '... c h w -> c h w', torch.mean) - self.mean[:,:h, :w]
        self.m2[:, :h, :w] = self.m2[:, :h, :w] + delta * delta_2

    @property
    def variance(self):
        """
        The variance where n==0 I define as 1.0
        """
        n = repeat(self.n, 'h w -> c h w', c=self.c)
        mask = n==0
        var = self.m2 / torch.clamp(n, 1)
        var[mask] = 1.0
        return var

    @property
    def std(self):
        return self.variance.sqrt()


