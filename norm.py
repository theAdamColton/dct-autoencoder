from einops import reduce
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
    def __init__(self, h: int, w: int, dtype=torch.float, device=torch.device('cpu')):
        super().__init__()
        self.h = h
        self.w = w
        self.n = nn.Parameter(torch.zeros(h,w,dtype=torch.long,device=device), requires_grad=False)
        self.mean = nn.Parameter(torch.zeros(h,w,dtype=dtype,device=device), requires_grad=False)
        self.m2 = nn.Parameter(torch.zeros(h,w,dtype=dtype,device=device), requires_grad=False)

    def forward(self, x: torch.Tensor):
        if not self.training:
            return

        *_, h, w =x.shape
        assert h <= self.h
        assert w <= self.w

        self.n[:h, :w] = self.n[:h, :w] + 1
        delta = reduce(x, '... h w -> h w', torch.mean) - self.mean[:h, :w]
        self.mean[:h, :w] = self.mean[:h, :w] + delta / self.n[:h, :w]
        delta_2 = reduce(x, '... h w -> h w', torch.mean) - self.mean[:h, :w]
        self.m2[:h, :w] = self.m2[:h, :w] + delta * delta_2

    @property
    def variance(self):
        return self.m2 / torch.clamp(self.n - 1, 1e-3)

    @property
    def std(self):
        return self.variance.sqrt()


