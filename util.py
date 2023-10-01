import torch
import matplotlib.pyplot as plt
from torch_dct import dct_2d, idct_2d

def dct2(x, norm=None):
    return dct_2d(x, norm)

def idct2(x, norm=None):
    return idct_2d(x, norm)


def calculate_perplexity(codes, codebook_size, null_index=-1):
    """
    Perplexity is 2^(H(p)) where H(p) is the entropy over the codebook likelyhood

    the null index is assumed to be -1, perplexity is only calculated over the
    non null codes
    """
    dtype, device = codes.dtype, codes.device
    codes = codes.flatten()
    codes = codes[codes!= null_index]
    src = torch.ones_like(codes)
    counts = torch.zeros(codebook_size).to(dtype).to(device)
    counts = counts.scatter_add_(0, codes, src)

    probs = counts / codes.numel()
    # Entropy H(x) when p(x)=0 is defined as 0
    logits = torch.log2(probs)
    logits[probs == 0.0] = 0.0
    entropy = -torch.sum(probs * logits)
    return 2**entropy


def imshow(x:torch.Tensor, ax=None):
    if len(x.shape) > 2:
        x = x.permute(1,2,0)
    if x.dtype == torch.int:
        x = x*1.0
    if x.dtype != torch.bool:
        x = x - x.quantile(0.1)
        x = x / x.quantile(0.9)
    if ax is None:
        plt.imshow(x)
        plt.show()
    else:
        ax.imshow(x)

def is_triangular_number(x:int):
    return (8*x+1)**.5%1>0

def get_upper_left_tri_p(shape, p:float):
    """
    p: approximate percent masked
    """
    h,w = shape[-2], shape[-1]

    x,y = torch.meshgrid(torch.arange(w), torch.arange(h), indexing='xy')

    # distance from left upper corner
    dist = (x+y) * - 1.0

    p_largest_dists = dist.quantile(p)
    mask = dist > p_largest_dists

    to_expand = len(shape) - mask.ndim

    for _ in range(to_expand):
        mask = mask.unsqueeze(0)
    return mask

def get_upper_left_tri(shape, triangle_n: int = 0):
    h,w = shape[-2], shape[-1]

    assert triangle_n >= 1
    max_diag = w
    min_diag = -h - 2
    diagonal = max_diag - triangle_n
    assert diagonal >= min_diag

    ul_tri = torch.ones(h,w).triu(diagonal=diagonal).flip(1)

    to_expand = len(shape) - ul_tri.ndim
    for _ in range(to_expand):
        ul_tri = ul_tri.unsqueeze(0)
    return ul_tri

def get_circular_mask(shape, p: float=0.5):
    """
    Generate a circular mask, the circle being in the last two dimensions.

    The center of the circle will be zero, the circle is 'cut out' of the mask

    p: approximate percent masked
    """
    h, w = shape[-2], shape[-1]
    c = torch.Tensor([h / 2 - 0.5, w / 2 - 0.5])

    x,y = torch.meshgrid(torch.arange(w), torch.arange(h), indexing='xy')

    y = y - c[0]
    x = x - c[1]

    dist = torch.sqrt(x**2 + y**2)

    p_largest_dists = dist.quantile(p)

    mask = dist > p_largest_dists

    to_expand = len(shape) - mask.ndim
    for _ in range(to_expand):
        mask = mask.unsqueeze(0)

    return mask
    

def inverse_fft(fft_amp, fft_pha):
    imag = fft_amp * torch.sin(fft_pha)
    real = fft_amp * torch.cos(fft_pha)
    fft_y = torch.complex(real, imag)
    y = torch.fft.ifft2(fft_y)
    return y.real


def fft(x):
    x_fft = torch.fft.fft2(x)
    x_fft_amp = torch.sqrt(x_fft.real ** 2 + x_fft.imag ** 2)
    x_fft_phase = torch.atan2(x_fft.imag, x_fft.real)
    return x_fft_amp, x_fft_phase


def gaussian_fn(M, std):
    n = torch.arange(0, M) - (M - 1.0) / 2.0
    sig2 = 2 * std * std
    w = torch.exp(-n ** 2 / sig2)
    return w

def gkern(size=256, std=None):
    """Returns a 2D Gaussian kernel array."""
    if std is None:
        std = size / 2
    gkern1d = gaussian_fn(size, std=std)
    gkern2d = torch.outer(gkern1d, gkern1d)
    return gkern2d

