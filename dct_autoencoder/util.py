import torch.nn.functional as F
from typing import Optional, List, Tuple
import torch
from torch import nn
import matplotlib.pyplot as plt
from torch_dct import dct_2d, idct_2d
import random
import math
from PIL import ImageDraw
from PIL import ImageFont
from einops import einsum


#https://ixora.io/projects/colorblindness/color-blindness-simulation-research/

# sRGB -> XYZ D65
MsRGB = torch.Tensor([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
#MsRGB = torch.Tensor([[.4360747, .3850649, .1430804],
#                  [0.2225045, 0.7168786, 0.0606169],
#                  [0.0139322, 0.0971045, 0.7141733]])
# XYZ D65 -> LMS
MHPE = torch.Tensor([[ 0.4002, 0.7076, -0.0807],
                 [-0.2280, 1.1500,  0.0612],
                 [      0,      0,  0.9184]])

# LMS -> ipt
Mipt = torch.Tensor([[0.4, 0.4, 0.2],
                     [4.455, -4.851, 0.3960],
                     [0.8056, .3572, -1.1628]])
Trgb2lms = MHPE @ MsRGB
Tlms2rgb = Trgb2lms.inverse()


def channel_mult(M, x):
    return einsum(
            M,
            x,
            'i j, ... j h w -> ... i h w')


def add_txt_to_pil_image(image, text):
    imd = ImageDraw.Draw(image)
    font = ImageFont.truetype('FreeMono.ttf', 48)
    imd.text((5, 5), text, font=font, fill =(255, 255, 255))


def rgb_to_lms(x:torch.Tensor):
    return channel_mult(
            Trgb2lms.to(x.dtype).to(x.device),
            x,)


def lms_to_rgb(x:torch.Tensor):
    return channel_mult(Tlms2rgb.to(x.dtype).to(x.device),
                x,)
    
def rgb_to_ipt(x:torch.Tensor):
    """
    page 147
    https://scholarworks.rit.edu/theses/2858/
    """
    x = rgb_to_lms(x)
    mask = x >= 0.0
    x[mask] = x[mask] ** 0.43
    x[~mask] = -1 * (-1 * x[~mask] ) ** 0.43
    return channel_mult(Mipt.to(x.dtype).to(x.device),
                  x,)

def ipt_to_rgb(x: torch.Tensor):
    """
    page 147
    https://scholarworks.rit.edu/theses/2858/
    """
    x = channel_mult(Mipt.inverse().to(x.dtype).to(x.device),
                  x,)
    mask = x >= 0.0
    x[mask] = x[mask] ** 2.3256
    x[~mask] = -1 * (-1 * x[~mask] ) ** 2.3256
    return lms_to_rgb(x)


def rgb_to_ycbcr(x:torch.Tensor):
    """
    x: (..., c, h, w)

    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion
    """

    r = x[..., 0, :, :]
    g = x[..., 1, :, :]
    b = x[..., 2, :, :]

    y = 0.299 * r + 0.587 * g + 0.114 * b

    cb = (-0.299 * r - 0.587 * g + 0.866* b ) / 1.772 + 0.5
    cr = (0.701 * r - 0.587 * g - 0.144 * b ) / 1.402 + 0.5

    return torch.stack([y, cb, cr], dim=-3)

def ycbcr_to_rgb(x: torch.Tensor):
    """
    x: (..., c, h, w)

    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion
    """
    y = x[..., 0, :, :]
    cb = x[..., 1, :, :]
    cr = x[..., 2, :, :]

    r = y + 1.402 * (cr - 0.5)
    g = y - (0.114*1.772 * (cb-0.5) + 0.299 * 1.402* (cr -0.5 ) ) / 0.587
    b = y + 1.772 * (cb -0.5)

    return torch.stack([r,g,b,], dim=-3)
    

def lightness(x: torch.Tensor, c_dim=-3, keepdim=False):
    """
    uses Y204 (Adobe) https://en.wikipedia.org/wiki/HSL_and_HSV#Lightness
    """
    color_weights = torch.Tensor([0.212, 0.701, 0.087]).to(x.device).to(x.dtype)
    to_unsqueeze = abs(c_dim+1)
    for _ in range(to_unsqueeze):
        color_weights = color_weights.unsqueeze(-1)
    return (color_weights* x).mean(dim=c_dim, keepdim=keepdim)




def pad_sequence(seq:List[torch.Tensor], max_seq_len):
    """
    pads the first dim to max_seq_len and stacks

    pads to the right
    """
    back_dims = seq[0].shape[1:]
    b = len(seq)
    out = torch.zeros((b, max_seq_len, *back_dims), dtype=seq[0].dtype, device=seq[0].device)
    for i in range(len(seq)):
        l = seq[i].shape[0]
        assert l <= max_seq_len
        out[i,:l] = seq[i]
    return out

def exp_dist(a: float) -> float:
    x = random.random()
    return -1 / a * math.log(x)

def uniform(a:float, b:float) -> float:
    x = random.random() 
    if a > b:
        tmp = a
        a = b
        b = tmp
    return x * (b-a) + a

def power_of_two(target:int) -> int:
    if target > 1:
        for i in range(1, int(target)):
            if (2 ** i >= target):
                return 2 ** i
    return 1


def divisible_by(numer, denom):
    return (numer % denom) == 0


def always(val):
    return lambda *_: val


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def ema_update_2d(old: torch.Tensor, new: torch.Tensor, alpha: float = 0.8):
    *_, h, w = new.shape
    old[:h, :w] = alpha * new[:h, :w] + (1 - alpha) * old[:h, :w]


def get_square_dct_basis(resolution: int = 16):
    """
    gets a square dct basis

    returns a (resolution, resolution, resolution, resolution) basis

    where the first two dimensions are the x,y dct coords
    """
    x, y = torch.meshgrid(torch.arange(resolution), torch.arange(resolution))
    u, v = torch.meshgrid(torch.arange(resolution), torch.arange(resolution))
    u = u.unsqueeze(-1).unsqueeze(-1)
    v = v.unsqueeze(-1).unsqueeze(-1)
    dct_basis_images = torch.cos(
        ((2 * x + 1) * u * torch.pi) / (2 * resolution)
    ) * torch.cos(((2 * y + 1) * v * torch.pi) / (2 * resolution))
    return dct_basis_images


def zigzag(h: int, w: int):
    """
    returns zigzag indices

    see:
        Zigzag ordering of JPEG image components
        https://en.wikipedia.org/wiki/JPEG#JPEG_codec_example
    """
    out = torch.empty((h,w), dtype=torch.long)

    row, col = 0, 0

    current_value = 0

    for _ in range(h * w):
        out[row, col] = current_value
        current_value += 1

        # goes /    on odd diagonals and  ^ on evens
        #     v                          /

        up_right = (row + col) % 2 == 0

        if up_right:
            # if can't go up right
            # because the col is at the edge
            if col == w - 1:
                row += 1
            elif row == 0:
                col += 1
            else:
                row -= 1
                col += 1
        else:
            if row == h - 1:
                col += 1
            elif col == 0:
                row += 1
            else:
                row += 1
                col -= 1

    return out


def flatten_zigzag(x: torch.Tensor, zigzag_indices: Optional[torch.Tensor] = None):
    """
    you should specify the zigzag_indices if you can because it will save processing on
    repeated calls:
      zigzag_indices = zigzag(h,w)

    x can have any number of leading dimensions

    returns x flattened in zigzag order
    """
    h, w = x.shape[-2], x.shape[-1]
    leading_dimensions = x.shape[:-2]

    if zigzag_indices is None:
        zigzag_indices = zigzag(h, w).to(x.device)

    x = x.reshape(*leading_dimensions, h * w)
    zigzag_indices = zigzag_indices.flatten().repeat(*leading_dimensions, 1)

    return torch.zeros_like(x).scatter(-1, zigzag_indices, x)


def unflatten_zigzag(
    x: torch.Tensor, h: int, w: int, zigzag_indices: Optional[torch.Tensor] = None
):
    """
    inverse of flatten_zigzag
    """
    leading_dimensions = x.shape[:-1]

    if zigzag_indices is None:
        zigzag_indices = zigzag(h, w).to(x.device)

    return torch.gather(
        x, -1, zigzag_indices.flatten().repeat(*leading_dimensions, 1)
    ).reshape(*leading_dimensions, h, w)


class ZigzagFlattener(nn.Module):
    def __init__(self, h: int, w: int):
        super().__init__()
        self.zigzag = nn.Parameter(zigzag(h, w), requires_grad=False)
        self.h = h
        self.w = w

    def flatten(self, x: torch.Tensor):
        h, w = x.shape[-2:]
        assert h == self.h
        assert w == self.w
        x = flatten_zigzag(x, self.zigzag)
        return x

    def unflatten(self, x: torch.Tensor):
        x = unflatten_zigzag(x, self.h, self.w, self.zigzag)
        return x


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
    codes = codes[codes != null_index]
    src = torch.ones_like(codes)
    counts = torch.zeros(codebook_size).to(dtype).to(device)
    counts = counts.scatter_add_(0, codes, src)

    probs = counts / codes.numel()
    # Entropy H(x) when p(x)=0 is defined as 0
    logits = torch.log2(probs)
    logits[probs == 0.0] = 0.0
    entropy = -torch.sum(probs * logits)
    return 2**entropy


def imshow(x: torch.Tensor, ax=None):
    x = x.cpu()
    if len(x.shape) > 2:
        x = x.permute(1, 2, 0)
    if x.dtype == torch.int:
        x = x * 1.0
    if x.dtype != torch.bool:
        x = x - x.quantile(0.1)
        x = x / x.quantile(0.9)
        x = x.clamp(0.0, 1.0)

    if ax is None:
        ax = plt
        ax.imshow(x)
        plt.show()
    else:
        ax.axis("off")
        ax.tick_params(
            axis="both",
            which="both",  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False,
        )
        ax.imshow(x)


def is_triangular_number(x: int):
    return (8 * x + 1) ** 0.5 % 1 > 0


def get_upper_left_tri_p_w_channel_preferences(shape, p: float, channel_preferences: Tuple):
    """
    shape: tuple of sizes: c, h, w

    p: percent unmasked, or percent integrity
    p=0.0 means no compression or loss, the mask is all ones

    the mask returned is 0 where the value should be dropped/zeroed and 1
    where the value should be kept

    channel_preferences: a tuple of floats indicating the relative importance
    of each channel, example: (4, 1, 1) if channel 0 is 4 times as important as
    channel 1 and channel 2

    """
    c,h,w = shape

    prefs = torch.Tensor(channel_preferences)
    prefs = prefs / prefs.sum()

    channel_ps = prefs * c * p


    tri_masks = [get_upper_left_tri_p((h,w), channel_p.item()) for channel_p in channel_ps]
    tri_masks = torch.stack(tri_masks, dim=0)

    return tri_masks


def get_upper_left_tri_p(shape, p: float):
    """
    p: approximate percent masked
    """
    h, w = shape[-2], shape[-1]

    x, y = torch.meshgrid(torch.arange(w), torch.arange(h), indexing="xy")

    # distance from left upper corner
    dist = (x + y) * -1.0

    p_largest_dists = dist.quantile(p)
    mask = dist > p_largest_dists

    to_expand = len(shape) - mask.ndim

    for _ in range(to_expand):
        mask = mask.unsqueeze(0)
    return mask


def get_upper_left_tri(shape, triangle_n: int = 0):
    h, w = shape[-2], shape[-1]

    assert triangle_n >= 1
    max_diag = w
    min_diag = -h - 2
    diagonal = max_diag - triangle_n
    assert diagonal >= min_diag

    ul_tri = torch.ones(h, w).triu(diagonal=diagonal).flip(1)

    to_expand = len(shape) - ul_tri.ndim
    for _ in range(to_expand):
        ul_tri = ul_tri.unsqueeze(0)
    return ul_tri


def get_circular_mask(shape, p: float = 0.5):
    """
    Generate a circular mask, the circle being in the last two dimensions.

    The center of the circle will be zero, the circle is 'cut out' of the mask

    p: approximate percent masked
    """
    h, w = shape[-2], shape[-1]
    c = torch.Tensor([h / 2 - 0.5, w / 2 - 0.5])

    x, y = torch.meshgrid(torch.arange(w), torch.arange(h), indexing="xy")

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
    x_fft_amp = torch.sqrt(x_fft.real**2 + x_fft.imag**2)
    x_fft_phase = torch.atan2(x_fft.imag, x_fft.real)
    return x_fft_amp, x_fft_phase


def gaussian_fn(M, std):
    n = torch.arange(0, M) - (M - 1.0) / 2.0
    sig2 = 2 * std * std
    w = torch.exp(-(n**2) / sig2)
    return w


def gkern(size=256, std=None):
    """Returns a 2D Gaussian kernel array."""
    if std is None:
        std = size / 2
    gkern1d = gaussian_fn(size, std=std)
    gkern2d = torch.outer(gkern1d, gkern1d)
    return gkern2d
