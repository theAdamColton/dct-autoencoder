"""
Lookup Free Quantization
Proposed in https://arxiv.org/abs/2310.05737

basically a 2-level FSQ (Finite Scalar Quantization) with entropy loss
https://arxiv.org/abs/2309.15505

from vector-quantize-pytorch
"""

from math import log2, ceil

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.nn import Module

from einops import rearrange, reduce, pack, unpack

# helper functions

def exists(v):
    return v is not None

def default(*args):
    for arg in args:
        if exists(arg):
            return arg() if callable(arg) else arg
    return None

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

def mult_along_first_dims(x, y):
    # returns x * y elementwise
    ndim_to_expand = x.ndim - y.ndim
    return x * y[..., *[None for _ in range(ndim_to_expand)]]

def masked_mean(x, m, dim=None):
    # takes the mean of the elements of x that are not masked
    # m is 1.0 where padding is
    x = mult_along_first_dims(x, ~m)
    x = x / (~m).sum()
    if dim is None:
        return x.sum()
    else:
        return x.sum(dim=dim)

# entropy

def entropy_loss(affinity:torch.Tensor, temperature=0.01, eps=1e-5, pad_mask=None):
    """
    affinity: last dim is the affinity to all codes

    same formulation as https://github.com/google-research/maskgit/blob/1db23594e1bd328ee78eadcd148a19281cd0f5b8/maskgit/libml/losses.py#L190
        as maskgit

    same default temperature as in maskgit vqgan

    pad_mask: contains False where padding is
        applies to the leading dims of affinity
    """
    pad_mask = rearrange(pad_mask, 'b s -> (b s)')
    affinity = rearrange(affinity, 'b s d z -> (b s) d z')

    probs = (affinity / temperature).softmax(dim=-1)
    log_probs = F.log_softmax((affinity / temperature) + eps, dim=-1)

    # masked mean over all dims apart from the last z dim
    # this should be within floating point errors of 
    # probs[~pad_mask].reshape(-1, probs.shape[-1]).mean(dim=0)
    avg_probs = masked_mean(probs, pad_mask, dim=0).mean(dim=0)

    avg_entropy = -1 * (avg_probs * (avg_probs + eps).log()).sum()
    sample_entropy = -1 * masked_mean((probs*log_probs).sum(dim=-1), pad_mask)
    loss = sample_entropy - avg_entropy
    return loss

# class

class LFQ(Module):
    def __init__(
        self,
        *,
        dim = None,
        codebook_size = None,
        diversity_gamma = 2.5,
        straight_through_activation = nn.Identity(),
        num_codebooks = 1,
        keep_num_codebooks_dim = None,
        codebook_scale = 1.  # for residual LFQ, codebook scaled down by 2x at each layer
    ):
        super().__init__()

        # some assert validations

        assert exists(dim) or exists(codebook_size), 'either dim or codebook_size must be specified for LFQ'
        assert not exists(codebook_size) or log2(codebook_size).is_integer(), f'your codebook size must be a power of 2 for lookup free quantization (suggested {2 ** ceil(log2(codebook_size))})'

        codebook_size = default(codebook_size, lambda: 2 ** dim)
        codebook_dim = int(log2(codebook_size))

        codebook_dims = codebook_dim * num_codebooks
        dim = default(dim, codebook_dims)

        has_projections = dim != codebook_dims
        self.project_in = nn.Linear(dim, codebook_dims) if has_projections else nn.Identity()
        self.project_out = nn.Linear(codebook_dims, dim) if has_projections else nn.Identity()
        self.has_projections = has_projections

        self.dim = dim
        self.codebook_dim = codebook_dim
        self.num_codebooks = num_codebooks

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        # straight through activation

        self.activation = straight_through_activation

        # entropy aux loss related weights

        self.diversity_gamma = diversity_gamma

        # codebook scale

        self.codebook_scale = codebook_scale

        # for no auxiliary loss, during inference

        self.register_buffer('mask', 2 ** torch.arange(codebook_dim - 1, -1, -1))
        self.register_buffer('zero', torch.tensor(0.), persistent = False)

        # codes

        all_codes = torch.arange(codebook_size)
        bits = ((all_codes[..., None].int() & self.mask) != 0).float()
        codebook = self.bits_to_codes(bits)

        self.register_buffer('codebook', codebook, persistent = False)

    def bits_to_codes(self, bits):
        return bits * self.codebook_scale * 2 - self.codebook_scale

    @property
    def dtype(self):
        return self.codebook.dtype

    def indices_to_codes(
        self,
        indices,
        project_out = True
    ):
        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))

        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, '... -> ... 1')

        # indices to codes, which are bits of either -1 or 1

        bits = ((indices[..., None].int() & self.mask) != 0).to(self.dtype)

        codes = self.bits_to_codes(bits)

        codes = rearrange(codes, '... c d -> ... (c d)')

        # whether to project codes out to original dimensions
        # if the input feature dimensions were not log2(codebook size)

        if project_out:
            codes = self.project_out(codes)

        # rearrange codes back to original shape

        if is_img_or_video:
            codes = rearrange(codes, 'b ... d -> b d ...')

        return codes

    def forward(
        self,
        x,
        temperature = 0.01,
        pad_mask=None,
    ):
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension, which is also log2(codebook size)
        c - number of codebook dim

        pad_mask: contains True where padding is
        """

        is_img_or_video = x.ndim >= 4

        if pad_mask is None:
            raise NotImplementedError('pad mask')

        # standardize image or video into (batch, seq, dimension)

        if is_img_or_video:
            x = rearrange(x, 'b d ... -> b ... d')
            x, ps = pack_one(x, 'b * d')

        assert x.shape[-1] == self.dim, f'expected dimension of {self.dim} but received {x.shape[-1]}'

        x = self.project_in(x)

        # split out number of codebooks

        x = rearrange(x, 'b n (c d) -> b n c d', c = self.num_codebooks)

        # quantize by eq 3.

        original_input = x

        codebook_value = torch.ones_like(x) * self.codebook_scale
        quantized = torch.where(x > 0, codebook_value, -codebook_value)

        # use straight-through gradients with tanh (or custom activation fn) if training

        if self.training:
            x = self.activation(x)
            x = x - x.detach() + quantized
        else:
            x = quantized

        # calculate indices

        indices = reduce((x > 0).int() * self.mask.int(), 'b n c d -> b n c', 'sum')

        if self.training:
            # the same as euclidean distance up to a constant
            distance = -2 * einsum('... i d, j d -> ... i j', original_input, self.codebook)
            entropy_aux_loss = entropy_loss(-distance, temperature=temperature, pad_mask=pad_mask)
        else:
            entropy_aux_loss = self.zero

        if self.training:
            if pad_mask is not None:
                # masked mse loss
                commit_loss = F.mse_loss(original_input, quantized.detach(), reduction='none')
                # Equivalent to commit_loss[~pad_mask].mean()
                commit_loss = masked_mean(commit_loss, pad_mask, dim=0).sum(0).mean()
            else:
                commit_loss = F.mse_loss(original_input, quantized.detach())
        else:
            commit_loss = self.zero

        # merge back codebook dim

        x = rearrange(x, 'b n c d -> b n (c d)')

        # project out to feature dimension if needed

        x = self.project_out(x)

        # reconstitute image or video dimensions

        if is_img_or_video:
            x = unpack_one(x, ps, 'b * d')
            x = rearrange(x, 'b ... d -> b d ...')

            indices = unpack_one(indices, ps, 'b * c')

        # whether to remove single codebook dim

        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, '... 1 -> ...')

        return x, indices, commit_loss, entropy_aux_loss
