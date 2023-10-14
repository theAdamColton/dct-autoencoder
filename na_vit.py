"""
thanks lucidrains from vit_pytorch
"""

from functools import partial
from typing import List, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.utils.rnn import pad_sequence as orig_pad_sequence

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

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
    def __init__(self, patch_height_dim: int, patch_width_dim: int, patch_dim:int, eps:float=1e-2, ema_alpha: float = 1e-1):
        super().__init__()
        self.eps=eps
        self.patch_dim = patch_dim

        self.n = nn.Parameter(torch.zeros(patch_height_dim,patch_width_dim,), requires_grad=False)
        self.mean = nn.Parameter(torch.zeros(patch_height_dim, patch_width_dim, patch_dim),requires_grad=False)
        # initializes m2 at eps
        self.m2 = nn.Parameter(torch.ones(patch_height_dim, patch_width_dim, patch_dim) * eps, requires_grad=False)

    @property
    def var(self):
        mask = self.n == 0
        var = self.m2 / self.n.unsqueeze(-1).clamp(1.0)
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

        if self.training:
            with torch.no_grad():
                # updates n by incrementing all values at pos_h and pos_w
                scatter_add_2d(self.n.unsqueeze(-1), pos_h, pos_w)

                # updates the mean
                delta = patches - self.mean[pos_h, pos_w]
                scatter_add_2d(self.mean, pos_h, pos_w, delta / self.n[pos_h, pos_w, None])

                delta2 = patches - self.mean[pos_h, pos_w]
                scatter_add_2d(self.m2, pos_h, pos_w, delta * delta2)
                self.m2.clamp_(self.eps, 1e7)

        patches = (patches - self.mean[pos_h, pos_w]) / (self.std[pos_h, pos_w] + self.eps)
        patches = patches.to(old_dtype)

        print("min ", patches.min().item(), "max", patches.max().item())

        out = torch.zeros(patches_shape, dtype=patches.dtype, device=patches.device)
        out[~key_pad_mask] = patches
        return out

    def inverse_norm(self, patches: torch.Tensor, pos_h: torch.LongTensor, pos_w: torch.LongTensor):
        return patches * (self.std[pos_h, pos_w] + self.eps) + self.mean[pos_h, pos_w]


def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def always(val):
    return lambda *args: val

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def divisible_by(numer, denom):
    return (numer % denom) == 0

# auto grouping images

def group_images_by_max_seq_len(
    images: List[Tensor],
    patch_size: int,
    calc_token_dropout = None,
    max_seq_len = 2048

) -> List[List[Tensor]]:

    calc_token_dropout = default(calc_token_dropout, always(0.))

    groups = []
    group = []
    seq_len = 0

    if isinstance(calc_token_dropout, (float, int)):
        calc_token_dropout = always(calc_token_dropout)

    for image in images:
        assert isinstance(image, Tensor)

        image_dims = image.shape[-2:]
        ph, pw = map(lambda t: t // patch_size, image_dims)

        image_seq_len = (ph * pw)
        image_seq_len = int(image_seq_len * (1 - calc_token_dropout(*image_dims)))

        assert image_seq_len <= max_seq_len, f'image with dimensions {image_dims} exceeds maximum sequence length'

        if (seq_len + image_seq_len) > max_seq_len:
            groups.append(group)
            group = []
            seq_len = 0

        group.append(image)
        seq_len += image_seq_len

    if len(group) > 0:
        groups.append(group)

    return groups

# normalization
# they use layernorm without bias, something that pytorch does not offer

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer('beta', torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

# they use a query-key normalization that is equivalent to rms norm (no mean-centering, learned gamma), from vit 22B paper

class RMSNorm(nn.Module):
    def __init__(self, heads, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, 1, dim))

    def forward(self, x):
        normed = F.normalize(x, dim = -1)
        return normed * self.scale * self.gamma

# feedforward

def FeedForward(dim, hidden_dim, dropout = 0.):
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, dim),
        nn.Dropout(dropout)
    )

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.norm = LayerNorm(dim)

        self.q_norm = RMSNorm(heads, dim_head)
        self.k_norm = RMSNorm(heads, dim_head)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias = False),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x,
        context = None,
        mask = None,
        attn_mask = None
    ):
        x = self.norm(x)
        kv_input = default(context, x)

        qkv = (self.to_q(x), *self.to_kv(kv_input).chunk(2, dim = -1))

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        q = self.q_norm(q)
        k = self.k_norm(k)

        dots = torch.matmul(q, k.transpose(-1, -2))

        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            dots = dots.masked_fill(~mask, -torch.finfo(dots.dtype).max)

        if exists(attn_mask):
            dots = dots.masked_fill(~attn_mask, -torch.finfo(dots.dtype).max)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

        self.norm = LayerNorm(dim)

    def forward(
        self,
        x,
        mask = None,
        attn_mask = None
    ):
        for attn, ff in self.layers:
            x = attn(x, mask = mask, attn_mask = attn_mask) + x
            x = ff(x) + x

        return self.norm(x)

class NaViT(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., token_dropout_prob = None, pos_embed_before_proj:bool=True):
        super().__init__()
        image_height, image_width = pair(image_size)

        # what percent of tokens to dropout
        # if int or float given, then assume constant dropout prob
        # otherwise accept a callback that in turn calculates dropout prob from height and width

        self.calc_token_dropout = None

        if callable(token_dropout_prob):
            self.calc_token_dropout = token_dropout_prob

        elif isinstance(token_dropout_prob, (float, int)):
            assert 0. < token_dropout_prob < 1.
            token_dropout_prob = float(token_dropout_prob)
            self.calc_token_dropout = lambda height, width: token_dropout_prob

        # calculate patching related stuff

        assert divisible_by(image_height, patch_size) and divisible_by(image_width, patch_size), 'Image dimensions must be divisible by the patch size.'

        patch_height_dim, patch_width_dim = (image_height // patch_size), (image_width // patch_size)
        patch_dim = channels * (patch_size ** 2)

        self.channels = channels
        self.patch_size = patch_size

        self.patch_norm = PatchNorm(patch_height_dim, patch_width_dim, patch_dim)

        self.to_patch_embedding = nn.Sequential(
            nn.Linear(patch_dim, dim),
            LayerNorm(dim),
        )

        pos_dim = patch_dim if pos_embed_before_proj else dim
        self.pos_embed_before_proj = pos_embed_before_proj

        self.pos_embed_height = nn.Parameter(torch.zeros(patch_height_dim, pos_dim))
        self.pos_embed_width = nn.Parameter(torch.zeros(patch_width_dim, pos_dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        batched_images: Union[List[Tensor], List[List[Tensor]]], # assume different resolution images already grouped correctly
        group_images = False,
        group_max_seq_len = 2048
    ):
        p, c, device, has_token_dropout = self.patch_size, self.channels, self.device, exists(self.calc_token_dropout)

        arange = partial(torch.arange, device = device)
        pad_sequence = partial(orig_pad_sequence, batch_first = True)

        # auto pack if specified

        if group_images:
            batched_images = group_images_by_max_seq_len(
                batched_images,
                patch_size = self.patch_size,
                calc_token_dropout = self.calc_token_dropout,
                max_seq_len = group_max_seq_len
            )

        # process images into variable lengthed sequences with attention mask

        num_images = []
        batched_sequences = []
        batched_positions = []
        batched_image_ids = []

        for images in batched_images:
            num_images.append(len(images))

            sequences = []
            positions = []
            image_ids = torch.empty((0,), device = device, dtype = torch.long)
            image_dimensions = []

            for image_id, image in enumerate(images):
                assert image.ndim ==3 and image.shape[0] == c
                image_dims = image.shape[-2:]
                assert all([divisible_by(dim, p) for dim in image_dims]), f'height and width {image_dims} of images must be divisible by patch size {p}'

                ph, pw = map(lambda dim: dim // p, image_dims)

                pos = torch.stack(torch.meshgrid((
                    arange(ph),
                    arange(pw)
                ), indexing = 'ij'), dim = -1)

                pos = rearrange(pos, 'h w c -> (h w) c')
                seq = rearrange(image, 'c (h p1) (w p2) -> (h w) (c p1 p2)', p1 = p, p2 = p)

                seq_len = seq.shape[-2]

                if has_token_dropout:
                    token_dropout = self.calc_token_dropout(*image_dims)
                    num_keep = max(1, int(seq_len * (1 - token_dropout)))
                    keep_indices = torch.randn((seq_len,), device = device).topk(num_keep, dim = -1).indices

                    seq = seq[keep_indices]
                    pos = pos[keep_indices]

                image_ids = F.pad(image_ids, (0, seq.shape[-2]), value = image_id)
                sequences.append(seq)
                positions.append(pos)
                image_dimensions.append(torch.Tensor(image_dims))

            batched_image_ids.append(image_ids)
            batched_sequences.append(torch.cat(sequences, dim = 0))
            batched_positions.append(torch.cat(positions, dim = 0))

        # derive key padding mask

        lengths = torch.tensor([seq.shape[-2] for seq in batched_sequences], device = device, dtype = torch.long)
        max_length = arange(lengths.amax().item())
        key_pad_mask = rearrange(lengths, 'b -> b 1') <= rearrange(max_length, 'n -> 1 n')

        # derive attention mask, and combine with key padding mask from above

        batched_image_ids = pad_sequence(batched_image_ids)
        attn_mask = rearrange(batched_image_ids, 'b i -> b 1 i 1') == rearrange(batched_image_ids, 'b j -> b 1 1 j')
        attn_mask = attn_mask & rearrange(key_pad_mask, 'b j -> b 1 1 j')

        # combine patched images as well as the patched width / height positions for 2d positional embedding

        patches = pad_sequence(batched_sequences)

        patch_positions = pad_sequence(batched_positions)

        # need to know how many images for final attention pooling

        num_images = torch.tensor(num_images, device = device, dtype = torch.long)        

        # factorized 2d absolute positional embedding

        h_indices, w_indices = patch_positions.unbind(dim = -1)

        # normalizes patches
        patches = self.patch_norm(patches, h_indices, w_indices, key_pad_mask)

        # possibly adds pos embedding info before projecting in
        if self.pos_embed_before_proj:
            h_pos = self.pos_embed_height[h_indices]
            w_pos = self.pos_embed_width[w_indices]
            x = patches + h_pos + w_pos
            x = self.to_patch_embedding(x)        
        else:
            x = self.to_patch_embedding(patches)        
            h_pos = self.pos_embed_height[h_indices]
            w_pos = self.pos_embed_width[w_indices]
            x = x + h_pos + w_pos

        # embed dropout

        x = self.dropout(x)

        # attention

        x = self.transformer(x, attn_mask = attn_mask)

        b, s, _ = x.shape


        def revert_patching(y: torch.Tensor):
            """
            This function takes a y, which has the same leading shape as x,
            and unpatches it. A list of images is returned, where at each spacial
            position in the image contains a zero vector (if the token was randomly dropped)
            or the vector from y corresponding to that image and that spacial position.
            """
            yb, ys, z = y.shape
            assert yb == b
            assert ys == s

            # doesnt work with token dropout
            # because then h,w might be incorrect
            assert not has_token_dropout

            images = []

            for batch_i, (image_ids, mask, positions) in enumerate(zip(batched_image_ids, key_pad_mask, patch_positions)):
                # take the tokens that have actual images associated with them
                for image_id in image_ids.unique():
                    image_mask = (image_ids == image_id) & ~mask
                    image_tokens = y[batch_i, image_mask, :]
                    image_positions = positions[image_mask]
                    h, w = image_positions.max(dim=0).values + 1
                    image = rearrange(image_tokens, '(h w) (c p1 p2) -> c (h p1) (w p2)', p1 = p, p2=p, w=w, h=h)
                    images.append(image)

            return images
                    

        return dict(x=x, revert_patching=revert_patching, key_pad_mask=key_pad_mask, patches=patches, h_indices=h_indices, w_indices=w_indices)
