"""
thanks lucidrains from vit_pytorch
"""

import torch
import torch.nn.functional as F
from torch import nn

from einops import rearrange

from .dct_processor import DCTPatches
from .util import default, exists, divisible_by

# helpers


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# auto grouping images


# normalization
# they use layernorm without bias, something that pytorch does not offer


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


# they use a query-key normalization that is equivalent to rms norm (no mean-centering, learned gamma), from vit 22B paper


class RMSNorm(nn.Module):
    def __init__(self, heads, dim):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(heads, 1, dim))

    def forward(self, x):
        normed = F.normalize(x, dim=-1)
        return normed * self.scale * self.gamma


# feedforward


def FeedForward(dim, hidden_dim, dropout=0.0):
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, dim),
        nn.Dropout(dropout),
    )


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.norm = LayerNorm(dim)

        self.q_norm = RMSNorm(heads, dim_head)
        self.k_norm = RMSNorm(heads, dim_head)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False), nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None, attn_mask=None):
        x = self.norm(x)
        kv_input = default(context, x)

        qkv = (self.to_q(x), *self.to_kv(kv_input).chunk(2, dim=-1))

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        q = self.q_norm(q)
        k = self.k_norm(k)

        dots = torch.matmul(q, k.transpose(-1, -2))

        if exists(mask):
            mask = rearrange(mask, "b j -> b 1 1 j")
            dots = dots.masked_fill(~mask, -torch.finfo(dots.dtype).max)

        if exists(attn_mask):
            dots = dots.masked_fill(~attn_mask, -torch.finfo(dots.dtype).max)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
            )

        self.norm = LayerNorm(dim)

    def forward(self, x, mask=None, attn_mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask, attn_mask=attn_mask) + x
            x = ff(x) + x

        return self.norm(x)


class NaViT(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        channels: int = 3,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        pos_embed_before_proj: bool = True,
    ):
        super().__init__()
        image_height, image_width = pair(image_size)

        # calculate patching related stuff

        assert divisible_by(image_height, patch_size) and divisible_by(
            image_width, patch_size
        ), "Image dimensions must be divisible by the patch size."

        patch_height_dim, patch_width_dim = (image_height // patch_size), (
            image_width // patch_size
        )
        patch_dim = channels * (patch_size**2)

        self.channels = channels
        self.patch_size = patch_size

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

    def forward(self, _input: DCTPatches) -> DCTPatches:
        patches, h_indices, w_indices, attn_mask = (
            _input.patches,
            _input.h_indices,
            _input.w_indices,
            _input.attn_mask,
        )

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

        x = self.transformer(x, attn_mask=attn_mask)

        return DCTPatches(
            patches=x,
            original_sizes=_input.original_sizes,
            attn_mask=attn_mask,
            batched_image_ids=_input.batched_image_ids,
            patch_positions=_input.patch_positions,
            key_pad_mask=_input.key_pad_mask,
            h_indices=h_indices,
            w_indices=w_indices,
            has_token_dropout=_input.has_token_dropout,
        )
