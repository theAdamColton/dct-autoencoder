from dataclasses import dataclass
from typing import List, Tuple
from functools import partial
from einops import rearrange
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence as orig_pad_sequence

from .norm import PatchNorm
from .util import (
    dct2,
    always,
    exists,
    default,
    idct2,
    divisible_by,
)


@dataclass
class DCTPatches:
    patches: torch.Tensor
    key_pad_mask: torch.BoolTensor
    h_indices: torch.LongTensor
    w_indices: torch.LongTensor
    attn_mask: torch.BoolTensor
    batched_image_ids: torch.LongTensor
    patch_positions: torch.LongTensor
    has_token_dropout: bool
    original_sizes: List[Tuple]

    def to(self, what):
        self.patches = self.patches.to(what)
        self.key_pad_mask = self.key_pad_mask.to(what)
        self.h_indices = self.h_indices.to(what)
        self.w_indices = self.w_indices.to(what)
        self.attn_mask = self.attn_mask.to(what)
        self.batched_image_ids = self.batched_image_ids.to(what)
        self.patch_positions = self.patch_positions.to(what)
        return self


class DCTProcessor:
    def __init__(
        self,
        channels: int,
        patch_size: int,
        dct_compression_factor: float,
        max_n_patches: int,
        max_seq_len: int,
        patch_norm_device="cpu",
        token_dropout_prob=None,
    ):
        self.channels = channels
        self.patch_size = patch_size
        self.dct_compression_factor = dct_compression_factor
        self.max_n_patches = max_n_patches
        self.max_res = patch_size * max_n_patches
        self.max_seq_len = max_seq_len
        self.patch_norm = PatchNorm(
            patch_size, patch_size, patch_size**2 * channels
        ).to(patch_norm_device)

        # what percent of tokens to dropout
        # if int or float given, then assume constant dropout prob
        # otherwise accept a callback that in turn calculates dropout prob from height and width
        self.calc_token_dropout = None

        if callable(token_dropout_prob):
            self.calc_token_dropout = token_dropout_prob

        elif isinstance(token_dropout_prob, (float, int)):
            assert 0.0 < token_dropout_prob < 1.0
            token_dropout_prob = float(token_dropout_prob)
            self.calc_token_dropout = lambda height, width: token_dropout_prob

    @torch.no_grad()
    def __call__(self, x):
        return self.preprocess(x)

    @torch.no_grad()
    def preprocess(self, x: List[torch.Tensor]) -> DCTPatches:
        dct_images = []
        all_original_sizes = []

        for im in x:
            im_dct, original_sizes = self.image_to_dct(im)
            dct_images.append(im_dct)
            all_original_sizes.append(original_sizes)

        dct_patches = self.patch_images(dct_images)
        dct_patches.original_sizes = all_original_sizes
        dct_patches.patches = self.patch_norm(
            dct_patches.patches, dct_patches.h_indices, dct_patches.w_indices
        )
        return dct_patches

    @torch.no_grad()
    def postprocess(self, x: DCTPatches) -> List[torch.Tensor]:
        x.patches = self.patch_norm.inverse_norm(x.patches, x.h_indices, x.w_indices)
        dct_images = self.revert_patching(x)

        if x.original_sizes is None:
            original_sizes = [im.shape[-2:] for im in dct_images]
        else:
            original_sizes = x.original_sizes

        ims = []
        for im, (h, w) in zip(dct_images, original_sizes):
            c, ih, iw = im.shape
            padded = torch.zeros(c, h, w, device=im.device, dtype=im.dtype)
            padded[:, :ih, :iw] = im
            im = padded

            im = idct2(im.detach().float().cpu(), "ortho")
            ims.append(im)
        return ims

    @torch.no_grad()
    def image_to_dct(self, x: torch.Tensor):
        """
        not batchable
        """
        # TODO take features that are closer to 0,0
        # rather than a rectangle
        c, h, w = x.shape

        assert c == self.channels
        assert h >= self.patch_size
        assert w >= self.patch_size

        x_dct = dct2(x, "ortho")

        h_c, w_c = (1 - self.dct_compression_factor) * h, (
            1 - self.dct_compression_factor
        ) * w

        p_h = round(h_c / self.patch_size)
        p_w = round(w_c / self.patch_size)
        p_h = max(p_h, 1)
        p_w = max(p_w, 1)

        # we need that
        # ar = p_h_c / p_w_c = p_h / p_w
        # p_h_c * p_w_c <= self.max_n_patches
        # ar * p_w_c <= self.max_n_patches / p_w_c
        # ar * p_w_c ** 2 <= self.max_n_patches
        # p_w_c = sqrt(self.max_n_patches / ar)
        # p_h_c = ar * p_w_c

        ar = p_h / p_w
        p_w_c = int((self.max_n_patches / ar) ** 0.5)
        p_h_c = int(ar * p_w_c)

        assert p_h_c * p_w_c <= self.max_n_patches

        p_h = min(p_h, p_h_c)
        p_w = min(p_w, p_w_c)
        p_h = max(p_h, 1)
        p_w = max(p_w, 1)

        dct_h = p_h * self.patch_size
        dct_w = p_w * self.patch_size

        assert dct_h % self.patch_size == 0
        assert dct_w % self.patch_size == 0
        assert dct_h <= self.max_res
        assert dct_w <= self.max_res
        assert dct_h >= self.patch_size
        assert dct_w >= self.patch_size

        x_dct = x_dct[..., :dct_h, :dct_w]

        return x_dct, (h, w)

    def group_images_by_max_seq_len(
        self,
        images: List[Tensor],
    ) -> List[List[Tensor]]:
        calc_token_dropout = default(self.calc_token_dropout, always(0.0))

        groups = []
        group = []
        seq_len = 0

        if isinstance(calc_token_dropout, (float, int)):
            calc_token_dropout = always(calc_token_dropout)

        for image in images:
            assert isinstance(image, Tensor)

            image_dims = image.shape[-2:]
            ph, pw = map(lambda t: t // self.patch_size, image_dims)

            image_seq_len = ph * pw
            image_seq_len = int(image_seq_len * (1 - calc_token_dropout(*image_dims)))

            assert (
                image_seq_len <= self.max_seq_len
            ), f"image with dimensions {image_dims} exceeds maximum sequence length"

            if (seq_len + image_seq_len) > self.max_seq_len:
                groups.append(group)
                group = []
                seq_len = 0

            group.append(image)
            seq_len += image_seq_len

        if len(group) > 0:
            groups.append(group)

        return groups

    @torch.no_grad()
    def patch_images(
        self,
        batched_images: List[Tensor],
        original_sizes=None,
    ) -> DCTPatches:
        p, c, device, has_token_dropout = (
            self.patch_size,
            self.channels,
            torch.device("cpu"),
            exists(self.calc_token_dropout),
        )

        arange = partial(torch.arange, device=device)
        pad_sequence = partial(orig_pad_sequence, batch_first=True)

        # auto pack if specified

        batched_images = self.group_images_by_max_seq_len(batched_images)

        # process images into variable lengthed sequences with attention mask

        num_images = []
        batched_sequences = []
        batched_positions = []
        batched_image_ids = []

        for images in batched_images:
            num_images.append(len(images))

            sequences = []
            positions = []
            image_ids = torch.empty((0,), device=device, dtype=torch.long)
            image_dimensions = []

            for image_id, image in enumerate(images):
                assert image.ndim == 3 and image.shape[0] == c
                image_dims = image.shape[-2:]
                assert all(
                    [divisible_by(dim, p) for dim in image_dims]
                ), f"height and width {image_dims} of images must be divisible by patch size {p}"

                ph, pw = map(lambda dim: dim // p, image_dims)

                pos = torch.stack(
                    torch.meshgrid((arange(ph), arange(pw)), indexing="ij"), dim=-1
                )

                pos = rearrange(pos, "h w c -> (h w) c")
                seq = rearrange(image, "c (h p1) (w p2) -> (h w) (c p1 p2)", p1=p, p2=p)

                seq_len = seq.shape[-2]

                if has_token_dropout:
                    token_dropout = self.calc_token_dropout(*image_dims)
                    num_keep = max(1, int(seq_len * (1 - token_dropout)))
                    keep_indices = (
                        torch.randn((seq_len,), device=device)
                        .topk(num_keep, dim=-1)
                        .indices
                    )

                    seq = seq[keep_indices]
                    pos = pos[keep_indices]

                image_ids = F.pad(image_ids, (0, seq.shape[-2]), value=image_id)
                sequences.append(seq)
                positions.append(pos)
                image_dimensions.append(torch.Tensor(image_dims))

            batched_image_ids.append(image_ids)
            batched_sequences.append(torch.cat(sequences, dim=0))
            batched_positions.append(torch.cat(positions, dim=0))

        # derive key padding mask

        lengths = torch.tensor(
            [seq.shape[-2] for seq in batched_sequences],
            device=device,
            dtype=torch.long,
        )
        max_length = arange(lengths.amax().item())
        key_pad_mask = rearrange(lengths, "b -> b 1") <= rearrange(
            max_length, "n -> 1 n"
        )

        # derive attention mask, and combine with key padding mask from above

        batched_image_ids = pad_sequence(batched_image_ids)
        attn_mask = rearrange(batched_image_ids, "b i -> b 1 i 1") == rearrange(
            batched_image_ids, "b j -> b 1 1 j"
        )
        attn_mask = attn_mask & rearrange(key_pad_mask, "b j -> b 1 1 j")

        # combine patched images as well as the patched width / height positions for 2d positional embedding

        patches = pad_sequence(batched_sequences)

        patch_positions = pad_sequence(batched_positions)

        # need to know how many images for final attention pooling

        num_images = torch.tensor(num_images, device=device, dtype=torch.long)

        # factorized 2d absolute positional embedding

        h_indices, w_indices = patch_positions.unbind(dim=-1)

        return DCTPatches(
            patches=patches,
            key_pad_mask=key_pad_mask,
            h_indices=h_indices,
            w_indices=w_indices,
            attn_mask=attn_mask,
            batched_image_ids=batched_image_ids,
            patch_positions=patch_positions,
            has_token_dropout=has_token_dropout,
            original_sizes=original_sizes,
        )

    def revert_patching(self, output: DCTPatches) -> List[torch.Tensor]:
        """
        This function takes a y, which has the same leading shape as x,
        and unpatches it. A list of images is returned, where at each spacial
        position in the image contains a zero vector (if the token was randomly dropped)
        or the vector from y corresponding to that image and that spacial position.
        """
        x = output.patches

        # doesnt work with token dropout
        # because then h,w might be incorrect
        assert not output.has_token_dropout

        images = []

        for batch_i, (image_ids, mask, positions) in enumerate(
            zip(output.batched_image_ids, output.key_pad_mask, output.patch_positions)
        ):
            # take the tokens that have actual images associated with them
            for image_id in image_ids.unique():
                image_mask = (image_ids == image_id) & ~mask
                image_tokens = x[batch_i, image_mask, :]
                image_positions = positions[image_mask]
                h, w = image_positions.max(dim=0).values + 1
                image = rearrange(
                    image_tokens,
                    "(h w) (c p1 p2) -> c (h p1) (w p2)",
                    p1=self.patch_size,
                    p2=self.patch_size,
                    w=w,
                    h=h,
                )
                images.append(image)

        return images
