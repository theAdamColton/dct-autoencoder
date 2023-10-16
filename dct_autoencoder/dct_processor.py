"""
-------- Preprocessing pipeline -----------

image Shape: (c, h, w)

| dct2d
v

image dct Shape: (c, h, w)

| rearrange
v

image of dct patches Shape: (ph, pw, pz), where pz = (c * patchsize ** 2)

| distance threshold, 
| take the top p closest patches
| to the top left corner
v

flattened image Shape: (s, pz) where s = int(ph * pw * p)
patch positions

|||||||| collate patches into batches by padding and packing
vvvvvvvv

patches Shape: (b, s, pz)
original patch positions: (b, s, 2)
image indices: (b, s)

|||||||| normalize patches based on patch positions
vvvvvvvv

normalized patches Shape: (b, s, pz)
original patch positions: (b, s, 2)
image indices: (b, s)




-------- Postprocessing pipeline ----------


normalized patches Shape: (b, s, pz)
original patch positions: (b, s, 2)
image indices: (b, s)

|||||||| de normalize patches based on patch positions
vvvvvvvv

patches Shape: (b, s, pz)
original patch positions: (b, s, 2)
image indices: (b, s)

| Un-collate patches by placing patches into a zeroed tensor
| using their stored patch positions and image indices
v

2d image of dct patches Shape: (ph, pw, pz)

| 2d dct image Shape: (c, h, w)
v

| idct2
v

2d image Shape: (c, h, w)


"""
from dataclasses import dataclass
from typing import Callable, List, Tuple
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
    # ph, pw of the patches
    patch_sizes: List[Tuple]
    # h,w of the original image pixels
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
        sample_patches: Callable[[], float],
        max_n_patches: int,
        max_seq_len: int,
        max_batch_size:int,
        patch_norm_device="cpu",
        token_dropout_prob=None,
    ):
        self.channels = channels
        self.patch_size = patch_size
        self.sample_patches = sample_patches
        self.max_n_patches = max_n_patches
        self.max_res = patch_size * max_n_patches
        self.max_seq_len = max_seq_len
        self.patch_norm = PatchNorm(
            max_n_patches, max_n_patches, patch_size**2 * channels
        ).to(patch_norm_device)
        self.max_batch_size = max_batch_size

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
    def _transform_image_in(self, x):
        """
        x is a image tensor, with values between 0.0 and 1.0
        """
        og_dtype = x.dtype
        return dct2(x.to(torch.float32), 'ortho').to(og_dtype)

    @torch.no_grad()
    def _transform_image_out(self, x):
        og_dtype = x.dtype
        image = idct2(x.to(torch.float32), 'ortho').to(og_dtype)
        # TODO clamp
        image = image.clamp(0.0, 1.0)
        return image
            

    @torch.no_grad()
    def preprocess(self, x: List[torch.Tensor]):
        """
        preprocess but don't batch
        """
        cropped_ims = []
        patch_sizes = []
        original_sizes = []
        for im in x:
            im = self._transform_image_in(im)
            c,h,w = im.shape
            original_sizes.append((h,w))

            cropped_im = self._crop_image(im)
            c,ch,cw = cropped_im.shape

            ph, pw = ch // self.patch_size, cw//self.patch_size
            patch_sizes.append((ph, pw))
            cropped_ims.append(cropped_im)

        patched_ims = []
        positions = []
        for cropped_im in cropped_ims:
            patches, pos = self._patch_image(cropped_im)
            patched_ims.append(patches)
            positions.append(pos)

        return patched_ims, positions, original_sizes, patch_sizes

    @torch.no_grad()
    def batch(self, patches: List[torch.Tensor], positions: List[torch.Tensor], original_sizes: List[Tuple[int,int]], patch_sizes: List[Tuple[int, int]], device='cpu') -> DCTPatches:
        """
        batch the result of preprocess
        """
        patches, positions = self._group_patches_by_max_seq_len(patches, positions)
        return self._batch_groups(patches, positions, original_sizes=original_sizes, patch_sizes=patch_sizes, device=device)


    @torch.no_grad()
    def postprocess(self, x: DCTPatches) -> List[torch.Tensor]:
        """
        return a list of images
        """
        dct_images = self.revert_patching(x)
        images = []
        for image, (h, w) in zip(dct_images, x.original_sizes):
            ch, cw = image.shape[-2:]

            im_pad = torch.zeros(self.channels, h, w, device=image.device, dtype=image.dtype)

            im_pad[:, :ch, :cw] = image

            im_pad = self._transform_image_out(im_pad)

            images.append(im_pad)

        return images

    def _get_crop_dims(self, h:int, w:int):
        assert h >= self.patch_size
        assert w >= self.patch_size

        # p_h, p_w, number of patches in height and width
        p_h = int(h / self.patch_size)
        p_w = int(w / self.patch_size)

        # crop height and crop width
        c_h = p_h * self.patch_size
        c_w = p_w * self.patch_size

        assert c_h % self.patch_size == 0
        assert c_w % self.patch_size == 0
        assert c_h <= h
        assert c_w <= w
        assert c_h >= self.patch_size
        assert c_w >= self.patch_size

        return c_h, c_w


    @torch.no_grad()
    def _crop_image(self, x: torch.Tensor) -> torch.Tensor:
        """
        not batchable

        takes an input image x,
        and crops it such that the height and width are both divisible by patch
        size
        """
        c, h, w = x.shape

        assert c == self.channels
        c_h, c_w = self._get_crop_dims(h,w)
        x = x[:, :c_h, :c_w]

        return x

    @torch.no_grad()
    def _patch_image(self, x: torch.Tensor):
        _, h, w = x.shape

        assert h % self.patch_size == 0
        assert w % self.patch_size == 0

        ph, pw = h//self.patch_size, w//self.patch_size

        h_indices, w_indices = torch.meshgrid(torch.arange(ph, device=x.device), torch.arange(pw, device=x.device), indexing='ij')

        # distances from upper left corner
        tri_distances = (h_indices + w_indices) * 1.0

        # takes the top p closest distances
        p = min(max(self.sample_patches(), 0.01), 1.0)
        _, indices_flat = tri_distances.view(-1).sort()
        k = round(len(indices_flat) * p)
        k = max(1, k)
        k = min(len(indices_flat), k)
        k = min(k, self.max_n_patches)
        indices_flat = indices_flat[:k]

        # patches x into a list of patches
        x = rearrange(x, "c (h p1) (w p2) -> (h w) (c p1 p2)", p1=self.patch_size, p2=self.patch_size, c = self.channels)


        # takes top k patches, and their indices
        x = x[indices_flat,:]
        h_indices = h_indices.flatten()[indices_flat]
        w_indices = w_indices.flatten()[indices_flat]

        s, z = x.shape
        assert z == self.channels * self.patch_size ** 2, f"{z} != {self.channels * self.patch_size ** 2}"
        assert s == k

        # x is shape c, k
        # h_indices is shape k
        # w_indices is shape k
        pos = torch.stack([h_indices, w_indices], dim=-1)
        return x, pos 

    @torch.no_grad()
    def _group_patches_by_max_seq_len(
            self,
            batched_patches: List[Tensor],
            batched_positions: List[Tensor],
            ) -> Tuple[List[List[Tensor]], List[List[Tensor]]]:
        """
        converts a list of ungrouped flattened patches
        into batched groups, such that the total number of batches is less than
        self.max_batch_size and the length of any sequence of the batches is less than
        self.max_seq_len.
        """
        groups = []
        groups_pos = []
        group = []
        group_pos = []
        seq_len = 0


        for i, (patches, pos) in enumerate(zip(batched_patches, batched_positions)):
            assert isinstance(patches, Tensor)

            k, _ = patches.shape

            assert (
            k <= self.max_n_patches and k <= self.max_seq_len
            ), f"patch with len {k} exceeds maximum sequence length"

            if (seq_len + k) > self.max_seq_len:
                groups.append(group)
                groups_pos.append(group_pos)
                group = []
                group_pos = []
                seq_len = 0

            group.append(patches)
            group_pos.append(pos)
            seq_len += k

            if len(groups) >= self.max_batch_size:
                print(f"Warning! Truncating {len(batched_patches) - i + 1} images from batch to reach batch size of {self.max_batch_size}")

        if len(group) > 0:
            groups.append(group)
            groups_pos.append(group_pos)

        return groups, groups_pos

    @torch.no_grad()
    def _batch_groups(self,
               grouped_batched_patches: List[List[Tensor]],
               grouped_batched_positions: List[List[Tensor]],
               device=torch.device('cpu'),
               **dct_patch_kwargs,
                      ) -> DCTPatches:
        """
        returns a batch of DCTPatches
        """
        arange = partial(torch.arange, device=device)
        pad_sequence = partial(orig_pad_sequence, batch_first=True)

        assert len(grouped_batched_positions) == len(grouped_batched_patches)

        # process patches into variable lengthed sequences with attention mask

        num_images = []
        batched_sequences = []
        batched_positions = []
        batched_image_ids = []

        for grouped_patches, grouped_positions in zip(grouped_batched_patches, grouped_batched_positions):
            assert len(grouped_patches) == len(grouped_positions)
            num_images.append(len(grouped_patches))

            image_ids = torch.empty((0,), device=device, dtype=torch.long)

            for image_id, (patches, positions) in enumerate(zip(grouped_patches, grouped_positions)):
                k, z = patches.shape

                assert z == self.channels * self.patch_size ** 2, f"{z} != {self.channels * self.patch_size ** 2}"

                assert positions.shape[0] == k

                image_ids = F.pad(image_ids, (0, k), value=image_id)

            batched_image_ids.append(image_ids)
            batched_sequences.append(torch.cat(grouped_patches, dim=0))
            batched_positions.append(torch.cat(grouped_positions, dim=0))

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

        patches = pad_sequence(batched_sequences).to(device)

        patch_positions = pad_sequence(batched_positions).to(device)

        # need to know how many images for final attention pooling

        num_images = torch.tensor(num_images, device=device, dtype=torch.long)

        h_indices, w_indices = patch_positions.unbind(dim=-1)

        return DCTPatches(
            patches=patches,
            key_pad_mask=key_pad_mask,
            h_indices=h_indices,
            w_indices=w_indices,
            attn_mask=attn_mask,
            batched_image_ids=batched_image_ids,
            patch_positions=patch_positions,
            **dct_patch_kwargs,
        )


    def revert_patching(self, output: DCTPatches) -> List[torch.Tensor]:
        """
        This function takes a y, which has the same leading shape as x,
        and unpatches it. A list of images is returned, where at each spacial
        position in the image contains a zero vector (if the token was randomly dropped)
        or the vector from y corresponding to that image and that spacial position.
        """
        x = output.patches
        z = x.shape[-1]

        images = []

        for batch_i, (image_ids, mask, positions) in enumerate(
            zip(output.batched_image_ids, output.key_pad_mask, output.patch_positions,)
        ):
            # take the tokens that have actual images associated with them
            for image_id in image_ids.unique():
                image_mask = (image_ids == image_id) & ~mask
                image_tokens = x[batch_i, image_mask, :]
                image_positions = positions[image_mask]
                ph, pw = output.patch_sizes[len(images)]

                image = torch.zeros(ph, pw, z, dtype=x.dtype, device=x.device)

                for token, pos in zip(image_tokens, image_positions):
                    pos_h,pos_w = pos.unbind(-1)
                    image[pos_h, pos_w, :] = token 

                image = rearrange(image.view(ph*pw, -1), '(ph pw) (c p1 p2) -> c (ph p1) (pw p2)', p1=self.patch_size, p2=self.patch_size, c=self.channels, ph=ph, pw=pw)
                images.append(image)

        return images
