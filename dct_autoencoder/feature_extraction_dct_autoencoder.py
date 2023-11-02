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
| take the top p closest patches to the top left corner, mixed with the top
| magnitude frequencies
| These positions need to be in-bounds with respect to the model's max patch h
| and max patch w.
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
from typing import List, Optional, Tuple
from functools import partial
from einops import rearrange
import torch
from torch import Tensor
import torch.nn.functional as F
from transformers.feature_extraction_utils import FeatureExtractionMixin

from .dct_patches import DCTPatches
from .util import (
    dct2,
    exp_dist,
    idct2,
    pad_sequence,
    ipt_to_rgb,
    rgb_to_ipt,
)


@dataclass
class GroupPatchesState:
    groups: List[List[Tensor]]
    groups_pos: List[List[Tensor]]
    group: List[Tensor]
    group_pos: List[Tensor]
    groups_channels: List[List[Tensor]]
    group_channels: List[Tensor]
    seq_len: int

class DCTAutoencoderFeatureExtractor(FeatureExtractionMixin):
    def __init__(
        self,
        channels: int,
        patch_size: int,
        sample_patches_beta: float,
        max_patch_h: int,
        max_patch_w: int,
        max_seq_len: int,
        channel_importances:Tuple[float,float,float] = (8,1,1),
    ):
        self.channels = channels
        self.patch_size = patch_size
        self.sample_patches_beta = sample_patches_beta
        self.max_patch_h = max_patch_h
        self.max_patch_w = max_patch_w
        self.max_seq_len = max_seq_len
        self.channel_importances = torch.Tensor(channel_importances)
        self.channel_importances = self.channel_importances / self.channel_importances.sum()


    @torch.no_grad()
    def _transform_image_in(self, x):
        """
        x is a image tensor containing rgb colors, with values between 0.0
        and 1.0
        """
        x = rgb_to_ipt(x)
        og_dtype = x.dtype
        return dct2(x.to(torch.float32), 'ortho').to(og_dtype)

    def _transform_image_out(self, x):
        og_dtype = x.dtype
        image = idct2(x.to(torch.float32), 'ortho').to(og_dtype)
        image = ipt_to_rgb(image)
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
            _,h,w = im.shape
            original_sizes.append((h,w))

            cropped_im = self._crop_image(im)
            _,ch,cw = cropped_im.shape

            ph, pw = ch // self.patch_size, cw//self.patch_size
            patch_sizes.append((ph, pw))
            cropped_ims.append(cropped_im)

        patched_ims = []
        positions = []
        all_channels = []
        for cropped_im in cropped_ims:
            patches, pos, channels = self._patch_image(cropped_im)
            patched_ims.append(patches)
            positions.append(pos)
            all_channels.append(channels)

        return patched_ims, positions, all_channels, original_sizes, patch_sizes

    @torch.no_grad()
    def iter_batches(self, dataloader, batch_size: Optional[int] = None):
        """
        dataloader: iterable, returns [patches, positions, channels, original_sizes, patch_sizes]
        """

        state = None
        cum_original_sizes = []
        cum_patch_sizes = []
        while True:
            patches, positions, channels, original_sizes, patch_sizes = next(dataloader)

            # keeps on cpu, as to not waste gpu space
            patches = [p.to('cpu') for p in patches]
            positions = [p.to('cpu') for p in positions]
            channels = [c.to('cpu') for c in channels]

            cum_original_sizes = cum_original_sizes + original_sizes
            cum_patch_sizes = cum_patch_sizes + patch_sizes

            state = self._group_patches_by_max_seq_len(patches, positions, channels, state)

            if batch_size is None:
                # immediately output a batch for this state
                if len(state.group) > 0:
                    state.groups.append(state.group)
                    state.groups_pos.append(state.group_pos)
                    state.groups_channels.append(state.group_channels)
                    state.seq_len = 0
                    state.group = []
                    state.group_pos = []
                    state.group_channels = []

            cur_batch_size = len(state.groups)

            if batch_size is None or cur_batch_size > batch_size:
                # clips state
                new_state = GroupPatchesState(
                        groups=state.groups[batch_size:],
                        groups_pos = state.groups_pos[batch_size:],
                        group=state.group,
                        group_pos=state.group_pos,
                        groups_channels=state.groups_channels[batch_size:],
                        group_channels=state.group_channels,
                        seq_len=state.seq_len,
                )
                state = GroupPatchesState(
                        groups=state.groups[:batch_size],
                        groups_pos = state.groups_pos[:batch_size],
                        groups_channels=state.groups_channels[:batch_size],
                        group_channels=[],
                        group=[],
                        group_pos=[],
                        seq_len=0,
                        )

                n_items_in_batch = sum(len(x) for x in state.groups)

                new_original_sizes = cum_original_sizes[n_items_in_batch:]
                new_patch_sizes = cum_patch_sizes[n_items_in_batch:]
                cum_original_sizes = cum_original_sizes[:n_items_in_batch]
                cum_patch_sizes = cum_patch_sizes[:n_items_in_batch]

                batch = self._batch_groups(state.groups, state.groups_pos, state.groups_channels, original_sizes = cum_original_sizes, patch_sizes = cum_patch_sizes)

                if batch_size is not None:
                    assert batch.patches.shape[0] == batch_size
                    assert batch.key_pad_mask.shape[0] == batch_size
                    assert batch.batched_image_ids.shape[0] == batch_size
                    assert batch.patch_positions.shape[0] == batch_size
                    assert batch.attn_mask.shape[0] == batch_size
                    assert batch.patch_channels.shape[0] == batch_size

                cum_patch_sizes = new_patch_sizes
                cum_original_sizes = new_original_sizes
                state = new_state

                yield batch


    def postprocess(self, x: DCTPatches) -> List[torch.Tensor]:
        """
        x: DCTPatches that are not normalized

        return a list of image rgb values
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


#        ar = h/w
#
#        if p_h > self.max_patch_h:
#            p_h = self.max_patch_h
#            p_w = int(p_h / ar)
#        if p_w > self.max_patch_w:
#            p_w = self.max_patch_w
#            p_h = int(ar * p_w)
#
#        p_h = min(p_h, self.max_patch_h)
#        p_w = min(p_w, self.max_patch_w)
        p_h = max(p_h, 1)
        p_w = max(p_w, 1)

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
        c, h, w = x.shape

        assert h % self.patch_size == 0
        assert w % self.patch_size == 0

        ph, pw = h//self.patch_size, w//self.patch_size

        # patches x into a list of patches
        x = rearrange(x, "c (h p1) (w p2) -> (h w) c (p1 p2)", p1=self.patch_size, p2=self.patch_size, c = self.channels)

        h_indices, w_indices = torch.meshgrid(torch.arange(ph, device=x.device), torch.arange(pw, device=x.device), indexing='ij')


        # makes it so that ph and pw that are out of bounds can't be selected
        _mask = torch.logical_and(h_indices < self.max_patch_h, w_indices < self.max_patch_w)
        x = x[_mask.flatten()]

        h_indices = h_indices[_mask]
        w_indices = w_indices[_mask]

        # s c
        mags = x.abs().amax(-1)

        # distances from upper left corner
        distance_weight = 0.5
        distances = ((h_indices + w_indices) * - distance_weight).flatten()

        # sorts by magnitude and distance
        _, per_channel_sorted_idx = (mags + distances.unsqueeze(-1)).sort(dim=0, descending=True)


        # k is the full length of dct patches needed to fully (losslessly)
        # represent x

        # k gets downsampled in two ways:
        # k can't be larger than the max number of patches or the current max
        # sequence length

        # also if sample_patches_beta is set, then k will be sampled from the
        # exponential distribution
        k = ph * pw
        k = min(k, self.max_patch_h * self.max_patch_w)
        k = min(k, self.max_seq_len)

        if self.sample_patches_beta > 0.00:
            # samples from the exponential distribution to get k
            k = min(round(exp_dist(self.sample_patches_beta)), k)
            # at least the base channels are needed
            k = max(c, k)

        # now splits up the k patches between the channels
        # each channel gets it's own patches
        channel_k = (self.channel_importances * k).round().long()
        # there should be at least one patch per channel
        channel_k = channel_k.clamp(1)

        # handles the remainder
        _total_channel_k = channel_k.sum()
        channel_k[0] = channel_k[0] - (_total_channel_k - k)
        assert channel_k.sum() == k

        h_indices_f = h_indices.reshape(-1)
        w_indices_f = w_indices.reshape(-1)

        x_out = []
        channels = []
        positions_h = []
        positions_w = []
        for i in range(c):
            ind = per_channel_sorted_idx[:channel_k[i].item(), i]

            x_c = x[ind, i, :]
            x_out.append(x_c)
            channels.append(
                    torch.Tensor([i] * channel_k[i])
            )

            positions_h.append(
                    h_indices_f[ind]
                    )
            positions_w.append(
                    w_indices_f[ind]
                    )

        channels = torch.concat(channels)
        channels = channels.to(x.dtype).to(torch.long)

        positions_h = torch.concat(positions_h)
        positions_w = torch.concat(positions_w)
        pos = torch.stack([positions_h, positions_w], dim=-1)

        # shape k, z
        x_out = torch.concat(x_out, dim=0)

        s, z = x_out.shape
        assert z == self.patch_size ** 2, f"{z} != {self.patch_size ** 2}"
        assert s == k

        # x is shape c, k
        return x_out, pos, channels

    @torch.no_grad()
    def _group_patches_by_max_seq_len(
            self,
            batched_patches: List[Tensor],
            batched_positions: List[Tensor],
            batched_channels: List[Tensor],
            state: Optional[GroupPatchesState]=None,
            ) -> GroupPatchesState:
        """
        converts a list of ungrouped tensors into batched groups
        """
        if state is None:
            groups = []
            group = []
            groups_pos = []
            group_pos = []
            groups_channels = []
            group_channels = []
            seq_len = 0
            state = GroupPatchesState(groups, groups_pos, group, group_pos, groups_channels, group_channels, seq_len)

        for patches, pos, channels in zip(batched_patches, batched_positions, batched_channels):
            assert isinstance(patches, Tensor)
            assert isinstance(channels, Tensor)
            assert isinstance(pos, Tensor)

            k, _ = patches.shape

            assert (
            k <= self.max_patch_h*self.max_patch_w*self.channels and k <= self.max_seq_len
            ), f"patch with len {k} exceeds maximum sequence length"

            assert k == channels.shape[0]

            if (state.seq_len + k) > self.max_seq_len:
                state.groups.append(state.group)
                state.groups_pos.append(state.group_pos)
                state.groups_channels.append(state.group_channels)
                state.group = []
                state.group_pos = []
                state.group_channels = []
                state.seq_len = 0

            state.group.append(patches)
            state.group_pos.append(pos)
            state.group_channels.append(channels)
            state.seq_len += k

        return state

    @torch.no_grad()
    def _batch_groups(self,
               grouped_batched_patches: List[List[Tensor]],
               grouped_batched_positions: List[List[Tensor]],
               grouped_batched_channels: List[List[Tensor]],
               device=torch.device('cpu'),
               **dct_patch_kwargs,
                      ) -> DCTPatches:
        """
        returns a batch of DCTPatches
        """
        arange = partial(torch.arange, device=device)


        assert len(grouped_batched_positions) == len(grouped_batched_patches)
        assert len(grouped_batched_positions) == len(grouped_batched_channels)

        # process patches into variable lengthed sequences with attention mask

        num_images = []
        batched_sequences = []
        batched_positions = []
        batched_channels = []
        batched_image_ids = []

        for grouped_patches, grouped_positions, grouped_channels in zip(grouped_batched_patches, grouped_batched_positions, grouped_batched_channels):
            assert len(grouped_patches) == len(grouped_positions)
            assert len(grouped_channels) == len(grouped_positions)
            num_images.append(len(grouped_patches))

            image_ids = torch.empty((0,), device=device, dtype=torch.long)

            for image_id, (patches, positions, channels) in enumerate(zip(grouped_patches, grouped_positions, grouped_channels)):
                k, z = patches.shape

                assert z == self.patch_size ** 2, f"{z} != {self.patch_size ** 2}"

                assert positions.shape[0] == k
                assert channels.shape[0] == k

                image_ids = F.pad(image_ids, (0, k), value=image_id)

            batched_image_ids.append(image_ids)
            batched_sequences.append(torch.cat(grouped_patches, dim=0))
            batched_positions.append(torch.cat(grouped_positions, dim=0))
            batched_channels.append(torch.cat(grouped_channels, dim=0))

        # derive key padding mask

        lengths = torch.tensor(
            [seq.shape[-2] for seq in batched_sequences],
            device=device,
            dtype=torch.long,
        )
        max_length = arange(self.max_seq_len)
        key_pad_mask = rearrange(lengths, "b -> b 1") <= rearrange(
            max_length, "n -> 1 n"
        )

        # derive attention mask, and combine with key padding mask from above

        batched_image_ids = pad_sequence(batched_image_ids, self.max_seq_len)
        attn_mask = rearrange(batched_image_ids, "b i -> b 1 i 1") == rearrange(
            batched_image_ids, "b j -> b 1 1 j"
        )
        attn_mask = attn_mask & rearrange(key_pad_mask, "b j -> b 1 1 j")

        # combine patched images as well as the patched width / height positions for 2d positional embedding

        patches = pad_sequence(batched_sequences, self.max_seq_len).to(device)

        patch_positions = pad_sequence(batched_positions, self.max_seq_len).to(device)
        patch_channels = pad_sequence(batched_channels, self.max_seq_len).to(device)

        # need to know how many images for final attention pooling

        num_images = torch.tensor(num_images, device=device, dtype=torch.long)

        return DCTPatches(
            patches=patches,
            key_pad_mask=key_pad_mask,
            attn_mask=attn_mask,
            batched_image_ids=batched_image_ids,
            patch_positions=patch_positions,
            patch_channels=patch_channels,
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

        for batch_i, (image_ids, mask, positions, channels) in enumerate(
            zip(output.batched_image_ids, output.key_pad_mask, output.patch_positions, output.patch_channels)
        ):
            # take the tokens that have actual images associated with them
            for image_id in image_ids.unique():
                is_image = (image_ids == image_id) & ~mask
                tokens_from_image = x[batch_i, is_image, :]
                positions_from_image = positions[is_image]
                channels_from_image = channels[is_image]
                ph, pw = output.patch_sizes[len(images)]

                image = torch.zeros(self.channels, ph, pw, z, dtype=x.dtype, device=x.device)

                for token, pos, channel in zip(tokens_from_image, positions_from_image, channels_from_image):
                    pos_h,pos_w = pos.unbind(-1)
                    image[channel, pos_h, pos_w, :] = token 

                image = rearrange(image.view(self.channels, ph*pw, -1), 'c (ph pw) (p1 p2) -> c (ph p1) (pw p2)', p1=self.patch_size, p2=self.patch_size, ph=ph, pw=pw, c=self.channels)
                images.append(image)

        return images
