"""
creates a gif
demonstrates decoding an image using 
"""
import imageio
import torch
import torchvision
from tqdm import tqdm
from PIL import ImageDraw

from dct_autoencoder.factory import get_model_and_processor
from dct_autoencoder.dataset import dict_collate, tuple_collate
from dct_autoencoder.util import image_clip

torch.set_grad_enabled(False)


def main(
    model_path: str,
    device="cuda",
    dtype=torch.float16,
    image_path: str = "images/bold.jpg",
):
    image = torchvision.io.read_image(image_path) / 255
    c, h, w = image.shape
    ar = h / w
    if w < h:
        w = min(1024, w)
        h = int(ar * w)
    else:
        h = min(1024, h)
        w = int(h / ar)

    image = torchvision.transforms.Resize((h, w))(image)

    image = image.to(dtype).to(device)
    _, h, w = image.shape

    autoenc, proc = get_model_and_processor(resume_path=model_path, device=device, dtype=dtype, sample_patches_beta=0.0)

    input_data = proc.preprocess(image)
    input_data = dict_collate([input_data])

    batch = next(iter(proc.iter_batches(iter([input_data]))))
    batch = batch.to(device)
    input_patches = batch.patches.clone()

    is_image_zero_mask = (
        (batch.batched_image_ids[0] == 0)
        & ~batch.key_pad_mask
        & (
            torch.arange(batch.patches.shape[0], device=batch.patches.device) == 0
        ).unsqueeze(-1)
    )
    n_patches_image_zero = is_image_zero_mask.sum().item()

    res = autoenc(batch, do_normalize=True)
    codes = res["codes"]

    def mask_and_rec(i: int):
        # mask all but i codes
        codes_masked = codes[:, :i, :]

        # slices along seq len
        # only works for a dct patches with a single image
        return autoenc.decode_from_codes(
            codes_masked,
            do_inv_norm=True,
            key_pad_mask=batch.key_pad_mask[:,:i],
            attn_mask=batch.attn_mask[:,:,:i,:i],
            batched_image_ids=batch.batched_image_ids[:,:i],
            patch_channels=batch.patch_channels[:,:i],
            patch_positions=batch.patch_positions[:,:i],
            patch_sizes=batch.patch_sizes,
            original_sizes=batch.original_sizes,
        )

    images = []
    end_n = min(n_patches_image_zero, 16)
    start_n = 1
    jmp = 1
    for i in tqdm(range(start_n, end_n + 1, jmp)):
        masked_dct_patches = mask_and_rec(i)

        _og_transform_out = proc._transform_image_out
        proc._transform_image_out = lambda x: x
        # an image of the DCT features
        image_dct_masked = proc.postprocess(masked_dct_patches)[0].cpu()
        image_dct_masked = (image_dct_masked.abs() * 10).clip(0,1)
        proc._transform_image_out = _og_transform_out

        # reconstructed image with RGB pixels
        image_reconstructed = proc.postprocess(masked_dct_patches)[0].cpu()
        image_reconstructed = image_clip(image_reconstructed)

        # ground truth image
        masked_dct_patches.patches = input_patches[:, :i, :]
        image_original = proc.postprocess(masked_dct_patches)[0].cpu()
        image_original = image_original.clip(0,1)

        im = torchvision.utils.make_grid(
            [image_original, image_reconstructed, image_dct_masked],
            nrow=3,
            scale_each=False,
            normalize=False,
        )
        im = torchvision.transforms.ToPILImage()(im)
        ImageDraw.Draw(im).text(  # Image
            (0, 0), f"#codes: {i * autoenc.config.vq_num_codebooks:05}", (150, 0, 0)  # Coordinates  # Text  # Color
        )

        images.append(im)

    total_gif_duration_ms = 7 * 1000
    n_frames = (end_n - start_n) / jmp
    ms_per_frame = total_gif_duration_ms / n_frames
    print("saving gif")
    imageio.mimsave("dct_zigzag.gif", images, duration=ms_per_frame, loop=0)


if __name__ == "__main__":
    import jsonargparse

    jsonargparse.CLI(main)
