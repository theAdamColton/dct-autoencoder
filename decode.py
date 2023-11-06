"""
creates a gif
demonstrates decoding an image using 
"""
import imageio
import torch
import torchvision
from tqdm import tqdm
from PIL import ImageDraw

from torch_dct import dct_2d, idct_2d
from dct_autoencoder.dct_patches import DCTPatches
from dct_autoencoder.factory import get_model_and_processor
from dct_autoencoder import DCTAutoencoder, DCTAutoencoderFeatureExtractor
from dct_autoencoder.dataset import tuple_collate

torch.set_grad_enabled(False)


def main(
    model_path: str,
    device="cuda",
    dtype=torch.bfloat16,
    image_path: str = "images/holygrail.jpg",
):
    image = torchvision.io.read_image(image_path) / 255
    c, h, w = image.shape
    ar = h / w
    if w < h:
        w = min(512, w)
        h = int(ar * w)
    else:
        h = min(512, h)
        w = int(h / ar)

    image = torchvision.transforms.Resize((h, w))(image)

    image = image.to(dtype).to(device)
    _, h, w = image.shape

    autoenc, proc = get_model_and_processor(resume_path=model_path, device=device, dtype=dtype)

    input_data = proc.preprocess(image)
    input_data = tuple_collate([input_data])

    batch = next(iter(proc.iter_batches(iter([input_data]))))
    batch = batch.to(device)
    input_patches = batch.patches.clone()

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
    end_n = min(batch.patches.shape[1], 60)
    start_n = 1
    jmp = 1
    for i in tqdm(range(start_n, end_n + 1, jmp)):
        masked_dct_patches = mask_and_rec(i)

        _og_transform_out = proc._transform_image_out
        proc._transform_image_out = lambda x: x
        # an image of the DCT features
        image_dct_masked = proc.postprocess(masked_dct_patches)[0].cpu()
        proc._transform_image_out = _og_transform_out

        # reconstructed image with RGB pixels
        image_reconstructed = proc.postprocess(masked_dct_patches)[0].cpu()
        image_reconstructed = image_reconstructed.clip(0,1)

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
            (0, 0), f"#codes: {i * autoenc.config.vq_num_codebooks:05}", (255, 100, 100)  # Coordinates  # Text  # Color
        )

        images.append(im)

    total_gif_duration_ms = 10 * 1000
    n_frames = (end_n - start_n) / jmp
    ms_per_frame = total_gif_duration_ms / n_frames
    print("saving gif")
    imageio.mimsave("dct_zigzag.gif", images, duration=ms_per_frame, loop=0)


if __name__ == "__main__":
    import jsonargparse

    jsonargparse.CLI(main)
