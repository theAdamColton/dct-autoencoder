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
from dct_autoencoder import DCTAutoencoder, DCTAutoencoderFeatureExtractor
from dct_autoencoder.util import tuple_collate
torch.set_grad_enabled(False)


def main(model_path: str, device="cuda", dtype=torch.bfloat16, image_path:str="images/holygrail.jpg"):
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

    autoenc = DCTAutoencoder.from_pretrained(model_path).to(device).to(dtype).eval()
    proc = DCTAutoencoderFeatureExtractor(
        channels=autoenc.config.image_channels,
        patch_size=autoenc.config.patch_size,
        sample_patches_beta=0.0,
        max_patch_h=autoenc.config.max_patch_h,
        max_patch_w=autoenc.config.max_patch_w,
        max_seq_len=autoenc.config.max_n_patches,
    )

    input_data = proc.preprocess(image)
    input_data = tuple_collate([input_data])

    batch = next(iter(proc.iter_batches(iter([input_data]))))
    batch = batch.to(device)

    res = autoenc(batch, do_normalize=True)
    codes = res['codes']
    dct_patches = res['dct_patches']

    def mask_and_rec(i: int):
        # mask all but i codes
        codes_masked = codes[:,:i,:]

        return autoenc.decode_from_codes(codes_masked, key_pad_mask=batch.key_pad_mask, 
                                        attn_mask=batch.attn_mask,
                                        batched_image_ids=batch.batched_image_ids,
                                        patch_channels=batch.patch_channels,
                                        patch_positions=batch.patch_positions,
                                        patch_sizes = batch.patch_sizes,
                                        original_sizes=batch.original_sizes,
                                    )

    images = []
    n = min(batch.patches.shape[1], 10)
    jmp = 1
    for i in tqdm(range(1, n+1, jmp)):
        masked_dct_patches = mask_and_rec(i)

        _og_transform_out = proc._transform_image_out
        proc._transform_image_out = lambda x:x
        # an image of the DCT features
        image_dct_masked = proc.postprocess(masked_dct_patches)[0].cpu()
        proc._transform_image_out = _og_transform_out

        # an image with RGB pixels
        image_masked = proc.postprocess(masked_dct_patches)[0].cpu()

        masked_dct_patches.patches = batch.patches.clone()[:, :i, :]
        #masked_dct_patches = inv_norm(masked_dct_patches)
        image_original = proc.postprocess(masked_dct_patches)[0].cpu()
        image_original = _norm_image(image_original)

#        image_masked = rgb2hsv_torch(image_masked.unsqueeze(0)).squeeze(0)
#        image_masked[1] = image_masked[1] + 0.0
#        image_masked = rgb2hsv_torch(image_masked.unsqueeze(0)).squeeze(0)

        image_masked = _norm_image(image_masked)

        im = torchvision.utils.make_grid(
            [image_original, image_masked, image_dct_masked], nrow=3, scale_each=False, normalize=False
        )
        im = torchvision.transforms.ToPILImage()(im)
        ImageDraw.Draw(im).text(  # Image
            (0, 0), f"{i:05}", (255, 255, 2555555)  # Coordinates  # Text  # Color
        )

        images.append(im)

    imageio.mimsave("dct_zigzag.gif", images, duration=3 * (n / jmp))


if __name__ == "__main__":
    import jsonargparse

    jsonargparse.CLI(main)
