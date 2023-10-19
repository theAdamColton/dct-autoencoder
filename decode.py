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
from dct_autoencoder.util import flatten_zigzag, unflatten_zigzag, zigzag
from dct_autoencoder import DCTAutoencoder, DCTAutoencoderFeatureExtractor
torch.set_grad_enabled(False)


def main(model_path: str, device="cuda", dtype=torch.bfloat16):
    image = torchvision.io.read_image("images/bold.jpg") / 255
    c, h, w = image.shape
    ar = h / w
    if w < h:
        w = min(256, w)
        h = int(ar * w)
    else:
        h = min(256, h)
        w = int(h / ar)
    image = torchvision.transforms.Resize((h, w))(image)
    image_dct = dct_2d(image, "ortho")
    image = image.to(dtype).to(device)
    c, h, w = image.shape

    autoenc = DCTAutoencoder.from_pretrained(model_path).to(device).to(dtype).eval()
    proc = DCTAutoencoderFeatureExtractor(
        channels=autoenc.config.image_channels,
        patch_size=autoenc.config.patch_size,
        sample_patches_beta=0.0,
        max_n_patches=autoenc.config.max_n_patches,
        max_seq_len=autoenc.config.max_n_patches,
    )

    input_data = proc.preprocess([image])

    batch = next(iter(proc.iter_batches(iter([input_data]))))
    batch = batch.to(device)

    #encoded_dct_patches, _, _ = autoenc.encode(batch, do_normalize=False)

    distances = (
        batch.patch_positions[0, :, 0] + batch.patch_positions[0, :, 1]
    ) * 1.0
    distances_i = distances.sort().indices

    def inv_norm(p):
        p.patches = autoenc.patchnorm.inverse_norm(
                p.patches,
                pos_h = p.patch_positions[...,0],
                pos_w = p.patch_positions[...,1],
                )
        return p


    def mask_and_rec(i: int):
        # mask all but i codes
#        distances_i_mask = distances_i[:i]
#        masked_dct_patches = DCTPatches(
#            patches=encoded_dct_patches.patches[:, distances_i_mask],
#            key_pad_mask=encoded_dct_patches.key_pad_mask[:, distances_i_mask],
#            attn_mask=encoded_dct_patches.attn_mask[:,:, distances_i_mask][..., distances_i_mask],
#            batched_image_ids=encoded_dct_patches.batched_image_ids[:, distances_i_mask],
#            patch_positions=encoded_dct_patches.patch_positions[:, distances_i_mask],
#            patch_sizes=encoded_dct_patches.patch_sizes,
#            original_sizes=encoded_dct_patches.original_sizes,
#        )
        masked_batch = DCTPatches(
                patches=batch.patches[:, :i].clone(),
                key_pad_mask=batch.key_pad_mask[:, :i].clone(),
                attn_mask=batch.attn_mask[:,:, :i,:][..., :i].clone(),
                batched_image_ids=batch.batched_image_ids[:, :i].clone(),
                patch_positions=batch.patch_positions[:, :i].clone(),
            patch_sizes=batch.patch_sizes,
            original_sizes=batch.original_sizes,
        )
        out = autoenc(masked_batch)
        out_dct_patches = out['dct_patches']
        out_dct_patches = inv_norm(out_dct_patches)
        return out_dct_patches

    def _norm_image(x):
        def norm_ip(img, low, high):
            img.clamp_(min=low, max=high)
            img.sub_(low).div_(max(high - low, 1e-5))
            return img

        def norm_range(t):
            return norm_ip(t, float(t.min()), float(t.max()))
        return norm_range(x)
        q = 0.01
        x = x + x.quantile(q, dim=0, keepdim=True)
        x = x / x.quantile(1-q, dim=0, keepdim=True)
        x = x.clamp(0,1)
        return x

    images = []
    n = autoenc.config.max_n_patches
    jmp = 32
    for i in tqdm(range(1, n, jmp)):
        masked_dct_patches = mask_and_rec(i)

        proc._transform_image_out = lambda x:x
        image_dct_masked = proc.postprocess(masked_dct_patches)[0].cpu()
        proc._transform_image_out = lambda x: idct_2d(x.float(), 'ortho')

        image_masked = proc.postprocess(masked_dct_patches)[0].cpu()

        #image_masked = image_dct.clamp(0.0, 1.0)
        #image_dct_masked = image_dct.clamp(0.0, 1.0)

        masked_dct_patches.patches = batch.patches.clone()[:, distances_i[:i]]
        #masked_dct_patches = inv_norm(masked_dct_patches)
        image_original = proc.postprocess(masked_dct_patches)[0].cpu()

        image_original = _norm_image(image_original)
        image_masked = _norm_image(image_masked)

        im = torchvision.utils.make_grid(
            [image_original, image_masked, image_dct_masked], nrow=3, scale_each=False, normalize=False
        )
        im = torchvision.transforms.ToPILImage()(im)
        ImageDraw.Draw(im).text(  # Image
            (0, 0), f"{i:05}", (255, 255, 2555555)  # Coordinates  # Text  # Color
        )

        images.append(im)

    imageio.mimsave("dct_zigzag.gif", images, duration=15 / (n / jmp))


if __name__ == "__main__":
    import jsonargparse

    jsonargparse.CLI(main)
