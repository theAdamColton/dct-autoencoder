from typing import Callable, List
from tqdm import tqdm
import torch
import webdataset as wds
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms.v2 as transforms
import torchvision
import os
import math
import time
import json
import wandb
import random
from PIL import ImageDraw

from dct_autoencoder.feature_extraction_dct_autoencoder import DCTAutoencoderFeatureExtractor
from dct_autoencoder.patchnorm import PatchNorm
from dct_autoencoder.util import power_of_two
from dct_autoencoder.dct_patches import DCTPatches, slice_dctpatches
from dct_autoencoder.modeling_dct_autoencoder import DCTAutoencoder
from dct_autoencoder.configuration_dct_autoencoder import DCTAutoencoderConfig


OUTDIR = f"out/{time.ctime()}/"

os.makedirs(OUTDIR, exist_ok=True)

def get_loss_dict(d):
    o = {}
    for k,v in d.items():
        if "loss" in k:
            if isinstance(v, torch.Tensor):
                o[k] = v.item()
            else:
                o[k] = v
    return o

def get_decay_fn(start_val:float, end_value:float, n:int):
    def fn(i: int):
        if i > n:
            return end_value
        return ((n-i) / n) * start_val + (i/n) * end_value
    return fn

def get_collate_fn(processor: DCTAutoencoderFeatureExtractor):
    def collate(x: List[dict]):
        patches = []
        positions = []
        original_sizes = []
        patch_sizes = []
        for d in x:
            patches.append(d["patches"])
            positions.append(d["positions"])
            original_sizes.append(d["original_sizes"])
            patch_sizes.append(d["patch_sizes"])
        return patches, positions, original_sizes, patch_sizes

    return collate


def make_image_grid(x, x_hat, filename=None, n: int = 10, max_size: int = 1024):
    ims = []

    n = min(len(x), n)



    for i in range(n):
        im = x[i]
        im_hat = x_hat[i]
        ims.append(im)
        ims.append(im_hat)

    ims = [transforms.Resize(512, max_size=max_size)(im) for im in ims]

    sizes = [im.shape for im in ims]
    h = max([s[1] for s in sizes])
    w = max([s[2] for s in sizes])

    ims = [F.pad(im, (w - im.shape[2], 0, h - im.shape[1], 0)) for im in ims]


    im = torchvision.utils.make_grid(ims, 2, normalize=False, scale_each=False)
    im = transforms.functional.to_pil_image(im.cpu().to(torch.float32))

    if filename:
        im.save(filename)
        print("saved ", filename)

    return im


def train_step(
           batch: DCTPatches,
           autoencoder: DCTAutoencoder,
           proc: DCTAutoencoderFeatureExtractor,
           max_batch_n = None,
           decode_pixels=False,
           return_loss=True,
           ):
    if max_batch_n is not None:
        batch = slice_dctpatches(batch, max_batch_n)[0]
    input_patches = batch.patches
    
    res = autoencoder(batch, return_loss)

    output_patches = res['dct_patches']

    if decode_pixels:
        # inverse normalization for output_patches
        output_patches.patches = autoencoder.patchnorm.inverse_norm(output_patches.patches, output_patches.h_indices, output_patches.w_indices)

        images_hat = proc.postprocess(output_patches)
        output_patches.patches = input_patches
        images = proc.postprocess(output_patches)

        pixel_loss = 0.0

        for im, im_hat in zip(images, images_hat):
            pixel_loss += F.mse_loss(im, im_hat)

        pixel_loss = pixel_loss / len(images_hat)

        d = dict(
            x_hat=images_hat,
            pixel_loss=pixel_loss,
            x=images,
        )
        res.update(d)

    return res


def train_patch_norm(patch_norm: PatchNorm,
               proc: DCTAutoencoderFeatureExtractor,
               train_ds,
               dtype,
               device,
               batch_size: int = 32,
               steps: int = 20,
               num_workers: int = 0,
                 rng=None,
               ):
    train_ds = train_ds.shuffle(10000, rng=rng)
    dataloader = DataLoader(
                train_ds, batch_size=batch_size, num_workers=num_workers, collate_fn=get_collate_fn(proc)
            )

    patch_norm = patch_norm.to(torch.float32).to(device)

    for i, batch in enumerate(tqdm(proc.iter_batches(iter(dataloader), batch_size))):
        if i+1 > steps:
            break
        batch = batch.to(device)
        batch.patches = batch.patches.to(torch.float32)
        normalized_patches = patch_norm.forward(
            batch.patches, batch.h_indices, batch.w_indices, batch.key_pad_mask
        )
        mean = normalized_patches[~batch.key_pad_mask].mean() .item()
        std = normalized_patches[~batch.key_pad_mask].std()   .item()
        _min = normalized_patches[~batch.key_pad_mask].min()  .item()
        _max = normalized_patches[~batch.key_pad_mask].max()  .item()
        print(f"{i+1:03} mean {mean:.2f} std {std:.2f} min {_min:.2f} max {_max:.2f}")

    patch_norm.frozen = True
    patch_norm = patch_norm.to(dtype)
    return patch_norm


def train(
    autoencoder: DCTAutoencoder,
    proc: DCTAutoencoderFeatureExtractor,
    train_ds,
    optimizer=None,
    batch_size: int = 32,
    learning_rate=3e-3,
    epochs: int = 1,
    device="cuda",
    dtype=torch.bfloat16,
    log_every=20,
    n_log: int = 10,
    max_steps:int=1000000,
    num_workers:int=0,
    warmup_steps=20,
    use_wandb: bool = False,
    vq_callables: List[Callable[[int], dict]] = [],
    rng=None,
):
    if optimizer is None:
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-9)

    collate_fn = get_collate_fn(proc)

    # ----------- Model Training --------------
    for epoch_i, epoch in enumerate(range(epochs)):
        train_ds = train_ds.shuffle(10000, rng=rng)
        dataloader = DataLoader(
            train_ds, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn
        )
        # list of columns, or None
        for i, batch in enumerate(tqdm(proc.iter_batches(iter(dataloader), batch_size))):

            # ----- LR Warmup -----
            if epoch_i == 0 and i < warmup_steps:
                for g in optimizer.param_groups:
                    g["lr"] = ((i + 1) / warmup_steps) * learning_rate


            batch = batch.to(device)
            batch.patches = batch.patches.to(dtype)

            seq_len = batch.patches.shape[1]
            batch_len = batch.patches.shape[0]

            if i % log_every == 0:
                print("logging images ....")
                autoencoder.eval()
                with torch.no_grad():
                    out = train_step(batch, autoencoder, proc, max_batch_n=n_log, decode_pixels=True, return_loss=False)
                autoencoder.train()
                image = make_image_grid(
                    out["x"], out["x_hat"], filename=f"{OUTDIR}/train image {i:04}.png"
                )
                image=wandb.Image(image)
                out = None
            else:
                image=None

            optimizer.zero_grad()


            out = train_step(batch, autoencoder, proc)

            loss = 0.0
            for k,v in out.items():
                if "loss" in k:
                    loss += v

            loss.backward()

            torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), 5.0)

            optimizer.step()


            log_dict = dict(
                            epoch=epoch,
                            step=i,
                            n_images_in_batch = len(batch.original_sizes),
                            perplexity=out["perplexity"].item(),
                            batch_len=batch_len,
                            seq_len=seq_len,
                            **get_loss_dict(out),
            )
            if image:
                log_dict['image'] = image

            for fn in vq_callables:
                d = fn(i)
                for k,v in d.items():
                    autoencoder.vq_model.__dict__[k] = v
                    log_dict[k] = v

            print(log_dict)
            if use_wandb:
                wandb.log(
                    {
                        "train": log_dict
                    },
                )

            if i > max_steps:
                break

    return autoencoder, optimizer


def load_and_transform_dataset(
    dataset_name_or_url: str,
    dct_processor: DCTAutoencoderFeatureExtractor,
    device='cpu',
):
    min_res = dct_processor.patch_size

    def filter_res(x):
        h, w = x["json"]["height"], x["json"]["width"]
        if h is None or w is None:
            return False
        return False if h < min_res or w < min_res else True

    def preproc(x):
        x['pixel_values'] = x['pixel_values'].to(device)
        patches, positions, original_sizes, patch_sizes = dct_processor.preprocess([x["pixel_values"]])
        x["patches"], x["positions"], x["original_sizes"], x["patch_sizes"]  = patches[0], positions[0], original_sizes[0], patch_sizes[0]
        return x

    ds = (
        wds.WebDataset(dataset_name_or_url)
        .map_dict(json=json.loads, handler=wds.handlers.warn_and_continue)
        .select(filter_res,)
        .decode("torchrgb", partial=True, handler=wds.handlers.warn_and_continue)
        .rename(pixel_values="jpg", handler=wds.handlers.warn_and_continue)
        .map(preproc, handler=wds.handlers.warn_and_continue)
        .rename_keys(patches="patches", positions="positions", original_sizes="original_sizes", patch_sizes="patch_sizes",)
    )

    return ds


def get_model(model_config: DCTAutoencoderConfig,
              device,
              dtype,
              sample_patches_beta,
              max_seq_len,
              ):

    proc = DCTAutoencoderFeatureExtractor(
        channels=model_config.image_channels,
        patch_size=model_config.patch_size,
        sample_patches_beta=sample_patches_beta,
        max_n_patches=model_config.max_n_patches,
        max_seq_len=max_seq_len,
    )

    model = DCTAutoencoder(model_config).to(dtype).to(device)

    return model, proc


def main(
    image_dataset_path_or_url:str = None,
    model_config_path = "./conf/patch4_small.json",
    device="cuda",
    dtype=torch.bfloat16,
    batch_size: int = 32,
    num_workers:int=0,
    use_wandb: bool = False,
    train_norm_iters: int = 10,
    max_iters:int = 300,
    sample_patches_beta:float= 2.5,
    batch_size_ft: int = 16,
    max_iters_ft:int = 100,
    sample_patches_beta_ft:float = 0.75,
    max_seq_len: int=512,
    max_seq_len_ft:int = 768,
    commitment_loss_weight_start:float = 5e-4,
    commitment_loss_weight_end:float = 1e-8,
    lfq_entropy_loss_start:float = 5e1,
    lfq_entropy_loss_end:float = 1e-1,
):

    model_config = DCTAutoencoderConfig.from_json_file(model_config_path)
    assert max_seq_len_ft <= model_config.max_n_patches

    autoencoder, processor = get_model(model_config, device, dtype, sample_patches_beta, max_seq_len)

    autoencoder.vq_model.entropy_loss_weight = lfq_entropy_loss_start
    autoencoder.vq_model.commitment_loss_weight = commitment_loss_weight_start

    image_channels: int = model_config.image_channels
    warmup_steps = 50

    commitment_loss_weight_n = max_iters
    commitment_loss_weight_fn = get_decay_fn(commitment_loss_weight_start, commitment_loss_weight_end, commitment_loss_weight_n)


    lfq_entropy_loss_n = max_iters
    lfq_decay_fn = get_decay_fn(lfq_entropy_loss_start, lfq_entropy_loss_end, lfq_entropy_loss_n)

    vq_callables = [lambda i: {"entropy_loss_weight":lfq_decay_fn(i),"commitment_loss_weight":commitment_loss_weight_fn(i)} ]

    learning_rate = 1e-3

    train_ds = load_and_transform_dataset(
        image_dataset_path_or_url,
        processor,
        device=device if num_workers == 0 else 'cpu',
    )

    bits = model_config.vq_num_codebooks * math.log2(model_config.vq_codebook_size) * processor.max_seq_len

    n_params = 0
    for p in autoencoder.parameters():
        n_params += p.nelement()

    run_d = dict(
        compression_over_patches=(processor.max_seq_len * image_channels * model_config.patch_size**2 * 16)  / bits,
        compression_over_image=(512 ** 2 * image_channels * 8)/bits,
        learning_rate=learning_rate,
        commitment_loss_weight_start=commitment_loss_weight_start,
        commitment_loss_weight_end=commitment_loss_weight_end,
        commitment_loss_weight_n=commitment_loss_weight_n,
        n_params=n_params,
        warmup_steps=warmup_steps,
        lfq_entropy_loss_start=lfq_entropy_loss_start,
        lfq_entropy_loss_end=lfq_entropy_loss_end,
        lfq_entropy_loss_n=lfq_entropy_loss_n,
    )

    print("starting run: ", run_d)
    if use_wandb:
        wandb.init(project="vq-experiments", config=run_d)

    rng = random.Random(42)

    # ----------- Norm Training ---------------
    print("training norm")
    autoencoder.patchnorm = train_patch_norm(autoencoder.patchnorm,
                                       processor,
                                       train_ds, dtype, device, batch_size=min(batch_size, 32), steps=train_norm_iters, num_workers=num_workers, rng=rng)
    print("done training norm")

    # This trains using shorter sequence lengths
    autoencoder, optimizer = train(
        autoencoder,
        processor,
        train_ds,
        device=device,
        dtype=dtype,
        learning_rate=learning_rate,
        batch_size=batch_size,
        use_wandb=use_wandb,
        warmup_steps=warmup_steps,
        num_workers=num_workers,
        max_steps=max_iters,
        vq_callables=vq_callables,
        rng=rng,
    )

    # this trains using longer sequence lengths and a smaller batch size
    print("------ft model--------")
    processor.sample_patches_beta = sample_patches_beta_ft
    processor.max_seq_len = max_seq_len_ft
    processor, _ = train(
        autoencoder,
        processor,
        train_ds,
        device=device,
        dtype=dtype,
        learning_rate=learning_rate,
        batch_size=batch_size_ft,
        use_wandb=use_wandb,
        warmup_steps=warmup_steps,
        num_workers=num_workers,
        max_steps=max_iters_ft,
        #vq_callables=vq_callables,
        rng=rng,
    )

    autoencoder.save_pretrained(OUTDIR + "/model/")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    import jsonargparse

    jsonargparse.CLI(main)
