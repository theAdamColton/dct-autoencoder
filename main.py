from typing import Callable, List, Optional
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

from dct_autoencoder.feature_extraction_dct_autoencoder import DCTAutoencoderFeatureExtractor
from dct_autoencoder.patchnorm import PatchNorm
from dct_autoencoder.util import power_of_two, calculate_perplexity
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

def get_collate_fn():
    def collate(x: List[dict]):
        patches = []
        positions = []
        original_sizes = []
        patch_sizes = []
        channels = []
        for d in x:
            patches.append(d["patches"])
            positions.append(d["positions"])
            original_sizes.append(d["original_sizes"])
            patch_sizes.append(d["patch_sizes"])
            channels.append(d['channels'])
        return patches, positions, channels, original_sizes, patch_sizes

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

    ims = [im.clamp(0.0, 1.0) for im in ims]

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

def get_loss_weight_in_patch(h,w,decay:float,device,dtype, eps=1e-2):
    _arange_h = torch.arange(h, device=device, dtype=dtype)
    _arange_w = torch.arange(w, device=device, dtype=dtype)
    _h_ind, _w_ind = torch.meshgrid(_arange_h, _arange_w, indexing='ij')
    in_patch_dist = _h_ind + _w_ind
    # exponential decay
    weight= torch.exp(-decay * in_patch_dist) + eps
    return weight

def get_loss_weight_between_patches(h_positions, w_positions, decay, eps=1e-2):
    weight = h_positions + w_positions
    # exponential decay
    weight = torch.exp(-decay * weight) + eps
    return weight

def weighted_mse_loss(x,y,patch_pixel_weights,patch_position_weights):
    loss = (x-y)**2
    loss = (loss* patch_pixel_weights.unsqueeze(0)).mean()
    loss = (loss * patch_position_weights).mean()
    return loss

def train_step(
           batch: DCTPatches,
           autoencoder: DCTAutoencoder,
           proc: DCTAutoencoderFeatureExtractor,
           max_batch_n = None,
           decode_pixels=False,
           patch_pixel_decay_alpha=0.05,
           patch_position_decay_alpha=0.1,
           ):
    if max_batch_n is not None:
        batch = slice_dctpatches(batch, max_batch_n)[0]

    # the batch is normalized
    batch = autoencoder._normalize(batch)

    # input patches are saved
    input_patches = batch.patches
    
    # runs the autoencoder
    res = autoencoder(batch, )

    output_patches = res['dct_patches']

    # gets the mask, 1s where the sequences wasn't padded
    mask = ~output_patches.key_pad_mask

    # loss weights
#    patch_pixel_weights = get_loss_weight_in_patch(
#            autoencoder.config.patch_size,
#           autoencoder.config.patch_size,
#           patch_pixel_decay_alpha,
#           autoencoder.device,
#           autoencoder.dtype,
#       )
#    patch_position_weights = get_loss_weight_between_patches(
#            output_patches.h_indices,
#            output_patches.w_indices,
#            patch_position_decay_alpha,
#    )

    # normalized loss
    res['rec_loss'] = F.mse_loss(output_patches.patches[mask], input_patches[mask])

    # mse loss between unnormalized features
    _var = autoencoder.patchnorm.var[output_patches.patch_channels,output_patches.h_indices,output_patches.w_indices][mask]
    res['rec_loss_unnormalized'] = (_var * (output_patches.patches[mask] - input_patches[mask]) ** 2).sum() / _var.nelement()

#    res['rec_loss'] = weighted_mse_loss(output_patches.patches[mask],
#                          input_patches[mask],
#                         patch_pixel_weights.flatten(),
#                          patch_position_weights[mask])


    with torch.no_grad():
        res['perplexity'] = calculate_perplexity(res['codes'][mask], autoencoder.config.vq_codebook_size)

    if decode_pixels:
        # inverse normalization for output_patches
        output_patches.patches = autoencoder.patchnorm.inverse_norm(output_patches.patches, output_patches.patch_channels, output_patches.h_indices, output_patches.w_indices)

        # inverse normalization for input patches
        input_patches = autoencoder.patchnorm.inverse_norm(input_patches, output_patches.patch_channels, output_patches.h_indices, output_patches.w_indices)

        # un-patchifies the patches, and converts from dct to pixels
        images_hat = proc.postprocess(output_patches)
        output_patches.patches = input_patches
        images = proc.postprocess(output_patches)

        pixel_loss = 0.0

        for im, im_hat in zip(images, images_hat):
            # rgb space pixel loss
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
                 rng=None,
               ):
    train_ds = train_ds.shuffle(10000, rng=rng)
    dataloader = DataLoader(
                train_ds, batch_size=batch_size, num_workers=0, collate_fn=get_collate_fn()
            )

    patch_norm = patch_norm.to(torch.float32).to(device)

    for i, batch in enumerate(tqdm(proc.iter_batches(iter(dataloader), batch_size))):
        if i+1 > steps:
            break
        batch = batch.to(device)
        batch.patches = batch.patches.to(torch.float32)
        normalized_patches = patch_norm.forward(
            batch.patches, batch.patch_channels, batch.h_indices, batch.w_indices, batch.key_pad_mask
        )
        mean = normalized_patches[~batch.key_pad_mask].mean() .item()
        std = normalized_patches[~batch.key_pad_mask].std()   .item()
        _min = normalized_patches[~batch.key_pad_mask].min()  .item()
        _max = normalized_patches[~batch.key_pad_mask].max()  .item()
        print(f"{i+1:03} mean {mean:.2f} std {std:.2f} min {_min:.2f} max {_max:.2f}")

    patch_norm.frozen = True
    patch_norm = patch_norm.to(dtype)
    return patch_norm

def __set_lr(
        optimizer,lr
        ):
    for g in optimizer.param_groups:
        g["lr"] = lr




def train(
    autoencoder: DCTAutoencoder,
    proc: DCTAutoencoderFeatureExtractor,
    train_ds,
    optimizer=None,
    batch_size: int = 32,
    learning_rate=3e-4,
    epochs: int = 1,
    device="cuda",
    dtype=torch.bfloat16,
    log_every=200,
    n_log: int = 10,
    max_steps:int=1000000,
    num_workers:int=0,
    warmup_steps=10,
    use_wandb: bool = False,
    vq_callables: List[Callable[[int], dict]] = [],
    rng=None,
    save_every=1000,
    use_pixel_loss=False,
    loss_weight={},
   patch_pixel_decay_alpha=0.05,
   patch_position_decay_alpha=0.1,
):
    if optimizer is None:
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-9)

    collate_fn = get_collate_fn()

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
                _lr = ((i + 1) / warmup_steps) * learning_rate
                __set_lr(optimizer, _lr)


            batch = batch.to(device)
            batch.patches = batch.patches.to(dtype)

            seq_len = batch.patches.shape[1]
            batch_len = batch.patches.shape[0]

            if i % log_every == 0:
                print("logging images ....")
                autoencoder.eval()
                with torch.no_grad():
                    out = train_step(batch, autoencoder, proc, max_batch_n=n_log, decode_pixels=True,
                               patch_pixel_decay_alpha=patch_pixel_decay_alpha,
                               patch_position_decay_alpha=patch_position_decay_alpha,
                                     )
                autoencoder.train()
                image = make_image_grid(
                    out["x"], out["x_hat"], filename=f"{OUTDIR}/train image {i:04}.png", n=n_log
                )

                image=wandb.Image(image)
                out = None
            else:
                image=None


            optimizer.zero_grad()


            out = train_step(batch, autoencoder, proc, decode_pixels=use_pixel_loss)

            loss = 0.0
            for k,v in out.items():
                if "loss" in k:
                    if k in loss_weight:
                        weight = loss_weight[k]
                    else:
                        weight = 1.0
                    loss += v * weight

            loss.backward()

            torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), 1.0)

            optimizer.step()


            log_dict = dict(
                            epoch=epoch,
                            step=i,
                            n_images_in_batch = len(batch.original_sizes),
                            perplexity=out["perplexity"].item(),
                            batch_len=batch_len,
                            seq_len=seq_len,
                            loss = loss.item(),
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

            if i % save_every == 0 and i > 0:
                autoencoder.save_pretrained(OUTDIR + "/model/")

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
        patches, positions, channels, original_sizes, patch_sizes = dct_processor.preprocess([x["pixel_values"]])
        x["patches"], x["positions"], x['channels'], x["original_sizes"], x["patch_sizes"]  = patches[0], positions[0], channels[0], original_sizes[0], patch_sizes[0]
        return x

    ds = (
        wds.WebDataset(dataset_name_or_url, handler=wds.handlers.warn_and_continue)
        .map_dict(json=json.loads, handler=wds.handlers.warn_and_continue)
        .select(filter_res,)
        .decode("torchrgb", partial=True, handler=wds.handlers.warn_and_continue)
        .rename(pixel_values="jpg", handler=wds.handlers.warn_and_continue)
        .map(preproc, handler=wds.handlers.warn_and_continue)
        .rename_keys(patches="patches", positions="positions", channels='channels', original_sizes="original_sizes", patch_sizes="patch_sizes", )
    )

    return ds


def get_model(model_config: DCTAutoencoderConfig,
              device,
              dtype,
              sample_patches_beta,
              max_seq_len,
              resume_path,
              ):

    proc = DCTAutoencoderFeatureExtractor(
        channels=model_config.image_channels,
        patch_size=model_config.patch_size,
        sample_patches_beta=sample_patches_beta,
        max_patch_h=model_config.max_patch_h,
        max_patch_w=model_config.max_patch_w,
        max_seq_len=max_seq_len,
    )

    if resume_path is not None:
        model = DCTAutoencoder.from_pretrained(resume_path).to(dtype).to(device)
        print("Loaded from ", resume_path)
    else:
        model = DCTAutoencoder(model_config).to(dtype).to(device)

    return model, proc


def main(
    image_dataset_path_or_url:str = None,
    model_config_path = "./conf/patch4_small.json",
    resume_path: Optional[str] = None,
    device="cuda",
    dtype=torch.bfloat16,
    batch_size: int = 32,
    num_workers:int=0,
    use_wandb: bool = False,
    train_norm_iters: int = 10,
    max_iters:int = 10000,
    sample_patches_beta:float= 2.5,
    batch_size_ft: int = 16,
    max_iters_ft:int = 10000,
    sample_patches_beta_ft:float = 0.01,
    commitment_loss_weight_start:float = 5e-3,
    commitment_loss_weight_end:float = 1e-9,
    lfq_entropy_loss_start:float = 5e1,
    lfq_entropy_loss_end:float = 1e-1,
    learning_rate:float = 1e-3,
    learning_rate_ft:float = 5e-4,
    loss_weight: dict[str,float] = {},
    max_iters_pixel_loss: int =  5000,
    batch_size_pixel_loss: int = 8,
    seed: int=42,
    patch_pixel_decay_alpha:float=0.05,
    patch_position_decay_alpha:float=0.1,
):
    model_config = DCTAutoencoderConfig.from_json_file(model_config_path)



    # the max sequence lengths is based off of the exp dist
    # CDF at some probability

    # CDF: F(x;beta) = 1-e^(-beta*x)
    # x is the sequence length
    # we pick x s.t.
    # .9 = 1-e^(-beta*x)
    # - ln(0.1) / beta = x

    cdf_p = 0.95
    image_channels: int = model_config.image_channels
    max_seq_len = round(-1*math.log(1-cdf_p) / sample_patches_beta)
    max_seq_len = power_of_two(max_seq_len)
    max_seq_len = min(model_config.max_patch_h * model_config.max_patch_w * image_channels, max_seq_len)
    max_seq_len_ft= model_config.max_patch_h * model_config.max_patch_w

    autoencoder, processor = get_model(model_config, device, dtype, sample_patches_beta, max_seq_len, resume_path)

    autoencoder.vq_model.entropy_loss_weight = lfq_entropy_loss_start
    autoencoder.vq_model.commitment_loss_weight = commitment_loss_weight_start

    warmup_steps = 50

    commitment_loss_weight_n = max_iters
    commitment_loss_weight_fn = get_decay_fn(commitment_loss_weight_start, commitment_loss_weight_end, commitment_loss_weight_n)


    lfq_entropy_loss_n = max_iters
    lfq_decay_fn = get_decay_fn(lfq_entropy_loss_start, lfq_entropy_loss_end, lfq_entropy_loss_n)

    vq_callables = [lambda i: {"entropy_loss_weight":lfq_decay_fn(i),"commitment_loss_weight":commitment_loss_weight_fn(i)} ]


    train_ds = load_and_transform_dataset(
        image_dataset_path_or_url,
        processor,
        device=device if num_workers == 0 else 'cpu',
    )

    n_params = 0
    for n,p in autoencoder.named_parameters():
        if 'patchnorm' not in n:
            n_params += p.nelement()

    run_d = dict(
        sample_patches_beta_ft=sample_patches_beta_ft,
        sample_patches_beta=sample_patches_beta,
        max_seq_len=max_seq_len,
        max_seq_len_ft = max_seq_len_ft,
        learning_rate=learning_rate,
        learning_rate_ft=learning_rate_ft,
        commitment_loss_weight_start=commitment_loss_weight_start,
        commitment_loss_weight_end=commitment_loss_weight_end,
        commitment_loss_weight_n=commitment_loss_weight_n,
        n_params=n_params,
        warmup_steps=warmup_steps,
        lfq_entropy_loss_start=lfq_entropy_loss_start,
        lfq_entropy_loss_end=lfq_entropy_loss_end,
        lfq_entropy_loss_n=lfq_entropy_loss_n,
        feature_dim=model_config.feature_dim,
        n_attention_heads = model_config.n_attention_heads,
        patch_size=model_config.patch_size,
        patch_pixel_decay_alpha=patch_pixel_decay_alpha,
        patch_position_decay_alpha=patch_position_decay_alpha,
        **loss_weight,
    )

    print("starting run: ", run_d)
    if use_wandb:
        wandb.init(project="vq-experiments", config=run_d)

    rng = random.Random(seed)

    # ----------- Norm Training ---------------
    if train_norm_iters > 0:
        # resets the norm
        if resume_path is not None:
            autoencoder.patchnorm = PatchNorm(autoencoder.config.max_patch_h, autoencoder.config.max_patch_w, autoencoder.config.patch_size, autoencoder.config.image_channels)

        print("training norm")
        processor.sample_patches_beta = 0.0
        processor.max_seq_len = model_config.max_patch_h * model_config.max_patch_w * 3
        autoencoder.patchnorm = train_patch_norm(autoencoder.patchnorm,
                                           processor,
                                           train_ds, dtype, device, batch_size=min(batch_size, 32), steps=train_norm_iters,rng=rng)
        processor.max_seq_len = max_seq_len 
        processor.sample_patches_beta = sample_patches_beta
        print("done training norm")

    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-9)

    # This trains using shorter sequence lengths
    if max_iters > 0:
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
            loss_weight=loss_weight,
            optimizer=optimizer,
            patch_pixel_decay_alpha=patch_pixel_decay_alpha,
            patch_position_decay_alpha=patch_position_decay_alpha,
        )

    # this trains using longer sequence lengths and a smaller batch size
    lfq_decay_fn = get_decay_fn(lfq_entropy_loss_start, lfq_entropy_loss_end, max_iters_ft)
    commitment_loss_weight_fn = get_decay_fn(commitment_loss_weight_start, commitment_loss_weight_end, max_iters_ft)
    vq_callables = [lambda i: {"entropy_loss_weight":lfq_decay_fn(i),"commitment_loss_weight":commitment_loss_weight_fn(i)} ]
    if max_iters_ft > 0:
        print("------ft model--------")
        processor.sample_patches_beta = sample_patches_beta_ft
        processor.max_seq_len = max_seq_len_ft
        processor, _ = train(
            autoencoder,
            processor,
            train_ds,
            optimizer=optimizer,
            device=device,
            dtype=dtype,
            learning_rate=learning_rate_ft,
            batch_size=batch_size_ft,
            use_wandb=use_wandb,
            warmup_steps=warmup_steps,
            num_workers=num_workers,
            max_steps=max_iters_ft,
            vq_callables=vq_callables,
            rng=rng,
            loss_weight=loss_weight,
            patch_pixel_decay_alpha=patch_pixel_decay_alpha,
            patch_position_decay_alpha=patch_position_decay_alpha,
        )

    # this trains using long sequence lengths as well as pixel loss
    if max_iters_pixel_loss > 0:
        print("------ft model with pixel loss--------")
        processor.sample_patches_beta = sample_patches_beta_ft
        processor.max_seq_len = max_seq_len_ft
        processor, _ = train(
            autoencoder=autoencoder,
            proc=processor,
            train_ds=train_ds,
            optimizer=optimizer,
            device=device,
            dtype=dtype,
            learning_rate=learning_rate_ft,
            batch_size=batch_size_pixel_loss,
            use_wandb=use_wandb,
            warmup_steps=warmup_steps,
            num_workers=num_workers,
            max_steps=max_iters_pixel_loss,
            rng=rng,
            loss_weight=loss_weight,
            use_pixel_loss=True,
            patch_pixel_decay_alpha=patch_pixel_decay_alpha,
            patch_position_decay_alpha=patch_position_decay_alpha,
        )


    print("done with all training")
    autoencoder.save_pretrained(OUTDIR + "/model/")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    import jsonargparse

    jsonargparse.CLI(main)
