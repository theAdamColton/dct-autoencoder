from typing import Callable, List
from tqdm import tqdm
import torch
import vector_quantize_pytorch
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

from dct_autoencoder.dct_processor import DCTPatches, DCTProcessor
from dct_autoencoder.dct_autoenc import DCTAutoencoder

OUTDIR = f"out/{time.ctime()}/"

os.makedirs(OUTDIR, exist_ok=True)


def get_decay_fn(start_val:float, end_value:float, n:int):
    def fn(i: int):
        if i > n:
            return end_value
        return ((n-i) / n) * start_val + (i/n) * end_value
    return fn

def get_collate_fn(processor: DCTProcessor):
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
        return processor.batch(patches, positions, original_sizes, patch_sizes)

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

    im = torchvision.utils.make_grid(ims, 2, normalize=True, scale_each=True)
    im = transforms.functional.to_pil_image(im.cpu().to(torch.float32))
    if filename:
        im.save(filename)
        print("saved ", filename)

    return im


def train(
    autoencoder: DCTAutoencoder,
    train_ds,
    batch_size: int = 32,
    learning_rate=3e-3,
    epochs: int = 1,
    device="cuda",
    dtype=torch.bfloat16,
    log_every=20,
    n_log: int = 10,
    max_steps=1e20,
    warmup_steps=20,
    train_norm_iters: int = 15,  # number of iterations to train the norm layer
    use_wandb: bool = False,
    vq_callables: List[Callable[[int], dict]] = [],
):
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-9)

    n_steps = 0

    collate_fn = get_collate_fn(autoencoder.dct_processor, )

    # trains norm layer
    train_ds = train_ds.shuffle(1000)

    def get_dataloader():
        return DataLoader(
            train_ds, batch_size=batch_size, num_workers=4, collate_fn=collate_fn
        )

    dataloader = get_dataloader()

    print("training norms")
    autoencoder.dct_processor.patch_norm.dtype = torch.float32
    for i, batch in enumerate(tqdm(dataloader)):
        if i > train_norm_iters:
            break
        batch = batch.to(device)
        normalized_patches = autoencoder.dct_processor.patch_norm.forward(
            batch.patches, batch.h_indices, batch.w_indices, batch.key_pad_mask
        )
        mean = normalized_patches[~batch.key_pad_mask].mean() .item()
        std = normalized_patches[~batch.key_pad_mask].std()   .item()
        _min = normalized_patches[~batch.key_pad_mask].min()  .item()
        _max = normalized_patches[~batch.key_pad_mask].max()  .item()
        print(f"{i:03} mean {mean:.2f} std {std:.2f} min {_min:.2f} max {_max:.2f}")

    autoencoder.dct_processor.patch_norm.frozen = True
    autoencoder.dct_processor.patch_norm = autoencoder.dct_processor.patch_norm.to(dtype)
    print("done training norm")

    for epoch_i, epoch in enumerate(range(epochs)):
        train_ds = train_ds.shuffle(5000)
        dataloader = get_dataloader()
        for i, batch in enumerate(tqdm(dataloader)):
            if epoch_i == 0 and i < warmup_steps:
                for g in optimizer.param_groups:
                    g["lr"] = ((i + 1) / warmup_steps) * learning_rate

            batch = batch.to(device)
            batch.patches = batch.patches.to(dtype)

            seq_len = batch.patches.shape[1]
            batch_len = batch.patches.shape[0]

            # normalizes 
            with torch.no_grad():
                batch.patches = autoencoder.dct_processor.patch_norm.forward(batch.patches, batch.h_indices, batch.w_indices, batch.key_pad_mask)

            if i % log_every == 0:
                print("logging images ....")
                autoencoder.eval()
                log_batch = DCTPatches(
                        batch.patches[:n_log],
                        batch.key_pad_mask[:n_log],
                        batch.h_indices[:n_log],
                        batch.w_indices[:n_log],
                        batch.attn_mask[:n_log],
                        batch.batched_image_ids[:n_log],
                        batch.patch_positions[:n_log],
                        batch.patch_sizes,
                        batch.original_sizes,
                )
                with torch.no_grad():
                    out = autoencoder(
                        log_batch,
                        decode=True,
                    )
                autoencoder.train()
                image = make_image_grid(
                    out["x"], out["x_hat"], filename=f"{OUTDIR}/train image {i:04}.png"
                )
                log_d = {
                    "train": dict(
                        epoch=epoch,
                        step=i,
                        image=wandb.Image(image),
                        pixel_loss=out["pixel_loss"].item(),
                        rec_loss=out["rec_loss"].item(),
                    )
                }
                print("log:", log_d)

                if use_wandb:
                    wandb.log(log_d)
                out = None

            optimizer.zero_grad()

            out = autoencoder(
                batch,
                decode=False,
            )

            commit_loss = out["commit_loss"].mean()

            loss = out["rec_loss"] + commit_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), 5.0)
            optimizer.step()

            log_dict = dict(
                            epoch=epoch,
                            step=i,
                            loss=loss.item(),
                            rec_loss=out["rec_loss"].item(),
                            commit_loss=commit_loss.item(),
                            perplexity=out["perplexity"].item(),
                            batch_len=batch_len,
                            seq_len=seq_len,
            )

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
                    }
                )

            if n_steps > max_steps:
                return autoencoder

            n_steps += 1

    return autoencoder


def load_and_transform_dataset(
    dataset_name_or_url: str,
    dct_processor: DCTProcessor,
):
    min_res = dct_processor.patch_size

    def filter_res(x):
        h, w = x["json"]["height"], x["json"]["width"]
        if h is None or w is None:
            return False
        return False if h < min_res or w < min_res else True

    def preproc(x):
        patches, positions, original_sizes, patch_sizes = dct_processor.preprocess([x["pixel_values"]])
        x["patches"], x["positions"], x["original_sizes"], x["patch_sizes"]  = patches[0], positions[0], original_sizes[0], patch_sizes[0]
        return x

    ds = (
        wds.WebDataset(dataset_name_or_url)
        .map_dict(json=json.loads)
        .select(filter_res)
        .decode("torchrgb", partial=True, handler=wds.handlers.warn_and_continue)
        .rename(pixel_values="jpg")
        .map(preproc)
        .rename_keys(patches="patches", positions="positions", original_sizes="original_sizes", patch_sizes="patch_sizes")
    )  # drops old columns

    return ds


def main(
    image_dataset_path_or_url="imagenet-1k",
    device="cuda",
    dtype=torch.bfloat16,
    batch_size: int = 32,
    use_wandb: bool = False,
    train_norm_iters: int = 10,
):
    max_iters = 200


    feature_channels: int = 1024
    image_channels: int = 3
    dct_compression_factor = 0.40
    max_n_patches = 256
    max_seq_len = 384
    max_batch_size = batch_size
    depth = 3
    warmup_steps = 50

    codebook_dim=32
    threshold_ema_dead_code=5
    straight_through=True
    sync_update_v=0.01
    learnable_codebook=True

    use_cosine_sim=False

    commitment_loss_weight_start = 5e-3
    commitment_loss_weight_end = 5e-5
    commitment_loss_weight_n = max_iters
    commitment_loss_weight_fn = get_decay_fn(commitment_loss_weight_start, commitment_loss_weight_end, commitment_loss_weight_n)

    lfq_entropy_loss_start = 5e-0
    lfq_entropy_loss_n = max_iters
    lfq_entropy_loss_end = 1e-4
    lfq_decay_fn = get_decay_fn(lfq_entropy_loss_start, lfq_entropy_loss_end, lfq_entropy_loss_n)

    sample_codebook_temp= 1.0


    def get_model(codebook_size, heads, patch_size, vq_type, ):
        if vq_type=="lfq":
            vq_model = vector_quantize_pytorch.LFQ(
                    dim=feature_channels,
                    codebook_size=codebook_size,
                    num_codebooks=heads,
                    entropy_loss_weight=lfq_entropy_loss_start,
                    commitment_loss_weight=commitment_loss_weight_start,
                )
            vq_callables = [lambda i: {"entropy_loss_weight":lfq_decay_fn(i),"commitment_loss_weight":commitment_loss_weight_fn(i)} ]
        elif vq_type == "vq":
            vq_model = vector_quantize_pytorch.VectorQuantize(
                feature_channels,
                codebook_size=codebook_size,
                codebook_dim=codebook_dim,
                threshold_ema_dead_code=threshold_ema_dead_code,
                heads=heads,
                channel_last=True,
                accept_image_fmap=False,
                use_cosine_sim=use_cosine_sim,
                kmeans_init=True,
                kmeans_iters=20,
                separate_codebook_per_head=False,

                learnable_codebook=learnable_codebook,
                ema_update=not learnable_codebook,
                affine_param=learnable_codebook,
                straight_through=straight_through,
                affine_param_batch_decay = 0.99,
                affine_param_codebook_decay = 0.9,
                sync_update_v=sync_update_v,

                sample_codebook_temp=sample_codebook_temp,
                commitment_weight=commitment_loss_weight_start,
            ).to(torch.float32).to(device)
            vq_callables = [lambda i: {"entropy_loss_weight":lfq_decay_fn(i),"commitment_weight":commitment_loss_weight_fn(i)} ]


        proc = DCTProcessor(
            channels=image_channels,
            patch_size=patch_size,
            dct_compression_factor=dct_compression_factor,
            max_n_patches=max_n_patches,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            patch_norm_device=device,
        )

        model = (
            DCTAutoencoder(
                vq_model,
                feature_channels=feature_channels,
                depth=depth,
                patch_size=patch_size,
                max_n_patches=max_n_patches,
                dct_processor=proc,
                codebook_size=codebook_size,
            )
            .to(dtype)
            .to(device)
        )

        model.vq_model = model.vq_model.to(torch.float32)

        return model, proc, vq_callables

    codebook_size = 512

    #    for codebook_size in codebook_sizes:
    #        for heads in head_numbers:
    for learning_rate in [5e-5, 1e-4, 5e-4]:
        for patch_size in [8, 16]:
            for vq_type in ["lfq",]:
                for heads in [16]:
                    autoencoder, processor, vq_callables = get_model(codebook_size, heads, patch_size, vq_type, )

                    ds_train = load_and_transform_dataset(
                        image_dataset_path_or_url,
                        processor,
                    )

                    bits = heads * math.log2(codebook_size) * max_seq_len

                    run_d = dict(
                        codebook_size=codebook_size,
                        heads=heads,
                        image_channels=image_channels,
                        dct_compression_factor=dct_compression_factor,
                        max_n_patches=max_n_patches,
                        depth=depth,
                        bits= bits,
                        use_cosine_sim=use_cosine_sim,
                        compression_over_patches=max_seq_len * image_channels * patch_size * patch_size / bits,
                        compression_over_image=256 ** 2 * image_channels/bits,
                        feature_dim=feature_channels,
                        patch_size=patch_size,
                        learning_rate=learning_rate,
                        codebook_dim=codebook_dim,
                        sample_codebook_temp =sample_codebook_temp,
                        commitment_loss_weight_start=commitment_loss_weight_start,
                        commitment_loss_weight_end=commitment_loss_weight_end,
                        commitment_loss_weight_n=commitment_loss_weight_n,
                        threshold_ema_dead_code=threshold_ema_dead_code,
                        warmup_steps=warmup_steps,
                        straight_through=straight_through,
                        sync_update_v=sync_update_v,
                        learnable_codebook=learnable_codebook,
                        vq_type=vq_type,
                        lfq_entropy_loss_start=lfq_entropy_loss_start,
                        lfq_entropy_loss_end=lfq_entropy_loss_end,
                        lfq_entropy_loss_n=lfq_entropy_loss_n,
                    )

                    print("starting run: ", run_d)

                    if use_wandb:
                        wandb.init(project="vq-experiments", config=run_d)

                    try:
                        autoencoder = train(
                            autoencoder,
                            ds_train,
                            device=device,
                            dtype=dtype,
                            batch_size=batch_size,
                            use_wandb=use_wandb,
                            train_norm_iters=train_norm_iters,
                            warmup_steps=warmup_steps,
                            max_steps=max_iters,
                            vq_callables=vq_callables,
                        )
                    except Exception as e:
                        print("training run crashed", e)

                    autoencoder = None

                    if use_wandb:
                        wandb.finish()


if __name__ == "__main__":
    import jsonargparse

    jsonargparse.CLI(main)
