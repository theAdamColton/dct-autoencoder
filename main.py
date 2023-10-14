from typing import List
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


def get_collate_fn(processor: DCTProcessor, device):
    def collate(x: List[dict]):
        dct_features = []
        original_sizes = []
        for d in x:
            dct_features.append(d["dct_features"])
            original_sizes.append(d["original_sizes"])
        return processor.patch_images(dct_features, original_sizes)

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
    learning_rate=1e-3,
    epochs: int = 1,
    device="cuda",
    dtype=torch.bfloat16,
    log_every=10,
    n_log: int = 10,
    max_steps=1e20,
    warmup_steps=20,
    train_norm_iters: int = 15,  # number of iterations to train the norm layer
    use_wandb: bool = False,
):
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-9)

    n_steps = 0

    collate_fn = get_collate_fn(autoencoder.dct_processor, device)

    # trains norm layer
    train_ds = train_ds.shuffle(1000)

    def get_dataloader():
        return DataLoader(
            train_ds, batch_size=batch_size, num_workers=0, collate_fn=collate_fn
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
    autoencoder.dct_processor.patch_norm.dtype = dtype
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
                        batch.has_token_dropout,
                        batch.original_sizes,
                )
                with torch.no_grad():
                    with torch.autocast(device):
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

            optimizer.zero_grad()

            with torch.autocast(device, dtype=dtype):
                out = autoencoder(
                    batch,
                    decode=False,
                )

            commit_loss = out["commit_loss"]

            loss = out["rec_loss"] + commit_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), 5.0)
            optimizer.step()

            print(
                f"epoch: {epoch} loss: {loss.item():.3f} rec_loss: {out['rec_loss'].item():.2f} commit_loss: {commit_loss.item():.2f} perplexity: {out['perplexity'].item():.2f}"
            )

            if use_wandb:
                wandb.log(
                    {
                        "train": dict(
                            epoch=epoch,
                            step=i,
                            loss=loss.item(),
                            rec_loss=out["rec_loss"].item(),
                            commit_loss=commit_loss.item(),
                            perplexity=out["perplexity"].item(),
                        )
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
        x["dct_features"], x["original_sizes"] = dct_processor.image_to_dct(
            x["pixel_values"]
        )
        return x

    ds = (
        wds.WebDataset(dataset_name_or_url)
        .map_dict(json=json.loads)
        .select(filter_res)
        .decode("torchrgb", partial=True, handler=wds.handlers.warn_and_continue)
        .rename(pixel_values="jpg")
        .map(preproc)
        .rename_keys(original_sizes="original_sizes", dct_features="dct_features")
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
    feature_channels: int = 768
    mlp_dim: int = 1024
    image_channels: int = 3
    dct_compression_factor = 0.75
    max_n_patches = 64
    max_seq_len = 80
    max_batch_size = batch_size
    depth = 4

    def get_model(codebook_size, heads, patch_size):
        vq_model = vector_quantize_pytorch.VectorQuantize(
            feature_channels,
            codebook_size=codebook_size,
            codebook_dim=32,
            threshold_ema_dead_code=1,
            heads=heads,
            channel_last=True,
            accept_image_fmap=False,
            kmeans_init=True,
            separate_codebook_per_head=False,
            sample_codebook_temp=0.8,
            decay=0.89,
            commitment_weight=1e-5,
        ).to(torch.float32).to(device)

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
                mlp_dim=mlp_dim,
                depth=depth,
                patch_size=patch_size,
                max_n_patches=max_n_patches,
                dct_processor=proc,
                codebook_size=codebook_size,
            )
            .to(dtype)
            .to(device)
        )

        model.dct_processor.patch_norm = model.dct_processor.patch_norm.to(
            torch.float32
        )

        model.vq_model = model.vq_model.to(torch.float32)

        return model, proc

    codebook_size = 256

    #    for codebook_size in codebook_sizes:
    #        for heads in head_numbers:
    for learning_rate in [5e-4]:
        for patch_size in [8]:
            heads = patch_size * 2
            autoencoder, processor = get_model(codebook_size, heads, patch_size)

            ds_train = load_and_transform_dataset(
                image_dataset_path_or_url,
                processor,
            )

            run_d = dict(
                codebook_size=codebook_size,
                heads=heads,
                mlp_dim=mlp_dim,
                image_channels=image_channels,
                dct_compression_factor=dct_compression_factor,
                max_n_patches=max_n_patches,
                depth=depth,
                bits=heads * math.log2(codebook_size),
                feature_dim=feature_channels,
                patch_size=patch_size,
                learning_rate=learning_rate,
            )

            print("starting run: ", run_d)

            if use_wandb:
                wandb.init(project="vq-experiments", config=run_d)

            autoencoder = train(
                autoencoder,
                ds_train,
                device=device,
                dtype=dtype,
                batch_size=batch_size,
                use_wandb=use_wandb,
                train_norm_iters=train_norm_iters,
            )

            autoencoder = None

            if use_wandb:
                wandb.finish()


if __name__ == "__main__":
    import jsonargparse

    jsonargparse.CLI(main)
