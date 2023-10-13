from typing import List
from tqdm import tqdm
import torch
import vector_quantize_pytorch
import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms.v2 as transforms
import torchvision
import os
import math
import time

from dct_preproc import DCTPreprocessor
from dct_autoenc import DCTAutoencoderTransformer
import wandb

OUTDIR = f"out/{time.ctime()}/"

os.makedirs(OUTDIR, exist_ok=True)

# https://github.com/pytorch/pytorch/issues/973
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


def get_dct_preproc(vq_autoencoder:DCTAutoencoderTransformer):
    return DCTPreprocessor(image_channels=vq_autoencoder.image_channels, patch_size=vq_autoencoder.patch_size, dct_compression_factor=vq_autoencoder.dct_compression_factor, max_n_patches=vq_autoencoder.max_n_patches)

def make_image_grid(x, x_hat, filename=None, n: int = 10, max_size:int=1024):
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

    ims = [F.pad(im, (w-im.shape[2], 0, h-im.shape[1], 0)) for im in ims]

    im = torchvision.utils.make_grid(ims, 2, normalize=True, scale_each=True)
    im = transforms.functional.to_pil_image(im.cpu().to(torch.float32))
    if filename:
        im.save(filename)

    return im

def get_collate(dct_preproc: DCTPreprocessor):
    def dict_collate(x: List[dict]):
        o = {}
        for d in x:
            for k,v in d.items():
                l = o.get(k)
                if l is not None:
                    o[k].append(v)
                else:
                    o[k] = [v]
        dct_features, original_sizes = dct_preproc(o['pixel_values'])

        o['dct_features'] = dct_features
        o['original_sizes'] = original_sizes

        return o
    return dict_collate

def validate(
    vq_autoencoder,
    val_ds,
    batch_size: int = 32,
    device="cuda",
    dtype=torch.float16,
    use_wandb: bool = False,
):
    vq_autoencoder.eval()
    collate = get_collate(get_dct_preproc(vq_autoencoder))
    dataloader = DataLoader(val_ds, batch_size=batch_size, num_workers=0, collate_fn=collate)
    for i, batch in enumerate(tqdm(dataloader)):
        x = batch["pixel_values"]
        x = x.to(device)
        with torch.no_grad():
            with torch.autocast(device):
                out = vq_autoencoder(x, decode=True)
        if use_wandb:
            wandb.log(
                {
                    "val": dict(
                        step=i,
                        rec_loss=out["rec_loss"].item(),
                        commit_loss=out["commit_loss"].item(),
                        perplexity=out["perplexity"].item(),
                        pixel_loss=out["pixel_loss"].item(),
                    )
                }
            )

        if i == 0:
            image = make_image_grid(out['x'], out['x_hat'], filename=f"{OUTDIR}/val image {i:05}.png")
            if use_wandb:
                wandb.log({"val": dict(image=wandb.Image(image))})


def train(
    vq_autoencoder: DCTAutoencoderTransformer,
    train_ds,
    batch_size: int = 32,
    alpha: float = 1e-3,
    learning_rate=1e-3,
    epochs: int = 1,
    device="cuda",
    dtype=torch.float16,
    val_every=10,
    n_log:int=10,
    max_steps=1e20,
    warmup_steps=10,
    use_wandb: bool = False,
):
    optimizer = torch.optim.Adam(vq_autoencoder.parameters(), lr=1e-9)

    n_steps = 0

    dct_preproc = get_dct_preproc(vq_autoencoder)
    collate = get_collate(dct_preproc)

    for epoch_i, epoch in enumerate(range(epochs)):
        train_ds = train_ds.shuffle(seed=epoch_i + 42)
        dataloader = DataLoader(
                train_ds, batch_size=batch_size, num_workers=0, collate_fn=collate
        )
        for i, batch in enumerate(tqdm(dataloader)):
            if epoch_i == 0 and i < warmup_steps:
                for g in optimizer.param_groups:
                    g["lr"] = ((i + 1) / warmup_steps) * learning_rate

            optimizer.zero_grad()

            dct_features = batch["dct_features"]
            original_sizes = batch["original_sizes"]
            dct_features = [x.to(device) for x in dct_features]

            with torch.autocast(device, dtype=dtype):
                out = vq_autoencoder(dct_features=dct_features, original_sizes=original_sizes, decode=False)

            commit_loss = out["commit_loss"] * alpha

            loss = out["rec_loss"] + commit_loss

            loss.backward()
            optimizer.step()

            print(
                f"epoch: {epoch} loss: {loss.item():.3f} rec_loss: {out['rec_loss'].item():.2f} commit_loss: {commit_loss.item():.2f} perpelxity: {out['perplexity'].item():.2f}"
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

            if i % val_every == 0:
                print("logging images ....")
                vq_autoencoder.eval()
                with torch.no_grad():
                    batch = collate([train_ds[i] for i in range(n_log)])
                    dct_features, original_sizes = batch['dct_features'], batch['original_sizes']
                    dct_features = [x.to(device).to(dtype) for x in dct_features]
                    with torch.autocast(device):
                        out = vq_autoencoder(dct_features=dct_features, original_sizes=original_sizes)
                vq_autoencoder.train()
                image = make_image_grid(out['x'], out['x_hat'], filename=f"{OUTDIR}/train image {i:04}.png")
                if use_wandb:
                    wandb.log(
                        {"train": dict(epoch=epoch, step=i, image=wandb.Image(image), pixel_loss=out['pixel_loss'], rec_loss=out['rec_loss'])}
                    )

            if n_steps > max_steps:
                return vq_autoencoder

            n_steps += 1

    return vq_autoencoder

def load_and_transform_dataset(
    dataset_name_or_url: str,
    split: str,
    min_res:int,
):
    ds = datasets.load_dataset(dataset_name_or_url, split=split)
    ds = ds.filter(lambda x: x['image'].size[0] > min_res and x['image'].size[1] > min_res)

    def f(examples):
        # norm parameters taken from clip
        _transforms = transforms.Compose(
            [
                #transforms.RandomResizedCrop((512,512)),
                transforms.ToImage(),
                transforms.ConvertImageDtype(torch.float32),
            ]
        )
        return {
            "pixel_values": [
                _transforms(image.convert("RGB")) for image in examples["image"]
            ]
        }

    ds.set_transform(f)
    return ds


def main(
    image_dataset_path_or_url="imagenet-1k",
    device="cuda",
    batch_size: int = 32,
    use_wandb: bool = False,
):

    feature_channels:int = 768
    dct_compression_factor=0.80

    ds_train = load_and_transform_dataset(
        image_dataset_path_or_url, split="train", min_res=32,
    )
    ds_test = load_and_transform_dataset(
        image_dataset_path_or_url, split="test", min_res=32,
    )
    dtype = torch.float16

    def get_vq_autoencoder(codebook_size, heads):
        vq_model = vector_quantize_pytorch.VectorQuantize(
            feature_channels,
            codebook_size=codebook_size,
            codebook_dim=64,
            threshold_ema_dead_code=10,
            heads=heads,
            channel_last=False,
            accept_image_fmap=True,
            kmeans_init=True,
            separate_codebook_per_head=False,
            sample_codebook_temp=1.0,
            decay=0.90,
        ).to(device)

        return DCTAutoencoderTransformer(
            vq_model,
            feature_channels=feature_channels,
            patch_size=patch_size,
            dct_compression_factor=dct_compression_factor,
        ).to(device)

    codebook_size = 256

    heads = 16


#    for codebook_size in codebook_sizes:
#        for heads in head_numbers:
    for learning_rate in [1e-3, 3e-3]:
        for patch_size in [32, 16]:
            vq_autoencoder = get_vq_autoencoder(codebook_size, heads)

            run_d = dict(
                codebook_size=codebook_size,
                heads=heads,
                bits=heads * math.log2(codebook_size),
                feature_dim=feature_channels,
                patch_size=patch_size,
                learning_rate=learning_rate,
            )

            print("starting run: ", run_d)

            if use_wandb:
                run = wandb.init(project="vq-experiments", config=run_d)

            vq_autoencoder = train(
                vq_autoencoder,
                ds_train,
                device=device,
                dtype=dtype,
                batch_size=batch_size,
                use_wandb=use_wandb,
            )
            validate(
                vq_autoencoder,
                ds_test,
                device=device,
                dtype=dtype,
                batch_size=batch_size,
                use_wandb=use_wandb,
            )

            vq_autoencoder = None

            if use_wandb:
                wandb.finish()


if __name__ == "__main__":
    import jsonargparse

    jsonargparse.CLI(main)
