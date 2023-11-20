from torch.cuda.amp import GradScaler
from typing import Callable, List, Optional
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
import os
import time
import wandb
import random


from dct_autoencoder.feature_extraction_dct_autoencoder import DCTAutoencoderFeatureExtractor
from dct_autoencoder.patchnorm import PatchNorm
from dct_autoencoder.util import calculate_perplexity, image_clip
from dct_autoencoder.factory import get_model_and_processor
from dct_autoencoder.dataset import dict_collate, load_and_transform_dataset, load_preprocessed_dataset, tuple_collate
from dct_autoencoder.dct_patches import DCTPatches, slice_dctpatches
from dct_autoencoder.modeling_dct_autoencoder import DCTAutoencoder
from dct_autoencoder.configuration_dct_autoencoder import DCTAutoencoderConfig


import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))


OUTDIR = f"out/{time.ctime()}/"

os.makedirs(OUTDIR, exist_ok=True)


def get_loss_dict(d):
    o = {}
    for k, v in d.items():
        if "loss" in k:
            if isinstance(v, torch.Tensor):
                o[k] = v.item()
            else:
                o[k] = v
    return o


def get_decay_fn(start_val: float, end_value: float, n: int):
    def fn(i: int):
        if i > n:
            return end_value
        return ((n - i) / n) * start_val + (i / n) * end_value

    return fn


def make_image_grid(x, x_hat, filename=None, n: int = 10, max_size: int = 1024):
    ims = []

    n = min(len(x), n)

    for i in range(n):
        im = x[i]
        im_hat = x_hat[i]
        ims.append(im)
        ims.append(im_hat)

    ims = [
        transforms.Resize(
            384, max_size=max_size, interpolation=transforms.InterpolationMode.BICUBIC
        )(im)
        for im in ims
    ]

    ims = [image_clip(im) for im in ims]
    ims = [im.clamp(0.0, 1.0) for im in ims]

    sizes = [im.shape for im in ims]
    h = max([s[1] for s in sizes])
    w = max([s[2] for s in sizes])

    ims = [F.pad(im, (w - im.shape[2], 0, h - im.shape[1], 0)) for im in ims]

    im = torchvision.utils.make_grid(ims, 2, normalize=False, scale_each=False)
    im = transforms.functional.to_pil_image(im.cpu().float())

    if filename:
        im.save(filename)
        print("saved ", filename)

    return im


def get_loss_weight_in_patch(h, w, decay: float, device, dtype, eps=1e-2):
    _arange_h = torch.arange(h, device=device, dtype=dtype)
    _arange_w = torch.arange(w, device=device, dtype=dtype)
    _h_ind, _w_ind = torch.meshgrid(_arange_h, _arange_w, indexing="ij")
    in_patch_dist = _h_ind + _w_ind
    # exponential decay
    weight = torch.exp(-decay * in_patch_dist) + eps
    return weight


def get_loss_weight_between_patches(h_positions, w_positions, decay, eps=1e-2):
    weight = h_positions + w_positions
    # exponential decay
    weight = torch.exp(-decay * weight) + eps
    return weight


def weighted_mse_loss(x, y, patch_pixel_weights, patch_position_weights):
    loss = (x - y) ** 2
    loss = (loss * patch_pixel_weights.unsqueeze(0)).mean()
    loss = (loss * patch_position_weights).mean()
    return loss


def train_step(
    batch: DCTPatches,
    autoencoder: DCTAutoencoder,
    proc: DCTAutoencoderFeatureExtractor,
    max_batch_n=None,
    decode_pixels=False,
):
    assert autoencoder.patchnorm.frozen
    if max_batch_n is not None:
        batch = slice_dctpatches(batch, max_batch_n)[0]

    # the batch is normalized
    batch = autoencoder._normalize(batch)

    # runs the autoencoder
    res = autoencoder(batch.shallow_copy())

    output_patches = res["dct_patches"]

    # gets the mask, 1s where the sequences wasn't padded
    mask = ~output_patches.key_pad_mask

    # normalized loss
    # l1 is used because dct features are usually considered laplacian distributed
    res["rec_loss"] = F.l1_loss(output_patches.patches[mask], batch.patches[mask])

    # l1 loss between unnormalized features
    _b = autoencoder.patchnorm.b[
        output_patches.patch_channels,
        output_patches.h_indices,
        output_patches.w_indices,
    ][mask]
    res["rec_loss_unnormalized"] = (
        _b * (output_patches.patches[mask] - batch.patches[mask])
    ).abs().mean()

    with torch.no_grad():
        res["perplexity"] = calculate_perplexity(
            res["codes"][mask], autoencoder.config.vq_codebook_size
        )

    if decode_pixels:
        # inverse normalization for input patches
        batch = autoencoder._inv_normalize(batch)
        # inverse normalization for output patches
        output_patches = autoencoder._inv_normalize(output_patches)

        # un-patchifies the patches, and converts from dct to pixels
        images_hat = proc.postprocess(output_patches)
        images = proc.postprocess(batch)

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


def train_patch_norm(
    patch_norm: PatchNorm,
    proc: DCTAutoencoderFeatureExtractor,
    train_ds,
    dtype,
    device,
    batch_size: int = 32,
    steps: int = 20,
    rng=None,
):
    train_ds = train_ds.shuffle(100000, rng=rng)
    dataloader = DataLoader(
        train_ds, batch_size=batch_size, num_workers=0, collate_fn=dict_collate
    )

    patch_norm = patch_norm.to(torch.float32).to(device).train()
    patch_norm.frozen = False

    for i, batch in enumerate(tqdm(proc.iter_batches(iter(dataloader), batch_size))):
        if i + 1 > steps:
            break
        batch = batch.to(device)
        batch.patches = batch.patches.to(torch.float32)
        normalized_patches = patch_norm.forward(batch)
        median = normalized_patches[~batch.key_pad_mask].median(0).values.mean().item()
        std = normalized_patches[~batch.key_pad_mask].std().item()
        _min = normalized_patches[~batch.key_pad_mask].min().item()
        _max = normalized_patches[~batch.key_pad_mask].max().item()
        print(
            f"{i+1:03} median {median:.2f} std {std:.2f} min {_min:.2f} max {_max:.2f}"
        )

    patch_norm.frozen = True
    patch_norm = patch_norm.to(dtype)
    return patch_norm


def __set_lr(optimizer, lr):
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
    max_steps: int = 1000000,
    num_workers: int = 0,
    warmup_steps=100,
    use_wandb: bool = False,
    vq_callables: List[Callable[[int], dict]] = [],
    rng=None,
    grad_accumulation_steps: int = 1,
    save_every=1000,
    use_pixel_loss=False,
    loss_weight={},
):
    if optimizer is None:
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)

    grad_scaler = GradScaler()

    # ----------- Model Training --------------
    for epoch_i, epoch in enumerate(range(epochs)):
        train_ds = train_ds.shuffle(100000, rng=rng)
        dataloader = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=dict_collate,
        )
        # list of columns, or None
        for i, batch in enumerate(
            tqdm(proc.iter_batches(iter(dataloader), batch_size))
        ):
            # ----- LR Warmup -----
            if epoch_i == 0 and i < warmup_steps:
                _lr = ((i + 1) / warmup_steps) * learning_rate
                __set_lr(optimizer, _lr)

            batch = batch.to(device)
            #batch.patches = batch.patches.to(dtype)

            seq_len = batch.patches.shape[1]
            batch_len = batch.patches.shape[0]

            if i % log_every == 0:
                print("logging images ....")
                autoencoder.eval()
                with torch.no_grad():
                    with torch.autocast(device_type=device, dtype=dtype):
                        out = train_step(
                            batch,
                            autoencoder,
                            proc,
                            max_batch_n=n_log,
                            decode_pixels=True,
                        )
                autoencoder.train()
                image = make_image_grid(
                    out["x"],
                    out["x_hat"],
                    filename=f"{OUTDIR}/train image {i:04}.png",
                    n=n_log,
                )

                image = wandb.Image(image)
                out = None
            else:
                image = None

            optimizer.zero_grad()

            with torch.autocast(device_type=device, dtype=dtype):
                out = train_step(batch, autoencoder, proc, decode_pixels=use_pixel_loss)

            loss = 0.0
            for k, v in out.items():
                if "loss" in k:
                    if k in loss_weight:
                        weight = loss_weight[k]
                    else:
                        weight = 1.0
                    loss += v * weight

            grad_scaler.scale(loss / grad_accumulation_steps).backward()

            if (i + 1) % grad_accumulation_steps == 0:
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), 1.0)
                grad_scaler.step(optimizer)
                grad_scaler.update()

            log_dict = dict(
                epoch=epoch,
                step=i,
                n_images_in_batch=len(batch.original_sizes),
                perplexity=out["perplexity"].item(),
                batch_len=batch_len,
                seq_len=seq_len,
                loss=loss.item(),
                **get_loss_dict(out),
            )
            if image:
                log_dict["image"] = image

            for fn in vq_callables:
                d = fn(i)
                for k, v in d.items():
                    autoencoder.vq_model.__dict__[k] = v
                    log_dict[k] = v

            print(log_dict)
            if use_wandb:
                wandb.log(
                    {"train": log_dict},
                )

            if i > max_steps:
                break

            if i % save_every == 0 and i > 0:
                autoencoder.save_pretrained(OUTDIR + "/model/")

    return autoencoder, optimizer


def main(
    # specify one of the below two
    # image_dataset_path_or_url refers to an unprocessed dataset
    image_dataset_path_or_url: str = None,
    # preprocessed_dataset_path_or_url refers to a dataset preprocessed using
    # ./preproc_dataset.py
    preprocessed_dataset_path_or_url: str = None,

    model_config_path="./conf/patch16L.json",
    resume_path: Optional[str] = None,
    device="cuda",
    autocast_dtype:str="float16",
    batch_size: int = 30,
    num_workers: int = 0,
    use_wandb: bool = False,
    train_norm_iters: int = 10,
    max_iters: int = 10000,
    # This only applies if you are not using a preprocessed dataset
    sample_patches_beta: float = 0.02,
    commitment_loss_weight_start: float = 1e-1,
    commitment_loss_weight_end: float = 1e-9,
    lfq_entropy_loss_start: float = 5e1,
    lfq_entropy_loss_end: float = 1e-1,
    learning_rate: float = 1e-4,
    # dict of k,v where k is the name of some loss term returned by train_step
    loss_weight: dict[str, float] = {},
    seed: int = 42,
    log_every: int = 200,
    grad_accumulation_steps: int = 1,

    torch_compile:bool=False,
):
    model_config: DCTAutoencoderConfig = DCTAutoencoderConfig.from_json_file(
        model_config_path
    )

    random.seed(seed)

    if autocast_dtype == "float32":
        dtype = torch.float32
    elif autocast_dtype == "float16":
        dtype = torch.float16
    elif autocast_dtype == "bfloat16":
        dtype = torch.bfloat16
    else: raise ValueError(autocast_dtype)

    autoencoder, processor = get_model_and_processor(
        model_config, device, dtype, sample_patches_beta, resume_path
    )
    autoencoder = autoencoder.to(torch.float32)

    if torch_compile:
        autoencoder = torch.compile(autoencoder)

    max_seq_len = processor.max_seq_len

    autoencoder.vq_model.entropy_loss_weight = lfq_entropy_loss_start
    autoencoder.vq_model.commitment_loss_weight = commitment_loss_weight_start

    warmup_steps = 100

    commitment_loss_weight_n = max_iters
    commitment_loss_weight_fn = get_decay_fn(
        commitment_loss_weight_start,
        commitment_loss_weight_end,
        commitment_loss_weight_n,
    )

    lfq_entropy_loss_n = max_iters
    lfq_decay_fn = get_decay_fn(
        lfq_entropy_loss_start, lfq_entropy_loss_end, lfq_entropy_loss_n
    )

    vq_callables = [
        lambda i: {
            "entropy_loss_weight": lfq_decay_fn(i),
            "commitment_loss_weight": commitment_loss_weight_fn(i),
        }
    ]

    if image_dataset_path_or_url is not None:
        train_ds = load_and_transform_dataset(
            image_dataset_path_or_url,
            processor,
            device=device if num_workers == 0 else "cpu",
        )
    else:
        train_ds = load_preprocessed_dataset(preprocessed_dataset_path_or_url)

    n_params = 0
    for n, p in autoencoder.named_parameters():
        if "patchnorm" not in n:
            n_params += p.nelement()

    run_d = dict(
        sample_patches_beta=sample_patches_beta,
        max_seq_len=processor.max_seq_len,
        learning_rate=learning_rate,
        commitment_loss_weight_start=commitment_loss_weight_start,
        commitment_loss_weight_end=commitment_loss_weight_end,
        commitment_loss_weight_n=commitment_loss_weight_n,
        n_params=n_params,
        warmup_steps=warmup_steps,
        lfq_entropy_loss_start=lfq_entropy_loss_start,
        lfq_entropy_loss_end=lfq_entropy_loss_end,
        lfq_entropy_loss_n=lfq_entropy_loss_n,
        feature_dim=model_config.feature_dim,
        n_attention_heads=model_config.n_attention_heads,
        grad_accumulation_steps=grad_accumulation_steps,
        patch_size=model_config.patch_size,
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
            autoencoder.patchnorm = PatchNorm(
                model_config.max_patch_h,
                model_config.max_patch_w,
                model_config.patch_size,
                model_config.image_channels,
            ).train()

        print("training norm")
        processor.sample_patches_beta = 0.0
        processor.max_seq_len = model_config.max_patch_h * model_config.max_patch_w * 3
        autoencoder.patchnorm = train_patch_norm(
            autoencoder.patchnorm,
            processor,
            train_ds,
            dtype,
            device,
            batch_size=min(batch_size, 32),
            steps=train_norm_iters,
            rng=rng,
        )
        processor.max_seq_len = max_seq_len
        processor.sample_patches_beta = sample_patches_beta
        print("done training norm")
    else:
        autoencoder.patchnorm.frozen = True


    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-9)

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
        log_every=log_every,
        grad_accumulation_steps=grad_accumulation_steps,
    )

    autoencoder.save_pretrained(OUTDIR + "/model/")
    print("done with all training")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    import jsonargparse

    jsonargparse.CLI(main)
