from typing import Callable, List, Optional
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from accelerate import Accelerator
from bitsandbytes.optim import PagedAdamW32bit
import wandb
import random
from transformers.optimization import get_cosine_schedule_with_warmup

from dct_autoencoder.feature_extraction_dct_autoencoder import (
    DCTAutoencoderFeatureExtractor,
)
from dct_autoencoder.patchnorm import PatchNorm
from dct_autoencoder.util import (
    calculate_perplexity,
    create_output_directory,
    make_image_grid,
    get_decay_fn,
)
from dct_autoencoder.factory import get_model_and_processor
from dct_autoencoder.dataset import (
    dict_collate,
    load_and_transform_dataset,
    load_preprocessed_dataset,
)
from dct_autoencoder.dct_patches import DCTPatches
from dct_autoencoder.modeling_dct_autoencoder import DCTAutoencoder
from dct_autoencoder.configuration_dct_autoencoder import DCTAutoencoderConfig


OUTDIR = create_output_directory()


def get_loss_dict(d):
    o = {}
    for k, v in d.items():
        if "loss" in k:
            assert isinstance(v, torch.Tensor)
            o[k] = float(round(v.item(), 5))
    return o


def step(
    batch: DCTPatches,
    autoencoder: DCTAutoencoder,
    proc: DCTAutoencoderFeatureExtractor,
    decode_pixels=False,
):
    assert autoencoder.patchnorm.frozen

    normalized_batch = autoencoder._normalize(batch.shallow_copy())

    res = autoencoder(normalized_batch.shallow_copy())

    output_patches = res["dct_patches"]

    # gets the mask, 1s where the sequences wasn't padded
    mask = ~output_patches.key_pad_mask

    # normalized loss
    # l1 is used because dct features are usually considered laplacian distributed
    rec_loss = F.l1_loss(output_patches.patches[mask], normalized_batch.patches[mask])

    with torch.no_grad():
        perplexity = calculate_perplexity(
            res["codes"][mask], autoencoder.config.vq_codebook_size
        )

    # inverse normalization for output patches
    output_patches = autoencoder._inv_normalize(output_patches)

#    # observation: We don't care about loss over the frequencies that are close
#    # to zero. We want the model to be more inclined to output zeros
#    low_freq_mask = batch.patches.abs() < 0.03
#    batch.patches[low_freq_mask] = 0.0

    rec_loss_unnormalized = F.l1_loss(
        output_patches.patches[mask], batch.patches[mask]
    )

    out = dict(
        rec_loss=rec_loss,
        rec_loss_unnormalized=rec_loss_unnormalized,
        perplexity=perplexity,
        commit_loss=res['commit_loss'],
    )

    if decode_pixels:
        # un-patchifies the patches, and converts from dct to pixels
        low_freq_mask = output_patches.patches.abs() < 0.025
        output_patches.patches[low_freq_mask] = 0.0
        images_hat = proc.postprocess(output_patches)
        images = proc.postprocess(batch)

        pixel_loss = 0.0

        for im, im_hat in zip(images, images_hat):
            # rgb space pixel loss
            pixel_loss += F.mse_loss(im, im_hat)

        pixel_loss = pixel_loss / len(images_hat)

        out["x_hat"] = images_hat
        out["pixel_loss"] = pixel_loss
        out["x"] = images

    return out


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
    train_ds = train_ds.shuffle(1000, rng=rng)
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


def train(
    autoencoder: DCTAutoencoder,
    proc: DCTAutoencoderFeatureExtractor,
    train_ds,
    optimizer,
    accelerator: Accelerator,
    scheduler=None,
    rng=None,
    batch_size: int = 32,
    epochs: int = 1,
    log_every=200,
    n_log: int = 10,
    max_steps: int = 1000000,
    num_workers: int = 0,
    vq_callables: List[Callable[[int], dict]] = [],
    save_every=1000,
    use_pixel_loss=False,
    loss_weight={},
):
    # ----------- Model Training --------------
    for epoch_i in range(epochs):
        train_ds = train_ds.shuffle(1000, rng=rng)
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
            with accelerator.accumulate(autoencoder):
                batch = batch.to(accelerator.device)
                batch.patches = batch.patches.to(autoencoder.dtype)
                seq_len = batch.patches.shape[1]
                batch_len = batch.patches.shape[0]

                if i % log_every == 0:
                    print("logging images ....")
                    autoencoder.eval()
                    with torch.inference_mode():
                        out = step(
                            batch,
                            autoencoder,
                            proc,
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

                out = step(batch, autoencoder, proc, decode_pixels=use_pixel_loss)

                loss = 0.0
                for k, v in out.items():
                    if "loss" in k:
                        if k in loss_weight:
                            weight = loss_weight[k]
                        else:
                            weight = 1.0
                        if weight != 0.0:
                            loss = loss + v * weight

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(autoencoder.parameters(), 1.0)
                    scheduler.step()

                optimizer.step()
                optimizer.zero_grad()

                log_dict = dict(
                    epoch=epoch_i,
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
                wandb.log(
                    {"train": log_dict},
                )

                if i > max_steps:
                    break

                if i % save_every == 0 and i > 0:
                    accelerator.save_state(OUTDIR + "/accelerator_state/")
                    autoencoder.save_pretrained(OUTDIR + "/model/")

    return autoencoder, optimizer


def main(
    # specify one of the below two
    # image_dataset_path_or_url refers to an unprocessed dataset
    image_dataset_path_or_url: str = None,
    # preprocessed_dataset_path_or_url refers to a dataset preprocessed using
    # ./preproc_dataset.py
    preprocessed_dataset_path_or_url: str = None,
    model_config_path="./conf/patch32-large.json",
    model_resume_path: Optional[str] = None,
    accelerator_resume_path: Optional[str] = None,
    device="cuda",
    autocast_dtype: str = "no",
    batch_size: int = 32,
    num_workers: int = 0,
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
    torch_compile: bool = False,
):
    model_config: DCTAutoencoderConfig = DCTAutoencoderConfig.from_json_file(
        model_config_path
    )

    random.seed(seed)

    if autocast_dtype == "fp16":
        dtype = torch.float16
    elif autocast_dtype == "bf16":
        dtype = torch.bfloat16
    elif autocast_dtype == "no":
        dtype = torch.float16
    else:
        raise ValueError(autocast_dtype)

    autoencoder, processor = get_model_and_processor(
        model_config, device, dtype, sample_patches_beta, model_resume_path
    )

    if autocast_dtype == "no":
        autoencoder = autoencoder.to(dtype)
    else:
        autoencoder = autoencoder.to(torch.float32)

    if torch_compile:
        # the forward method is patched,
        def _c(mod):
            mod.forward = torch.compile(
                mod.forward,
                fullgraph=True,
                dynamic=False,
            )

        _c(autoencoder)

    max_seq_len = processor.max_seq_len

    autoencoder.vq_model.entropy_loss_weight = lfq_entropy_loss_start
    autoencoder.vq_model.commitment_loss_weight = commitment_loss_weight_start

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
    for p in autoencoder.parameters():
        if p.requires_grad:
            n_params += p.nelement()

    run_d = dict(
        sample_patches_beta=sample_patches_beta,
        max_seq_len=processor.max_seq_len,
        learning_rate=learning_rate,
        commitment_loss_weight_start=commitment_loss_weight_start,
        commitment_loss_weight_end=commitment_loss_weight_end,
        commitment_loss_weight_n=commitment_loss_weight_n,
        n_params=n_params,
        lfq_entropy_loss_start=lfq_entropy_loss_start,
        lfq_entropy_loss_end=lfq_entropy_loss_end,
        lfq_entropy_loss_n=lfq_entropy_loss_n,
        encoder_config=model_config.encoder_config,
        decoder_config=model_config.decoder_config,
        grad_accumulation_steps=grad_accumulation_steps,
        patch_size=model_config.patch_size,
        vq_num_codebooks=model_config.vq_num_codebooks,
        vq_codebook_size=model_config.vq_codebook_size,
        **loss_weight,
    )

    print("starting run: ", run_d)
    wandb.init(project="vq-experiments", config=run_d)

    rng = random.Random(seed)

    # ----------- Norm Training ---------------
    if train_norm_iters > 0:
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
    autoencoder.patchnorm.frozen = True

    optimizer = PagedAdamW32bit(autoencoder.parameters(), lr=learning_rate, 
                                betas=(0.9, 0.99),
                                weight_decay=1e-1
                                )

    scheduler = get_cosine_schedule_with_warmup(optimizer, 400, max_iters)

    accelerator = Accelerator(
        gradient_accumulation_steps=grad_accumulation_steps,
        mixed_precision=autocast_dtype,
    )

    autoencoder, optimizer, scheduler = accelerator.prepare(autoencoder, optimizer, scheduler)

    if accelerator_resume_path is not None:
        print("loading accelerator state...")
        accelerator.load_state(accelerator_resume_path)

    autoencoder, optimizer = train(
        autoencoder,
        processor,
        train_ds,
        optimizer=optimizer,
        scheduler=scheduler,
        accelerator=accelerator,
        batch_size=batch_size,
        num_workers=num_workers,
        max_steps=max_iters,
        vq_callables=vq_callables,
        rng=rng,
        loss_weight=loss_weight,
        log_every=log_every,
    )

    autoencoder.save_pretrained(OUTDIR + "/model/")
    print("done with all training")

    wandb.finish()


if __name__ == "__main__":
    import jsonargparse
    jsonargparse.CLI(main)
