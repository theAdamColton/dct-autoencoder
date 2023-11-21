import torch
import math
from typing import Optional

from .configuration_dct_autoencoder import DCTAutoencoderConfig
from .modeling_dct_autoencoder import DCTAutoencoder
from .feature_extraction_dct_autoencoder import DCTAutoencoderFeatureExtractor
from .util import power_of_two


def get_max_seq_length(
    model_config: DCTAutoencoderConfig, sample_patches_beta: float, cdf_p=0.95
):
    # the max sequence lengths is based off of the exp dist
    # CDF at some probability

    # CDF: F(x;beta) = 1-e^(-beta*x)
    # x is the sequence length
    # we pick x s.t.
    # .9 = 1-e^(-beta*x)
    # - ln(0.1) / beta = x
    if sample_patches_beta <= 0:
        return  model_config.max_patch_h* model_config.max_patch_w* model_config.image_channels

    max_seq_len = round(-1 * math.log(1 - cdf_p) / sample_patches_beta)
    max_seq_len = power_of_two(max_seq_len)
    max_seq_len = min(
        model_config.max_patch_h
        * model_config.max_patch_w
        * model_config.image_channels,
        max_seq_len,
    )
    return max_seq_len


def get_model_and_processor(
    model_config: Optional[DCTAutoencoderConfig] = None,
    device="cpu",
    dtype=torch.float32,
    sample_patches_beta: float = 0.02,
    resume_path: Optional[str]=None,
):


    if resume_path is not None:
        model:DCTAutoencoder = DCTAutoencoder.from_pretrained(resume_path, ignore_mismatched_sizes=True).to(dtype).to(device)
        model_config = model.config
        print("Loaded from ", resume_path)
    else:
        assert model_config is not None
        model = DCTAutoencoder(model_config).to(dtype).to(device)
    
    max_seq_len = get_max_seq_length(model_config, sample_patches_beta)

    proc = DCTAutoencoderFeatureExtractor(
        channels=model_config.image_channels,
        patch_size=model_config.patch_size,
        sample_patches_beta=sample_patches_beta,
        max_patch_h=model_config.max_patch_h,
        max_patch_w=model_config.max_patch_w,
        max_seq_len=max_seq_len,
    )

    return model, proc
