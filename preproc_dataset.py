"""
The feature extraction pipeline can be run as a preprocessing step for a
dataset. In practice this allows you to spend some upfront compute cost 
in order to get much faster training times.

What is computed in the up-front preprocessing step:
    1.) load RGB pixels
    2.) convert to IPT
    3.) crop to multiple of patch size
    4.) DCT transform
    5.) patchify

There are some variables which can't be adjusted between preprocessing datasets:
    You can't change the sample beta parameter
    You can't change the patch size
    You can't change the channel weights or magnitude weight
"""
import torch
import random
import webdataset as wds
import os
from tqdm import tqdm

from main import get_model_and_processor, load_and_transform_dataset
from dct_autoencoder.configuration_dct_autoencoder import DCTAutoencoderConfig
from dct_autoencoder.factory import get_model_and_processor


def main(
    image_dataset_path_or_url: str = None,
    model_config_path="./conf/patch16L.json",
    output_dir: str = None,
    resume_path: str = None,
    device="cpu",
    dtype=torch.bfloat16,
    sample_patches_beta: float = 0.02,
    # one million
    n: int = 1000000,
):
    model_config: DCTAutoencoderConfig = DCTAutoencoderConfig.from_json_file(model_config_path)

    if device == "cuda":
        print("Warning! CUDA FFT is known to have memory leak issues! https://github.com/pytorch/pytorch/issues/94893")

    model, processor = get_model_and_processor(
        model_config, device, dtype, sample_patches_beta, resume_path
    )
    del model

    dataset = load_and_transform_dataset(image_dataset_path_or_url, processor, device=device)
    dataset = dataset.with_length(n)

    os.makedirs(output_dir, exist_ok=True)
    with wds.ShardWriter(output_dir + "/%06d.tar", maxsize=1e9, compress=True) as shardwriter:
        for i, data in tqdm(enumerate(dataset), total=n):
            patches, positions, channels, original_size, patch_size = data

            patches = patches.cpu()
            positions = positions.cpu()
            channels = channels.cpu()

            shardwriter.write(
                    {
                        "__key__": f"{i:08}",
                        "patches.pth": patches,
                        "positions.pth":positions,
                        "channels.pth":channels,
                        "original_size.pyd":original_size,
                        "patch_size.pyd":patch_size,
                        })

if __name__ == "__main__":
    import jsonargparse

    jsonargparse.CLI(main)
