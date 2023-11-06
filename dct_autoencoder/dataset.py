from typing import List, Tuple
from torchvision import transforms
import webdataset as wds
import json

from .feature_extraction_dct_autoencoder import DCTAutoencoderFeatureExtractor

def tuple_collate(x: List[Tuple]):
    assert len(x) > 0
    num_columns = len(x[0])
    lists = [list() for _ in range(num_columns)]
    for row in x:
        for i, col in enumerate(row):
            lists[i].append(col)
    return lists


def load_preprocessed_dataset(
    dataset_url: str,
    ):
    dataset = wds.WebDataset(dataset_url, handler=wds.handlers.warn_and_continue) \
            .decode(partial=True) \
            .to_tuple("patches.pth", "positions.pth", "channels.pth", "original_size.pyd", "patch_size.pyd")
    return dataset

def load_and_transform_dataset(
    dataset_url: str,
    dct_processor: DCTAutoencoderFeatureExtractor,
    device="cpu",
) -> wds.WebDataset:
    """
    returns a dataset that is an iterable over whatever the
    dct_processor.preprocess returns
    """
    min_res = dct_processor.patch_size

    def filter_res(x):
        h, w = x["json"]["height"], x["json"]["width"]
        if h is None or w is None:
            return False
        return False if h < min_res or w < min_res else True

    # some buffer room, gives you better dct features to use bigger images
    max_size = max(
        dct_processor.patch_size
        * max(dct_processor.max_patch_w, dct_processor.max_patch_h),
        768,
    )

    def crop(pixel_values):
        pixel_values = pixel_values.to(device)
        _, h, w = pixel_values.shape
        if max(h, w) > max_size:
            ar = h / w
            if h > w:
                h = max_size
                w = int(h / ar)
            else:
                w = max_size
                h = int(ar * w)

            rz = transforms.Resize(min(h, w))
            pixel_values = rz(pixel_values)
        return pixel_values

    ds = (
        wds.WebDataset(dataset_url, handler=wds.handlers.warn_and_continue, verbose=True, cache_dir=None, detshuffle=True,)
        .map_dict(json=json.loads, handler=wds.handlers.warn_and_continue)
        .select(
            filter_res,
        )
        .decode("torchrgb", partial=True, handler=wds.handlers.warn_and_continue)
        .rename(pixel_values="jpg", handler=wds.handlers.warn_and_continue)
        .map_dict(pixel_values=crop)
        .map(lambda d: dct_processor.preprocess(d['pixel_values']))#, handler=wds.handlers.warn_and_continue)
    )

    return ds
