from typing import Dict, List, Tuple
from torchvision import transforms
import webdataset as wds
import json

from .feature_extraction_dct_autoencoder import DCTAutoencoderFeatureExtractor

def dict_collate(x: List[Dict]):
    assert len(x) > 0
    columns = x[0].keys()
    column_dicts = {k:list() for k in columns}
    for row in x:
        for k in columns:
            column_dicts[k].append(row[k])
    return column_dicts


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
            .map(lambda row: dict(patches=row["patches.pth"], positions=row["positions.pth"], channels=row["channels.pth"], original_sizes=row["original_size.pyd"], patch_sizes=row["patch_size.pyd"]), handler=wds.handlers.warn_and_continue)
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
    min_res = dct_processor.patch_size * 12

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

            rz = transforms.Resize(min(h, w), antialias=True)
            pixel_values = rz(pixel_values)
        return pixel_values

    handler = wds.handlers.warn_and_continue

    ds = (
        wds.WebDataset(dataset_url, handler=handler, verbose=True, cache_dir=None, detshuffle=True,)
        .map_dict(json=json.loads, handler=handler)
        .select(
            filter_res,
        )
        .decode("torchrgb", partial=True, handler=handler)
        .rename(pixel_values="jpg", handler=handler)
        .map_dict(pixel_values=crop)
        .map(lambda d: dct_processor.preprocess(d['pixel_values']), handler=handler)
    )

    return ds
