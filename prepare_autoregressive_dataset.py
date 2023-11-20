"""
take a pretrained dct autoencoder and make a 
dataset for training an autoregressive model
"""
import torch
import datasets
from torchvision.transforms.functional import pil_to_tensor

from dct_autoencoder import DCTAutoencoder, DCTAutoencoderFeatureExtractor
from dct_autoencoder.dataset import dict_collate, tuple_collate
from dct_autoencoder.factory import get_model_and_processor
from dct_autoencoder import dct_patches

def main(
    # image_dataset_path_or_url refers to an unprocessed dataset of images and texts
    image_dataset_path_or_url: str = "laion/dalle-3-dataset",

    caption_column: str = 'caption',
    image_column: str = 'image',

    dtype = torch.float16,
    device="cuda",
    torch_compile = True,

    model_load_path:str = None,

    batch_size: int = 32,

    dataset_path: str = "laion/dalle-3-dataset",
        ):

    autoencoder, processor= get_model_and_processor(None, device, torch.float16, sample_patches_beta = 0.01, resume_path = model_load_path)


    def preproc(row):
        image = pil_to_tensor(row[image_column])/255

        preprocess_res = processor.preprocess(image)

        row.update(preprocess_res)
        return row

    ds = datasets.load_dataset(dataset_path, streaming=True)['train'].map(preproc)

    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, collate_fn=dict_collate)

    batch_iterator = processor.iter_batches(iter(dl), batch_size = batch_size)

    output_data = []

    for i, batch in enumerate(batch_iterator):
        captions = batch._data['caption']

        batch = batch.to(device)
        batch.patches = batch.patches.to(dtype)

        with torch.inference_mode():
            outputs = autoencoder(batch)
        dict_data = dct_patches.to_dict(outputs['dct_patches'], outputs['codes'])
        for caption, data in zip(captions, dict_data):
            output_data.append(
                    {
                        "caption": caption,
                        "image_embedding": data,
                    }
                )

    print("saving.....")
    dataset = datasets.Dataset.from_list(output_data)
    dataset.save_to_disk("./out/tokenized_dataset/")
        


if __name__ == "__main__":
    import jsonargparse
    jsonargparse.CLI(main)
