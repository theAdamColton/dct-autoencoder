import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torch
import vector_quantize_pytorch
import datasets
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as transforms
import os

from autoencoder import Encoder, Decoder, VQAutoencoder
import matplotlib.pyplot as plt

IMAGEDIR="out/"

os.makedirs(IMAGEDIR, exist_ok=True)



def log_images(vq_autoencoder, x, n:int=10, filename:str = f"{IMAGEDIR}/reconstructed.png"):
    with torch.no_grad():
        out = vq_autoencoder(x)
    f, axarr = plt.subplots(n,2)
    unnormalize_f = unnormalize()
    for i in range(n):
        im = x[i].clamp(0, 1).cpu()
        im_hat = out['x_hat'][i].clamp(0,1).cpu()
        axarr[i,0].imshow(im.permute(1,2,0))
        axarr[i,1].imshow(im_hat.permute(1,2,0))
    plt.show()
    plt.savefig(filename)

        

def train(vq_autoencoder: VQAutoencoder, train_ds, batch_size:int = 256, alpha: float = 1e-5, learning_rate=5e-3, epochs: int = 10, device='cuda'):
    dataloader = DataLoader(train_ds, batch_size=batch_size, drop_last=True,)
    optimizer = torch.optim.AdamW(vq_autoencoder.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        for batch in tqdm(dataloader):
            optimizer.zero_grad()
            x = batch['pixel_values']

            with torch.autocast(device):
                out = vq_autoencoder(x)

            commit_loss = out['commit_loss'] * alpha

            loss = out['rec_loss'] + commit_loss

            loss.backward()
            optimizer.step()

            log_images(vq_autoencoder, x)

            print(f"epoch: {epoch} loss: {loss.item():.3f} rec_loss: {out['rec_loss'].item():.2f} commit_loss: {commit_loss.item():.2f} perpelxity: {out['perplexity'].item():.2f}")

    return vq_autoencoder


def unnormalize():
    return transforms.Compose([ transforms.Normalize(( 0., 0., 0. ),(1/0.26862954,1/0.26130258,1/0.27577711)),
                                transforms.Normalize((0.48145466,0.4578275,0.40821073) ,( 1., 1., 1. )),
                               ])

def load_and_transform_dataset(dataset_name_or_url: str, split:str, image_channels: int = 3, height:int=128, width:int=128, device='cuda'):
    ds = datasets.load_dataset(dataset_name_or_url, split=split)
    def f(examples):
        # norm parameters taken from clip
        _transforms = transforms.Compose([transforms.CenterCrop((height, width)), transforms.ToImageTensor(), transforms.ConvertImageDtype(torch.float32), transforms.Normalize((0.48145466,0.4578275,0.40821073), (0.26862954,0.26130258,0.27577711))])
        examples["pixel_values"] = [_transforms(image.convert("RGB")) for image in examples["image"]]
        return examples

    ds = ds.map(f, remove_columns=['image'], batched=True).with_format('torch', device=device)
    pixel_features = datasets.Array3D(shape=(image_channels, height, width), dtype='float32')
    ds.features['pixel_values'] = pixel_features
    ds = ds.shuffle()
    return ds


def main(image_dataset_path_or_url="imagenet-1k", device='cuda'):
    ds_train = load_and_transform_dataset(image_dataset_path_or_url, split='train', device=device)
    ds_test = load_and_transform_dataset(image_dataset_path_or_url, split='test', device=device)
    dtype = torch.float16

    codebook_sizes = [2**i for i in range(5, 14, 2)]

    image_channels = 3
    input_channels = image_channels * 2 # 3 real number channels, 3 imaginary number channels

    def get_vq_autoencoder():
        encoder = Encoder(input_channels).to(device)
        decoder = Decoder(input_channels).to(device)
        vq_model = vector_quantize_pytorch.VectorQuantize(64, codebook_size=codebook_size, threshold_ema_dead_code=2, channel_last=False, accept_image_fmap=True, kmeans_init=True).to(device)
        vq_autoencoder = VQAutoencoder(vq_model, encoder, decoder)

        return vq_autoencoder

    for codebook_size in codebook_sizes:
        vq_autoencoder = get_vq_autoencoder()
        vq_autoencoder = train(vq_autoencoder, ds_train, device=device)

if __name__ == "__main__":
    import jsonargparse
    jsonargparse.CLI(main)
